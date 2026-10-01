import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ==============================================================================
# BẢNG TRA CỨU TÊN TIẾNG VIỆT ĐẦY ĐỦ DẤU (51 LỚP BIỂN BÁO)
# ==============================================================================
VIETNAMESE_CLASSES = {
    'Bien bao ben xe bus': 'Biển báo bến xe bus',
    'Bien bao cam cac xe o to tai co khoi luong chuyen cho': 'Biển báo cấm ô tô tải theo tải trọng',
    'Bien bao cam di nguoc chieu': 'Biển báo cấm đi ngược chiều',
    'Bien bao cam di thang': 'Biển báo cấm đi thẳng',
    'Bien bao cam do xe': 'Biển báo cấm đỗ xe',
    'Bien bao cam dung xe va do xe': 'Biển báo cấm dừng và đỗ xe',
    'Bien bao cam nguoi di bo': 'Biển báo cấm người đi bộ',
    'Bien bao cam oto re trai va quay dau xe': 'Biển báo cấm ô tô rẽ trái & quay đầu',
    'Bien bao cam quay dau xe': 'Biển báo cấm quay đầu xe',
    'Bien bao cam re phai': 'Biển báo cấm rẽ phải',
    'Bien bao cam re phai va quay dau': 'Biển báo cấm rẽ phải & quay đầu',
    'Bien bao cam re trai': 'Biển báo cấm rẽ trái',
    'Bien bao cam re trai va quay dau': 'Biển báo cấm rẽ trái & quay đầu',
    'Bien bao cam vuot': 'Biển báo cấm vượt',
    'Bien bao cam xe may': 'Biển báo cấm xe máy',
    'Bien bao cam xe may re trai': 'Biển báo cấm xe máy rẽ trái',
    'Bien bao cam xe o to re phai': 'Biển báo cấm xe ô tô rẽ phải',
    'Bien bao cam xe o to re trai': 'Biển báo cấm xe ô tô rẽ trái',
    'Bien bao cam xe oto': 'Biển báo cấm xe ô tô',
    'Bien bao cam xe oto khach va xe oto tai': 'Biển báo cấm ô tô khách & ô tô tải',
    'Bien bao cam xe oto tai': 'Biển báo cấm xe ô tô tải',
    'Bien bao cam xe so-mi ro-mooc': 'Biển báo cấm xe sơ-mi rơ-moóc',
    'Bien bao chieu cao tinh khong thuc te': 'Biển báo chiều cao tĩnh không thực tế',
    'Bien bao cho ngoat nguy hiem': 'Biển báo chỗ ngoặt nguy hiểm',
    'Bien bao cho quay xe': 'Biển báo chỗ quay xe',
    'Bien bao cong truong': 'Biển báo công trường đang thi công',
    'Bien bao di cham': 'Biển báo đi chậm',
    'Bien bao doc xuong nguy hiem': 'Biển báo dốc xuống nguy hiểm',
    'Bien bao duong bi thu hep': 'Biển báo đường bị thu hẹp',
    'Bien bao duong cao toc phia truoc': 'Biển báo đường cao tốc phía trước',
    'Bien bao duong co go giam toc': 'Biển báo đường có gờ giảm tốc',
    'Bien bao duong giao nhau': 'Biển báo đường giao nhau',
    'Bien bao duong nguoi di bo cat ngang': 'Biển báo đường người đi bộ cắt ngang',
    'Bien bao giao nhau co tin hieu den': 'Biển giao nhau có tín hiệu đèn',
    'Bien bao giao nhau voi duong khong uu tien': 'Biển giao nhau với đường không ưu tiên',
    'Bien bao giao nhau voi duong uu tien': 'Biển giao nhau với đường ưu tiên',
    'Bien bao han che chieu cao': 'Biển báo hạn chế chiều cao',
    'Bien bao huong di phai theo': 'Biển báo hướng đi phải theo',
    'Bien bao huong di thang phai theo': 'Biển báo hướng đi thẳng phải theo',
    'Bien bao huong phai di vong chuong ngai vat': 'Biển báo đi vòng chướng ngại vật',
    'Bien bao lan duong danh cho o to con': 'Biển báo làn đường dành cho ô tô con',
    'Bien bao lan duong danh cho xe con va xe buyt': 'Biển báo làn xe con và xe buýt',
    'Bien bao lan duong danh cho xe may va xe 3 banh': 'Biển báo làn xe máy và xe 3 bánh',
    'Bien bao nguy hiem khac': 'Biển báo nguy hiểm khác',
    'Bien bao nhieu cho ngoat nguy hiem lien tiep': 'Biển nhiều chỗ ngoặt nguy hiểm liên tiếp',
    'Bien bao noi giao nhau chay theo vong xuyen': 'Biển báo giao nhau theo vòng xuyến',
    'Bien bao toc do toi da cho phep': 'Biển báo tốc độ tối đa cho phép',
    'Bien bao toc do toi thieu cho phep': 'Biển báo tốc độ tối thiểu cho phép',
    'Bien bao tre em': 'Biển báo khu vực có trẻ em',
    'Bien hieu lenh bat dau khu dong dan cu': 'Biển bắt đầu khu đông dân cư',
    'Bien hieu lenh het khu dong dan cu': 'Biển hết khu đông dân cư'
}

CLASS_ID_LIST = [
    'Biển báo bến xe bus', 'Biển báo cấm ô tô tải theo tải trọng', 'Biển báo cấm đi ngược chiều',
    'Biển báo cấm đi thẳng', 'Biển báo cấm đỗ xe', 'Biển báo cấm dừng và đỗ xe',
    'Biển báo cấm người đi bộ', 'Biển báo cấm ô tô rẽ trái & quay đầu', 'Biển báo cấm quay đầu xe',
    'Biển báo cấm rẽ phải', 'Biển báo cấm rẽ phải & quay đầu', 'Biển báo cấm rẽ trái',
    'Biển báo cấm rẽ trái & quay đầu', 'Biển báo cấm vượt', 'Biển báo cấm xe máy',
    'Biển báo cấm xe máy rẽ trái', 'Biển báo cấm xe ô tô rẽ phải', 'Biển báo cấm xe ô tô rẽ trái',
    'Biển báo cấm xe ô tô', 'Biển báo cấm ô tô khách & ô tô tải', 'Biển báo cấm xe ô tô tải',
    'Biển báo cấm xe sơ-mi rơ-moóc', 'Biển báo chiều cao tĩnh không thực tế', 'Biển báo chỗ ngoặt nguy hiểm',
    'Biển báo chỗ quay xe', 'Biển báo công trường đang thi công', 'Biển báo đi chậm',
    'Biển báo dốc xuống nguy hiểm', 'Biển báo đường bị thu hẹp', 'Biển báo đường cao tốc phía trước',
    'Biển báo đường có gờ giảm tốc', 'Biển báo đường giao nhau', 'Biển báo đường người đi bộ cắt ngang',
    'Biển giao nhau có tín hiệu đèn', 'Biển giao nhau với đường không ưu tiên', 'Biển giao nhau với đường ưu tiên',
    'Biển báo hạn chế chiều cao', 'Biển báo hướng đi phải theo', 'Biển báo hướng đi thẳng phải theo',
    'Biển báo đi vòng chướng ngại vật', 'Biển báo làn đường dành cho ô tô con', 'Biển báo làn xe con và xe buýt',
    'Biển báo làn xe máy và xe 3 bánh', 'Biển báo nguy hiểm khác', 'Biển nhiều chỗ ngoặt nguy hiểm liên tiếp',
    'Biển báo giao nhau theo vòng xuyến', 'Biển báo tốc độ tối đa cho phép', 'Biển báo tốc độ tối thiểu cho phép',
    'Biển báo khu vực có trẻ em', 'Biển bắt đầu khu đông dân cư', 'Biển hết khu đông dân cư'
]

def get_vietnamese_name(raw_name, cls_id=None):
    """Chuyển đổi tên lớp sang tiếng Việt chuẩn."""
    if raw_name in VIETNAMESE_CLASSES:
        return VIETNAMESE_CLASSES[raw_name]
    if cls_id is not None and 0 <= cls_id < len(CLASS_ID_LIST):
        return CLASS_ID_LIST[cls_id]
    return raw_name

def classify_sign_category(name):
    """
    Phân loại nhóm biển báo giao thông ADAS & gán mã màu:
    - CẤM: Đỏ neon (Neon Red)
    - CẢNH BÁO: Vàng hổ phách neon (Neon Amber)
    - HIỆU LỆNH: Xanh dương công nghệ (Electric Blue)
    - CHỈ DẪN: Xanh ngọc lục bảo (Emerald Cyan)
    """
    lower = name.lower()
    if 'cấm' in lower or 'cam' in lower:
        return 'BIỂN CẤM', (35, 45, 245), (255, 60, 60)
    elif any(k in lower for k in ['nguy hiểm', 'nguy hiem', 'giao nhau', 'dốc', 'doc', 'chậm', 'cham', 'công trường', 'cong truong', 'thu hẹp', 'thu hep', 'cao tốc', 'cao toc', 'gờ', 'go', 'trẻ em', 'tre em', 'hạn chế', 'han che']):
        return 'CẢNH BÁO', (0, 200, 255), (255, 210, 0)
    elif any(k in lower for k in ['hiệu lệnh', 'hieu lenh', 'phải theo', 'phai theo', 'vòng xuyến', 'vong xuyen', 'bắt đầu', 'bat dau', 'hết', 'het']):
        return 'HIỆU LỆNH', (255, 140, 0), (0, 180, 255)
    else:
        return 'CHỈ DẪN', (0, 230, 120), (0, 240, 150)

# ==============================================================================
# QUẢN LÝ FONT VÀ RENDER CHỮ CHUYÊN NGHIỆP
# ==============================================================================
def load_system_font(font_size=16, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf",
        "arial.ttf"
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT_HEADER = load_system_font(18, bold=True)
FONT_SECTION = load_system_font(14, bold=True)
FONT_BODY = load_system_font(15, bold=False)
FONT_BODY_BOLD = load_system_font(15, bold=True)
FONT_TAG = load_system_font(13, bold=True)
FONT_TELEMETRY = load_system_font(12, bold=False)

def wrap_text(text, font, max_width, draw):
    words = text.split()
    if not words:
        return []
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
    return lines

# ==============================================================================
# XỬ LÝ ẢNH SIÊU NÉT & ĐỒ HỌA HUD CYBERPUNK
# ==============================================================================
def enhance_sign_crop(crop_bgr, target_w, target_h):
    """Phóng to giữ nguyên tỉ lệ, nội suy Lanczos4 và Unsharp Mask làm sắc nét."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    ch, cw = crop_bgr.shape[:2]
    scale = min(target_w / cw, target_h / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    
    if nw > cw or nh > ch:
        resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Unsharp Masking
    gaussian = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(resized, 1.55, gaussian, -0.55, 0)
    
    # Khung canvas đen xám chuyên nghiệp
    canvas = np.full((target_h, target_w, 3), (22, 22, 26), dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = sharpened
    return canvas

def draw_corner_brackets(img, x1, y1, x2, y2, color, length=18, thickness=2):
    """Vẽ 4 góc ngắm quang học AI (HUD Target Reticles) thay vì ô chữ nhật đơn điệu."""
    length = min(length, (x2 - x1) // 3, (y2 - y1) // 3)
    if length <= 0:
        return
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

# ==============================================================================
# GIAO DIỆN HUD ADAS DASHBOARD CAO CẤP
# ==============================================================================
def draw_hud(frame, last_crop, last_sign_name, last_conf, last_bbox, fps, is_paused, detected_count, recent_history):
    h, w, _ = frame.shape

    # ---------------- 1. Top Header Glassmorphism Bar ----------------
    header_h = 50
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), (12, 14, 18), -1)
    cv2.addWeighted(header_overlay, 0.88, frame, 0.12, 0, frame)
    # Gradient accent line dưới header
    cv2.line(frame, (0, header_h - 1), (w, header_h - 1), (0, 215, 255), 1)

    # ---------------- 2. HUD Box Zoom Bên Phải ----------------
    hud_w = 340
    hud_h = 360
    margin = 16
    hud_x1 = w - hud_w - margin
    hud_y1 = header_h + margin
    hud_x2 = hud_x1 + hud_w
    hud_y2 = hud_y1 + hud_h

    # Card background mờ cao cấp
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (14, 16, 22), -1)
    cv2.addWeighted(hud_overlay, 0.85, frame, 0.15, 0, frame)

    # Khung viền HUD Box & Góc công nghệ
    cv2.rectangle(frame, (hud_x1, hud_y1), (hud_x2, hud_y2), (45, 52, 65), 1)
    draw_corner_brackets(frame, hud_x1, hud_y1, hud_x2, hud_y2, (0, 215, 255), length=16, thickness=2)

    # Vùng hiển thị ảnh Zoom
    crop_w = hud_w - 24
    crop_h = 180
    crop_x1 = hud_x1 + 12
    crop_y1 = hud_y1 + 42
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    cat_name, cat_color_bgr, cat_color_rgb = classify_sign_category(last_sign_name) if last_sign_name else ("", (100, 100, 100), (100, 100, 100))

    if last_crop is not None and last_crop.size > 0:
        enhanced = enhance_sign_crop(last_crop, crop_w, crop_h)
        if enhanced is not None:
            crop_py_limit = crop_y1 + crop_h
            crop_px_limit = crop_x1 + crop_w
            frame[crop_y1:crop_py_limit, crop_x1:crop_px_limit] = enhanced
            cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (60, 70, 85), 1)
            # 4 góc ngắm nhỏ bên trong ảnh zoom
            draw_corner_brackets(frame, crop_x1 + 2, crop_y1 + 2, crop_x2 - 2, crop_y2 - 2, (0, 255, 200), length=10, thickness=1)
            # Tâm ngắm (+)
            cx, cy = (crop_x1 + crop_x2) // 2, (crop_y1 + crop_y2) // 2
            cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 255, 200), 1)
            cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 255, 200), 1)
    else:
        cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (22, 24, 30), -1)
        cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (40, 48, 60), 1)

    # ---------------- 3. Bottom-Left History Stream (Góc dưới bên trái) ----------------
    if recent_history:
        hist_w = 360
        hist_h = 24 + len(recent_history[:3]) * 24
        hist_x1 = margin
        hist_y2 = h - margin
        hist_y1 = hist_y2 - hist_h

        hist_overlay = frame.copy()
        cv2.rectangle(hist_overlay, (hist_x1, hist_y1), (hist_x1 + hist_w, hist_y2), (12, 14, 18), -1)
        cv2.addWeighted(hist_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (hist_x1, hist_y1), (hist_x1 + hist_w, hist_y2), (40, 48, 60), 1)
        cv2.line(frame, (hist_x1, hist_y1 + 20), (hist_x1 + hist_w, hist_y1 + 20), (0, 215, 255), 1)

    # ---------------- 4. Vẽ Toàn bộ Chữ Tiếng Việt Bằng Pillow ----------------
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Header - Left: AI Status & Pulse
    draw.ellipse([16, 19, 26, 29], fill=(0, 255, 120))
    draw.text((34, 14), "ADAS AI VISION", font=FONT_HEADER, fill=(0, 255, 200))
    draw.text((200, 16), "|", font=FONT_BODY, fill=(80, 90, 105))
    
    # Header - Center: Telemetry
    fps_color = (0, 255, 120) if fps >= 20 else (255, 170, 0)
    draw.text((220, 16), f"FPS: {fps:.1f}", font=FONT_BODY_BOLD, fill=fps_color)
    draw.text((320, 16), f"Phát hiện: {detected_count} mục tiêu", font=FONT_BODY, fill=(230, 230, 230))
    
    # Header - Right: Status Controls Badge
    curr_time_str = datetime.now().strftime("%H:%M:%S")
    draw.text((w - 380, 16), curr_time_str, font=FONT_TELEMETRY, fill=(160, 175, 195))

    if is_paused:
        status_txt = "[TẠM DỪNG - SPACE: Tiếp tục]"
        status_clr = (255, 215, 0)
    else:
        status_txt = "[ĐANG QUÉT - SPACE: Dừng | Q: Thoát]"
        status_clr = (180, 220, 255)
    s_bbox = draw.textbbox((0, 0), status_txt, font=FONT_TAG)
    draw.text((w - (s_bbox[2] - s_bbox[0]) - 16, 16), status_txt, font=FONT_TAG, fill=status_clr)

    # HUD Box Header
    draw.text((hud_x1 + 14, hud_y1 + 10), "AI TARGET INSPECTION", font=FONT_SECTION, fill=(0, 220, 255))
    if last_sign_name:
        # Category Badge (Biển cấm / Cảnh báo / Hiệu lệnh / Chỉ dẫn)
        cat_bbox = draw.textbbox((0, 0), cat_name, font=FONT_TAG)
        cat_w = cat_bbox[2] - cat_bbox[0]
        bx2 = hud_x2 - 12
        bx1 = bx2 - cat_w - 12
        draw.rectangle([bx1, hud_y1 + 10, bx2, hud_y1 + 28], fill=cat_color_rgb)
        draw.text((bx1 + 6, hud_y1 + 11), cat_name, font=FONT_TAG, fill=(255, 255, 255) if cat_name != 'CẢNH BÁO' else (20, 20, 20))

    # Thông tin chi tiết dưới ảnh Zoom
    if last_crop is not None and last_crop.size > 0:
        info_y = crop_y2 + 8
        
        # Tên biển báo đầy đủ có dấu
        lines = wrap_text(last_sign_name, FONT_BODY_BOLD, hud_w - 24, draw)
        for line in lines[:2]:
            draw.text((hud_x1 + 12, info_y), line, font=FONT_BODY_BOLD, fill=(255, 255, 255))
            info_y += 20
        
        # Confidence Progress Bar
        bar_y = hud_y2 - 50
        bar_w = hud_w - 24
        bar_h = 10
        draw.rectangle([hud_x1 + 12, bar_y, hud_x1 + 12 + bar_w, bar_y + bar_h], fill=(35, 40, 50))
        fill_w = int(bar_w * min(1.0, max(0.0, last_conf)))
        draw.rectangle([hud_x1 + 12, bar_y, hud_x1 + 12 + fill_w, bar_y + bar_h], fill=cat_color_rgb)

        # Confidence Text & Bounding box dimensions
        draw.text((hud_x1 + 12, hud_y2 - 36), f"Độ tin cậy: {last_conf * 100:.1f}%", font=FONT_TAG, fill=(0, 255, 140))
        if last_bbox:
            bx, by, bw, bh = last_bbox
            telemetry_str = f"Tọa độ: [{bx},{by}] {bw}x{bh}px"
            t_bbox = draw.textbbox((0, 0), telemetry_str, font=FONT_TELEMETRY)
            draw.text((hud_x2 - 12 - (t_bbox[2] - t_bbox[0]), hud_y2 - 34), telemetry_str, font=FONT_TELEMETRY, fill=(140, 155, 175))
    else:
        # Trạng thái chờ quét
        wait_text = "Đang quét tìm biển báo giao thông..."
        wb = draw.textbbox((0, 0), wait_text, font=FONT_BODY)
        draw.text((crop_x1 + (crop_w - (wb[2] - wb[0])) // 2, crop_y1 + 80), wait_text, font=FONT_BODY, fill=(120, 135, 155))

    # Bottom-Left Recent Log Stream
    if recent_history:
        draw.text((hist_x1 + 10, hist_y1 + 3), "LỊCH SỬ NHẬN DIỆN GẦN NHẤT", font=FONT_TAG, fill=(0, 215, 255))
        for idx, item in enumerate(recent_history[:3]):
            row_y = hist_y1 + 22 + idx * 22
            cat_tag, _, cat_rgb = classify_sign_category(item['name'])
            # Dot color
            draw.ellipse([hist_x1 + 10, row_y + 4, hist_x1 + 18, row_y + 12], fill=cat_rgb)
            # Item text
            time_txt = f"[{item['time']}]"
            draw.text((hist_x1 + 24, row_y), time_txt, font=FONT_TELEMETRY, fill=(130, 145, 165))
            
            # Shortened clean name if needed
            short_name = item['name'] if len(item['name']) <= 26 else item['name'][:24] + ".."
            draw.text((hist_x1 + 80, row_y), short_name, font=FONT_BODY, fill=(240, 240, 240))
            draw.text((hist_x1 + hist_w - 52, row_y), f"{item['conf']*100:.0f}%", font=FONT_TAG, fill=(0, 255, 140))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==============================================================================
# VẼ BOUNDING BOX & FLOATING HUD TARGET TRÊN VIDEO CHÍNH
# ==============================================================================
def draw_detections(frame, boxes, model):
    clean_crops = []
    h, w, _ = frame.shape
    
    # 1. Trích xuất ảnh crop sạch từ frame gốc
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        raw_name = model.names.get(cls_id, f"Class {cls_id}")
        viet_name = get_vietnamese_name(raw_name, cls_id)

        pad = int(max(x2 - x1, y2 - y1) * 0.08) + 4
        crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
        crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if cropped.size > 0:
            clean_crops.append({
                'crop': cropped,
                'name': viet_name,
                'conf': conf,
                'box': (x1, y1, x2, y2),
                'dim': (x1, y1, x2 - x1, y2 - y1),
                'cls_id': cls_id
            })

    # 2. Vẽ hiệu ứng HUD Targeting trên video
    for item in clean_crops:
        x1, y1, x2, y2 = item['box']
        cat_name, cat_bgr, _ = classify_sign_category(item['name'])

        # Viền mỏng mờ bao quanh
        cv2.rectangle(frame, (x1, y1), (x2, y2), cat_bgr, 1)
        
        # 4 Góc ngắm công nghệ sắc sảo
        draw_corner_brackets(frame, x1, y1, x2, y2, cat_bgr, length=14, thickness=2)

        # Tâm ngắm nhỏ ở chính giữa biển báo
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 2, cat_bgr, -1)

    # 3. Vẽ Floating Glass Tag (Nhãn nổi tiếng Việt)
    if clean_crops:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for item in clean_crops:
            x1, y1, _, _ = item['box']
            cat_name, _, cat_rgb = classify_sign_category(item['name'])
            label_text = f"{item['name']} ({item['conf']*100:.0f}%)"
            
            bbox = draw.textbbox((0, 0), label_text, font=FONT_TAG)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            tag_y1 = max(0, y1 - lh - 8)
            tag_y2 = tag_y1 + lh + 6
            tag_x1 = max(0, x1)
            tag_x2 = min(w, tag_x1 + lw + 12)
            
            # Khung nền nổi tối với viền màu nhóm
            draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=(18, 20, 28), outline=cat_rgb, width=1)
            draw.text((tag_x1 + 6, tag_y1 + 2), label_text, font=FONT_TAG, fill=(255, 255, 255))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return frame, clean_crops

# ==============================================================================
# HÀM MAIN CHẠY SUY LUẬN REAL-TIME
# ==============================================================================
def main():
    video_source = "demo_traffic.mp4"
    model_path = "runs/detect/traffic_sign_exp/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"[!] CẢNH BÁO: Không tìm thấy '{model_path}'.")
        if os.path.exists("yolov8n.pt"):
            print("[*] Tạm thời dùng 'yolov8n.pt' làm dự phòng.")
            model_path = "yolov8n.pt"
        else:
            print("[!] Vui lòng chạy 'python step2_train.py' trước để tạo file model.")
            sys.exit(1)

    if not os.path.exists(video_source):
        if os.path.exists("demo_traffic.mp4.mp4"):
            video_source = "demo_traffic.mp4.mp4"
        else:
            print(f"[!] Không tìm thấy video '{video_source}'.")
            choice = input("Nhập đường dẫn video (hoặc 0 dùng webcam, Enter để thoát): ").strip()
            if choice == "0":
                video_source = 0
            elif choice and os.path.exists(choice):
                video_source = choice
            else:
                print("Thoát chương trình.")
                sys.exit(1)

    print(f"[*] Đang nạp mô hình: {model_path}")
    model = YOLO(model_path)

    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[!] Không thể mở nguồn video/camera: {video_source}")
        sys.exit(1)

    window_name = "YOLOv8 ADAS Traffic Sign Detection Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    is_paused = False
    last_crop = None
    last_sign_name = ""
    last_conf = 0.0
    last_bbox = None

    recent_history = []
    last_history_sign = ""
    last_history_time = 0

    prev_time = time.time()
    fps = 0.0

    print("\n" + "=" * 70)
    print("  HỆ THỐNG GIÁM SÁT & NHẬN DIỆN BIỂN BÁO ADAS CHUYÊN NGHIỆP (YOLOv8)  ")
    print("  - Giao diện Dashboard Cyberpunk / Tesla ADAS Vision cao cấp")
    print("  - Phân loại nhóm biển: Cấm (Đỏ) | Cảnh báo (Vàng) | Hiệu lệnh (Xanh)")
    print("  - Phóng to & phục hồi chi tiết biển báo Lanczos4 + Unsharp Masking")
    print("  - SPACE: Tạm dừng / Tiếp tục | 'q' hoặc ESC: Thoát")
    print("=" * 70 + "\n")

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("[*] Video hoàn tất, tự động phát lặp lại (Loop)...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff) if fps > 0 else 1.0 / time_diff
            prev_time = curr_time

            raw_clean_frame = frame.copy()

            # YOLOv8 Inference
            results = model.predict(source=frame, conf=0.35, iou=0.45, verbose=False)[0]
            boxes = results.boxes
            detected_count = len(boxes)

            frame, detected_items = draw_detections(raw_clean_frame, boxes, model)

            if detected_items:
                best_item = max(detected_items, key=lambda x: x['conf'])
                last_crop = best_item['crop']
                last_sign_name = best_item['name']
                last_conf = best_item['conf']
                last_bbox = best_item['dim']

                # Cập nhật Recent History log
                now_str = datetime.now().strftime("%H:%M:%S")
                if last_sign_name != last_history_sign or (curr_time - last_history_time) > 2.5:
                    recent_history.insert(0, {
                        'name': last_sign_name,
                        'conf': last_conf,
                        'time': now_str
                    })
                    recent_history = recent_history[:5]
                    last_history_sign = last_sign_name
                    last_history_time = curr_time

            display_frame = draw_hud(frame, last_crop, last_sign_name, last_conf, last_bbox, fps, is_paused, detected_count, recent_history)
        else:
            display_frame = draw_hud(frame.copy(), last_crop, last_sign_name, last_conf, last_bbox, fps, is_paused, detected_count, recent_history)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1 if not is_paused else 30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32:
            is_paused = not is_paused
            print(f"[*] Trạng thái: {'TẠM DỪNG (PAUSED)' if is_paused else 'TIẾP TỤC (PLAYING)'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
