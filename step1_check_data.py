import os
import random
import glob
import cv2
import yaml

def load_classes(yaml_path="data.yaml"):
    """Đọc danh sách nhãn lớp từ file data.yaml."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Không tìm thấy file: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def check_random_sample(images_dir="train/images", labels_dir="train/labels", yaml_path="data.yaml"):
    """Đọc ngẫu nhiên 1 ảnh và vẽ Bounding Box màu đỏ kèm tên nhãn biển báo."""
    class_names = load_classes(yaml_path)
    
    # Lấy danh sách ảnh hợp lệ
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
    if not image_files:
        print(f"[!] Không tìm thấy ảnh nào trong thư mục: {images_dir}")
        return

    print(f"[*] Tổng số ảnh trong tập train: {len(image_files)}")
    print("[*] Nhấn phím SPACE hoặc phím bất kỳ để xem ảnh ngẫu nhiên tiếp theo. Nhấn 'q' hoặc ESC để thoát.")

    while True:
        img_path = random.choice(image_files)
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # Đọc ảnh
        image = cv2.imread(img_path)
        if image is None:
            print(f"[!] Không thể đọc ảnh: {img_path}")
            continue

        h, w, _ = image.shape
        num_boxes = 0

        # Đọc và vẽ nhãn
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    bbox_width = float(parts[3]) * w
                    bbox_height = float(parts[4]) * h

                    # Chuyển đổi tọa độ YOLO (x_center, y_center, w, h) sang (x1, y1, x2, y2)
                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)

                    # Đảm bảo tọa độ nằm trong ảnh
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    # Lấy tên nhãn
                    label_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    display_text = f"{class_id}: {label_name}"

                    # 1. Vẽ Bounding Box màu đỏ (BGR: 0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # 2. Vẽ nền cho chữ để dễ đọc
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                    
                    label_y = max(y1 - 5, text_h + 5)
                    cv2.rectangle(image, (x1, label_y - text_h - 4), (x1 + text_w + 4, label_y + 2), (0, 0, 255), -1)
                    cv2.putText(image, display_text, (x1 + 2, label_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    num_boxes += 1
        else:
            print(f"[!] Không tìm thấy file nhãn: {label_path}")

        # Thêm thông tin header trên ảnh
        info_bar = f"File: {img_name} | Boxes: {num_boxes} | Next: Any key | Exit: 'q'/ESC"
        cv2.putText(image, info_bar, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Hiển thị ảnh
        cv2.imshow("Kiem tra du lieu bien bao (Check Data)", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' hoặc ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_random_sample()
