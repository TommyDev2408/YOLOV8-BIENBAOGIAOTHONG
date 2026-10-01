import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 60)
    print(" BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH YOLOV8 NHẬN DIỆN BIỂN BÁO GIAO THÔNG ")
    print("=" * 60)

    # Kiểm tra GPU CUDA
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Thiết bị huấn luyện (Device): {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Đường dẫn file cấu hình dataset
    data_yaml_path = os.path.abspath("data.yaml")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Không tìm thấy file: {data_yaml_path}")

    # 1. Khởi tạo mô hình YOLOv8 nano
    print("[*] Đang tải mô hình nền tảng yolov8n.pt...")
    model = YOLO("yolov8n.pt")

    # 2. Huấn luyện mô hình
    # - epochs: 30
    # - imgsz: 640
    # - project & name: runs/detect/traffic_sign_exp
    print("[*] Bắt đầu quá trình huấn luyện với 30 Epochs...")
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        project="runs/detect",
        name="traffic_sign_exp",
        exist_ok=True,         # Ghi đè vào thư mục này nếu đã tồn tại thay vì tạo exp2, exp3...
        device=device,
        batch=16,
        plots=True,            # Vẽ biểu đồ loss, confusion matrix, PR curve
        save=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("[✓] HUẤN LUYỆN HOÀN TẤT!")
    print(f"[✓] Trọng số tốt nhất đã lưu tại: runs/detect/traffic_sign_exp/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
