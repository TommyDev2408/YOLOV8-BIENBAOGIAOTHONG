import os
import sys
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" ĐÁNH GIÁ MÔ HÌNH YOLOV8 TRÊN TẬP TEST (TEST SPLIT EVALUATION) ")
    print("=" * 65)

    model_weights = "runs/detect/traffic_sign_exp/weights/best.pt"
    data_yaml_path = os.path.abspath("data.yaml")

    # Kiểm tra file trọng số
    if not os.path.exists(model_weights):
        print(f"[!] Không tìm thấy file trọng số: {model_weights}")
        print("[!] Hãy chắc chắn bạn đã chạy huấn luyện qua 'python step2_train.py' trước.")
        sys.exit(1)

    print(f"[*] Đang tải mô hình: {model_weights}")
    model = YOLO(model_weights)

    print("[*] Đang tiến hành đánh giá trên tập dữ liệu test...")
    # Thực hiện validate trên split 'test'
    metrics = model.val(
        data=data_yaml_path,
        split='test',
        imgsz=640,
        batch=16,
        save_json=True,
        plots=True
    )

    # Trích xuất các chỉ số đánh giá chính
    precision = metrics.box.mp         # Mean Precision
    recall = metrics.box.mr            # Mean Recall
    map50 = metrics.box.map50          # mAP @ IoU 0.50
    map50_95 = metrics.box.map         # mAP @ IoU 0.50:0.95

    print("\n" + "=" * 65)
    print(" KẾT QUẢ ĐÁNH GIÁ ĐỘ CHÍNH XÁC (METRICS SUMMARY) ")
    print("=" * 65)
    print(f" • Precision (Độ chuẩn xác)       : {precision:.4f} ({precision * 100:.2f}%)")
    print(f" • Recall (Độ nhạy / Thu hồi)     : {recall:.4f} ({recall * 100:.2f}%)")
    print(f" • mAP@50 (mAP tại IoU 0.50)       : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f" • mAP@50-95 (mAP trung bình tổng) : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print("=" * 65)
    print(f"[✓] Chi tiết các biểu đồ đánh giá đã lưu trong thư mục: {metrics.save_dir}")

if __name__ == "__main__":
    main()
