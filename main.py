import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Tải mô hình
def load_model(model_path):
    try:
        model = YOLO(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Mo hinh đa tai tren {device.upper()}")
        return model
    except Exception as e:
        print(f"Loi khi tai mo hinh: {e}")
        return None

# Danh sách các class cần nhận diện (theo ID của mô hình)
SELECTED_CLASSES = {
    0: "Nguoi",
    1: "Tai",
    2: "Bit tai",
    # 3: "Mat",
    4: "Bao ve mat",
    5: "Mat na",
    # 6: "Chan",
    # 7: "Dung cu",
    8: "Kinh",
    9: "Gang tay",
    10: "Mu bao ho",
    # 11: "Tay",
    # 12: "Đau",
    # 13: "Đo y te",
    14: "Giay",
    15: "Đo bao ho",
    16: "Ao bao ho",
}

# Các vật dụng bảo hộ bắt buộc
REQUIRED_PPE = {"Mu bao ho","Giay","Do bao ho", "Ao bao ho", "Gang tay"}

# Hàm nhận diện trang bị bảo hộ lao động (PPE)
def detect_ppe(model, image_path, output_path="output.jpg"):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Loi: Khong the đoc anh tu {image_path}")
        return []

    # Dự đoán với ngưỡng nhận diện thấp hơn để phát hiện nhiều hơn
    results = model(image, conf=0.25, iou=0.35)[0]

    # Kiểm tra xem có phát hiện nào không
    if results.boxes is None or len(results.boxes) == 0:
        print("Khong phat hien vat the nao trong anh!")
        return []

    bounding_boxes = []
    detected_classes = set()

    for box in results.boxes.data:
        if len(box) < 6:
            continue  # Bỏ qua nếu box không hợp lệ

        x1, y1, x2, y2, score, class_id = map(float, box)
        class_id = int(class_id)

        if class_id in SELECTED_CLASSES:
            class_name = SELECTED_CLASSES[class_id]
            detected_classes.add(class_name)
            label = f"{class_name} ({score:.2f})"
            bounding_boxes.append({
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            # Vẽ bounding box màu xanh lá cây
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Kiểm tra thiếu đồ bảo hộ
    missing_ppe = REQUIRED_PPE - detected_classes
    if missing_ppe:
        warning_text = f"Thieu: {', '.join(missing_ppe)}"
        print(f"CANH BAO: {warning_text}")

        # Hiển thị cảnh báo trên ảnh với màu đỏ nổi bật
        cv2.putText(image, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Lưu kết quả
    cv2.imwrite(output_path, image)
    print(f"Ket qua đa luu tai {output_path}")

    return bounding_boxes

if __name__ == "__main__":
    model_path = "best.pt"  # Đường dẫn đến mô hình YOLO
    image_path = "img_1.png"  # Thay bằng ảnh cần kiểm tra

    model = load_model(model_path)

    if model:
        detected_boxes = detect_ppe(model, image_path)
        print("Ket qua nhan dien:", detected_boxes)
