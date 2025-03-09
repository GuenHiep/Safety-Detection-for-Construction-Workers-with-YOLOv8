import cv2
import torch
import numpy as np
from ultralytics import YOLO


# Tải mô hình
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None


# Danh sách các class cần nhận diện (theo ID của mô hình)
SELECTED_CLASSES = {
    0: "Nguoi",
    10: "Mu bao ho",
    14: "Giay",
    9: "Gang tay",
    15: "Do bao ho",
    16: "Ao bao ho",
}

# Các vật dụng bảo hộ bắt buộc
REQUIRED_PPE = {"Mu bao ho", "Ao bao ho", "Gang tay", "Giay", "Do bao ho"}


# Hàm nhận diện trang bị bảo hộ lao động (PPE)
def detect_ppe(model, image_path, output_path="output.jpg"):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return []

    results = model(image)[0]
    bounding_boxes = []
    detected_classes = set()

    for box in results.boxes.data:
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
            # Vẽ bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Kiểm tra thiếu đồ bảo hộ
    missing_ppe = REQUIRED_PPE - detected_classes
    if missing_ppe:
        warning_text = f"Thieu: {', '.join(missing_ppe)}"
        print(f"CANH BAO: {warning_text}")
        cv2.putText(image, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Lưu kết quả
    cv2.imwrite(output_path, image)
    print(f"Kết quả đã lưu tại {output_path}")

    return bounding_boxes


if __name__ == "__main__":
    model_path = "best.pt"  # Đường dẫn đến mô hình YOLO
    image_path = "img.png"  # Thay bằng ảnh cần kiểm tra

    model = load_model(model_path)

    if model:
        detected_boxes = detect_ppe(model, image_path)
        print("Kết quả nhận diện:", detected_boxes)
