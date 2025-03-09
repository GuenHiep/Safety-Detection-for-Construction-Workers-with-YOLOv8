import cv2
import torch
import numpy as np
from ultralytics import YOLO


# Tải mô hình YOLOv8
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None


# Danh sách các class PPE
SELECTED_CLASSES = {
    0: "Người",
    1: "Tải",
    2: "Bịt tai",
    3: "Mặt",
    4: "Bảo vệ mắt",
    5: "Mặt nạ",
    6: "Chân",
    7: "Dụng cụ",
    8: "Kính",
    9: "Găng tay",
    10: "Mũ bảo hộ",
    11: "Tay",
    12: "Đầu",
    13: "Đồ y tế",
    14: "Giày",
    15: "Đồ bảo hộ",
    16: "Áo bảo hộ",
}

# Màu sắc cho từng loại PPE
COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in SELECTED_CLASSES.keys()}


# Hàm nhận diện PPE từ video
def detect_ppe_video(model, video_path, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Lấy thông tin video (FPS, kích thước)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cấu hình lưu video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Dừng khi hết video

        # Nhận diện PPE
        with torch.no_grad():
            results = model(frame)[0]

        for box in results.boxes.data:
            x1, y1, x2, y2, score, class_id = map(float, box)
            class_id = int(class_id)

            if class_id in SELECTED_CLASSES and score > 0.5:
                label = f"{SELECTED_CLASSES[class_id]} ({score:.2f})"
                color = COLORS[class_id]

                # Vẽ bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị video theo thời gian thực
        cv2.imshow("PPE Detection", frame)

        # Ghi frame vào video đầu ra
        out.write(frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video đã lưu tại {output_path}")


if __name__ == "__main__":
    model_path = "best.pt"  # Đường dẫn đến mô hình
    video_path = "video_input.mp4"  # Video đầu vào

    model = load_model(model_path)

    if model:
        detect_ppe_video(model, video_path)
