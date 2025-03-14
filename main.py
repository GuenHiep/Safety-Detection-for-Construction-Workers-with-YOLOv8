import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os


# Tai mo hinh YOLO
# Kiem tra xem co su dung GPU duoc khong, neu khong thi dung CPU
def load_model(model_path):
    try:
        model = YOLO(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Mo hinh da tai tren {device.upper()}")
        return model
    except Exception as e:
        print(f"Loi khi tai mo hinh: {e}")
        return None


# Danh sach cac class can nhan dien
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

# Cac vat dung bao ho bat buoc
REQUIRED_PPE = {"Mu bao ho", "Giay", "Do bao ho", "Ao bao ho", "Gang tay"}


# Ham xu ly nhan dien vat dung bao ho
def detect_ppe(model, frame):
    results = model(frame, conf=0.25, iou=0.35)[0]
    bounding_boxes = []
    detected_classes = set()

    for box in results.boxes.data:
        if len(box) < 6:
            continue

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

            # Ve khung chu nhat quanh vat the nhan dien duoc
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Kiem tra thieu do bao ho
    missing_ppe = REQUIRED_PPE - detected_classes
    if missing_ppe:
        warning_text = f"Thieu: {', '.join(missing_ppe)}"
        print(f"CANH BAO: {warning_text}")
        cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame, bounding_boxes


# Ham xu ly nhan dien tu anh hoac video
def process_input(model, input_path, output_path="output.jpg"):
    if not os.path.exists(input_path):
        print(f"Loi: Khong tim thay file {input_path}")
        return

    file_ext = input_path.split('.')[-1].lower()

    if file_ext in ["jpg", "jpeg", "png"]:
        print("Nhan dien anh...")
        image = cv2.imread(input_path)
        if image is None:
            print("Loi: Khong the doc anh")
            return
        processed_image, _ = detect_ppe(model, image)
        cv2.imwrite(output_path, processed_image)
        print(f"Ket qua da luu tai {output_path}")

    elif file_ext in ["mp4", "avi", "mov"]:
        print("Nhan dien video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Loi: Khong the mo video")
            return

        output_video_path = "output.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        paused = False
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, _ = detect_ppe(model, frame)
                out.write(processed_frame)
                cv2.imshow("Video Processing", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Dung hoac tiep tuc video khi nhan Space
                paused = not paused

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Ket qua video da luu tai {output_video_path}")

    else:
        print("Dinh dang file khong duoc ho tro")


if __name__ == "__main__":
    model_path = "best.pt"
    input_path = "img_1.png"  # Thay bang duong dan file anh hoac video

    model = load_model(model_path)
    if model:
        process_input(model, input_path)