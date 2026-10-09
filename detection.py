import cv2

from config import BBOX_COLOR, BBOX_THICKNESS


def gaze_to_pixel(norm_pos, w, h): # prevodjene normalizovane pozicije pogleda (0-1) u piksele slike
    x = int(norm_pos[0] * w)
    y = int((1 - norm_pos[1]) * h) # pupil core racuna od donjeg lijevog, a openCV od gornjeg lijevog ugla
    return x, y


def draw_yolo_boxes(frame, yolo_result, class_names): # crtanje bbox-ova i klase svakog detektovanog prdmeta
    for i in range(len(yolo_result.boxes)):
        x1, y1, x2, y2 = yolo_result.boxes.xyxy[i].tolist()
        cls_id = int(yolo_result.boxes.cls[i])
        conf = float(yolo_result.boxes.conf[i])
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      BBOX_COLOR, BBOX_THICKNESS)
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 1)


def find_object_at_gaze(yolo_result, gaze_x, gaze_y, class_names, conf_min=0.0): # vraca (klasa, bbox) predmeta koji se gleda
    kandidati = []
    for i in range(len(yolo_result.boxes)):
        x1, y1, x2, y2 = yolo_result.boxes.xyxy[i].tolist()
        conf = float(yolo_result.boxes.conf[i])

        # preskoci nesigurne detekcije
        if conf < conf_min:
            continue

        # da li tacka pogleda pada unutar ovog okvira
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            povrsina = (x2 - x1) * (y2 - y1)
            cls_id = int(yolo_result.boxes.cls[i])
            kandidati.append(
                (povrsina, class_names[cls_id], (int(x1), int(y1), int(x2), int(y2)))
            )

    # ako gaze nije u bboxu, ne gledamo nista
    if not kandidati:
        return None, None

    kandidati.sort(key=lambda c: c[0]) # po povrsini rastuce, zastita od preklapanja bboxova
    _, ime, bbox = kandidati[0]  # biramo onaj sa najmanjom povrsinom
    return ime, bbox


def find_marker_in_bbox(markers, bbox): # trazenje markera koji pripada predmetu koji se gleda
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    for m in markers:
        cx, cy = m["center"]
        if x1 <= cx <= x2 and y1 <= cy <= y2: # trazimo marker ciji centar pada unutar bboxa
            return m
    return None
