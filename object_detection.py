import argparse
import json
import time
import zmq
import cv2
import multiprocessing as mp
from pathlib import Path
import numpy as np

# ИМПОРТИРУЕМ все нужные вспомогательные функции (draw_detections, colors и т.д.)
from models import YOLOv5
from utils.general import (
    check_img_size, scale_boxes, increment_path,
    LoadMedia, draw_detections, colors
)
from process_points import load_interest_points, find_nearest_people

def zmq_sender(data_queue, endpoint_file="zmq_endpoint.json"):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)

    # 1) Просим ZeroMQ выбрать свободный порт (ephemeral)
    socket.bind("tcp://127.0.0.1:*")

    # 2) Узнаём, что выбрано
    endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
    print(f"[object_detection] ZMQ bound to ephemeral endpoint: {endpoint}")

    # 3) Сохраняем в файл
    with open(endpoint_file, "w", encoding="utf-8") as f:
        json.dump({"endpoint": endpoint}, f)

    # 4) Далее обычная работа: ждём из data_queue people_points и отправляем
    while True:
        people_points = data_queue.get()
        if people_points is None:
            break
        try:
            socket.send_string(json.dumps(people_points))
        except zmq.ZMQError as e:
            print(f"ZMQ send error: {e}")

    socket.close(linger=0)
    context.term()

def run_object_detection(data_queue,
                         weights,
                         source,
                         img_size,
                         conf_thres,
                         iou_thres,
                         max_det,
                         save,
                         view,
                         project,
                         name,
                         max_fps=None,
                         save_path=None):
    """Основной процесс для выполнения детекции."""
    # Если нужно сохранять результат, создаём выходную папку
    
    if save:
        save_dir = increment_path(Path(project) / name)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # Загружаем модель
    model = YOLOv5(weights, conf_thres, iou_thres, max_det)
    # Проверяем / пригоняем img_size
    img_size = check_img_size(img_size, s=model.stride)
    # Загружаем медиаданные (видео / веб-камера / изображение / rtsp)
    dataset = LoadMedia(source, img_size=img_size)

    # Если нужно сохранять видео, создаём VideoWriter (включая RTSP)
    vid_writer = None
    if save and dataset.type in ["video", "webcam", "rtsp"]:
        cap = dataset.cap
        out_path = save_path if save_path else str(save_dir / Path(source).name)
        if not out_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            out_path = out_path.rstrip('/').rstrip('\\') + '.mp4'
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # RTSP часто не отдаёт fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            print("[object_detection] WARNING: Invalid frame size from source, video save may fail.")
        vid_writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        if not vid_writer.isOpened():
            print(f"[object_detection] ERROR: Failed to open output file: {out_path}")
            vid_writer = None
        else:
            print(f"[object_detection] Saving video to: {out_path}")

    # Ограничение частоты обработки (напр. RTSP 25 fps → обрабатываем 5 fps)
    last_process_time = None

    # Загружаем точки из points.json для отрисовки связей с людьми
    points_file = Path(__file__).parent / "points.json"
    interest_points = load_interest_points(str(points_file))

    # Обрабатываем все кадры / изображения
    for resized_image, original_image, status in dataset:
        # Пропуск кадра, если задан max_fps и ещё не прошло 1/max_fps сек
        if max_fps is not None and max_fps > 0:
            now = time.perf_counter()
            if last_process_time is not None and (now - last_process_time) < 1.0 / max_fps:
                continue
            last_process_time = now

        # Модель инференса
        boxes, scores, class_ids = model(resized_image)

        # Масштабируем боксы обратно к размеру original_image
        boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

        # Координаты «людей» для дальнейшего использования (берём нижнюю середину бокса)
        # и отправляем в очередь → ZMQ
        people_points = [
            (int((box[0] + box[2]) / 2), int(box[3]))
            for box in boxes
        ]
        data_queue.put(people_points)

        # ----- DRAWING: точки из points.json и линии к людям -----
        if interest_points:
            matched_points = find_nearest_people(interest_points, people_points)
            line_color = (0, 255, 0)   # BGR: зелёный для линий
            point_color = (0, 0, 255)  # BGR: красный для точек интереса
            point_radius = 8
            line_thickness = 2
            for pt in interest_points:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(original_image, (x, y), point_radius, point_color, -1)
                cv2.circle(original_image, (x, y), point_radius, (255, 255, 255), 1)
            for poi, person in matched_points:
                pt1 = (int(poi[0]), int(poi[1]))
                pt2 = (int(person[0]), int(person[1]))
                cv2.line(original_image, pt1, pt2, line_color, line_thickness, lineType=cv2.LINE_AA)

        # ----- DRAWING BOUNDING BOXES -----
        # Рисуем прямоугольники и подписи
        for box, score, class_id in zip(boxes, scores, class_ids):
            # draw_detections( image, box, score, class_name, color )
            draw_detections(
                original_image,
                box,
                score,
                model.names[int(class_id)],    # название класса (например, 'person')
                colors(int(class_id))          # цвет
            )

        # Печатаем лог, сколько объектов каждого класса
        for c in np.unique(class_ids):
            n = int((class_ids == c).sum())  # detections per class
            status += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "

        # Если нужно показывать окно
#        if view:
#           cv2.imshow('Webcam Inference', original_image)
 #          if cv2.waitKey(1) & 0xFF == ord('q'):
#                break

        if view:
            # Масштабируем под удобное окно, например ширину 960 пикселей
            max_window_width = 960
            h, w = original_image.shape[:2]
            if w > max_window_width:
                scale = max_window_width / w
                resized = cv2.resize(original_image, (int(w * scale), int(h * scale)))
            else:
                resized = original_image

            cv2.imshow('Webcam Inference', resized)
            cv2.waitKey(1)


        # Печатаем статус (лог) для каждого кадра
        # print(status)

        # Если нужно сохранять результат (кадр)
        if save:
            if dataset.type == "image":
                # Сохраняем как jpg
                frame_path = str(save_dir / f"frame_{dataset.frame:04d}.jpg")
                cv2.imwrite(frame_path, original_image)
            elif dataset.type in ["video", "webcam", "rtsp"] and vid_writer and vid_writer.isOpened():
                vid_writer.write(original_image)

    # Отправляем None в очередь, чтобы завершить ZMQ
    data_queue.put(None)

    # Освобождаем ресурсы
    if save and vid_writer is not None:
        vid_writer.release()

    if save and save_dir:
        print(f"Results saved to {save_dir}")

    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/crowdhuman.onnx", help="model path")
    parser.add_argument("--source", type=str, default="0", help="Path to video/image/webcam/rtsp")
    parser.add_argument("--max-fps", type=float, default=5, help="Max processing FPS (e.g. 5 = process 5 fps from 25 fps RTSP)")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--save", action="store_true", help="Save detected images")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output video path (e.g. C:\\output2.mp4)")
    parser.add_argument("--view", action="store_true", help="View inferenced images")
    parser.add_argument("--project", default="runs", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    args = parser.parse_args()

    # Если пользователь указал img_size только одно число, удваиваем для h,w
    if len(args.img_size) == 1:
        args.img_size = args.img_size * 2
    return args

def main():
    params = parse_args()
    data_queue = mp.Queue()  # Очередь для передачи данных между процессами

    # Запуск ZMQ-процесса
    zmq_process = mp.Process(target=zmq_sender, args=(data_queue,))
    zmq_process.start()

    # Запуск основного процесса детекции
    try:
        run_object_detection(
            data_queue=data_queue,
            weights=params.weights,
            source=params.source,
            img_size=params.img_size,
            conf_thres=params.conf_thres,
            iou_thres=params.iou_thres,
            max_det=params.max_det,
            save=params.save,
            view=params.view,
            project=params.project,
            name=params.name,
            max_fps=params.max_fps,
            save_path=params.output
        )
    finally:
        # Ожидаем завершения ZMQ-процесса
        zmq_process.join()

if __name__ == "__main__":
    main()
