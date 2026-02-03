import json
import zmq
import numpy as np
import cv2
from typing import List, Tuple
import socket
import os

def load_config(config_file="config.json") -> dict:
    """
    Загружает настройки из config.json. Если не найден,
    возвращает дефолтные значения.
    """
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"Config file '{config_file}' not found. Using defaults.")
        return {
            "rtspUrl": "rtsp://127.0.0.1:8554/mystream",
            "standIp": "127.0.0.1",
            "standPort": 7777
        }

def load_interest_points(file_path: str) -> List[List[int]]:
    """
    Загружает JSON, в котором хранится массив объектов вида:
    [
      {"x": ..., "y": ...},
      {"x": ..., "y": ...},
      ...
    ]
    и преобразует в формат:
    [
      [x1, y1],
      [x2, y2],
      ...
    ]
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # [{"x":..., "y":...}, ...]
        points = []
        for item in data:
            x_val = item.get("x", 0)
            y_val = item.get("y", 0)
            points.append([x_val, y_val])
        print(f"Interest points loaded from '{file_path}': {points}")
        return points
    except FileNotFoundError:
        print(f"File with interest points '{file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
        return []

def load_homography_matrix(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            matrix = json.load(f)
        return np.array(matrix, dtype=np.float64)
    except FileNotFoundError:
        raise FileNotFoundError(f"Homography matrix file '{file_path}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error loading homography matrix: {e}")

def apply_homography(points: List[Tuple[int, int]], homography_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Применение гомографии к списку пиксельных точек.
    Возвращает список (x, y) в "реальных" координатах.
    """
    transformed_points = []
    for x, y in points:
        pixel_coords = np.array([[x, y, 1]], dtype=np.float64).T
        real_coords = np.dot(homography_matrix, pixel_coords)
        real_coords /= real_coords[2]  # нормализация
        transformed_points.append((int(real_coords[0][0]), int(real_coords[1][0])))
    return transformed_points

def find_nearest_people(interest_points, people_points):

    matched_points = []
    for ip in interest_points:
        min_distance = float("inf")
        nearest_person = None
        for pp in people_points:
            distance = np.linalg.norm(np.array(ip) - np.array(pp))
            if distance < min_distance:
                min_distance = distance
                nearest_person = pp
        if nearest_person:
            matched_points.append((ip, nearest_person))
    return matched_points

def send_coordinates_udp(json_data: str, stand_ip: str, port: int):
    """
    Отправка JSON-координат по UDP на указанный IP и порт.
    """
    print(f"Sending JSON to {stand_ip}:{port} → {json_data}")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(json_data.encode('utf-8'), (stand_ip, port))

def main():
    # Загружаем конфиг (IP и порт стенда)
    config = load_config("config.json")
    stand_ip = config.get("standIp", "127.0.0.1")
    stand_port = config.get("standPort", 7777)

    # 1) Считываем endpoint из файла (например, zmq_endpoint.json)
    endpoint_file = "zmq_endpoint.json"
    if not os.path.exists(endpoint_file):
        print(f"Error: endpoint file '{endpoint_file}' not found. Cannot connect to object_detection.")
        return

    with open(endpoint_file, "r", encoding="utf-8") as f:
        endpoint_data = json.load(f)
    endpoint = endpoint_data.get("endpoint", "tcp://127.0.0.1:5555")

    print(f"[process_points] Connecting to ZMQ endpoint: {endpoint}")

    # 2) Настраиваем ZeroMQ (PULL) и подключаемся к endpoint
    context = zmq.Context()
    pull_socket = context.socket(zmq.PULL)
    pull_socket.setsockopt(zmq.LINGER, 0)
    pull_socket.connect(endpoint)

    interest_points_file = "points.json"
    homography_file = "homography_matrix.json"

    # Загружаем точки интереса и матрицу гомографии
    interest_points = load_interest_points(interest_points_file)
    homography_matrix = load_homography_matrix(homography_file)

    while True:
        # Получаем JSON от object_detection.py (список people_points)
        message = pull_socket.recv_string()
        if not message:
            continue


        people_points = json.loads(message)

        # Соответствие точек интереса и ближайших людей
        matched_points = find_nearest_people(interest_points, people_points)

        # Готовим словарь для отправки
        transformed_points = {}

        for i, (poi, person) in enumerate(matched_points):
            # Применяем гомографию к точке человека
            real_coords = apply_homography([person], homography_matrix)[0]
            # Записываем в словарь
            transformed_points[str(i + 1)] = f"{real_coords[0]:.2f},{real_coords[1]:.2f}"

            # Формируем JSON
            json_data = json.dumps(transformed_points)
            # Отправляем на IP/порт стенда
            send_coordinates_udp(json_data, stand_ip, stand_port)

            # Вывод отладочной информации
            print(f"POI: {poi}, Person: {person}, Transformed: {real_coords}")

if __name__ == "__main__":
    main()
