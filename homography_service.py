# homography_service.py
import cv2
import numpy as np
import json

def calculate_and_save_homography(points_data,
                                  matrix_file="homography_matrix.json",
                                  points_file="homography_points.json"):
    """
    Вычисляет матрицу гомографии и сохраняет два JSON-файла:
      1) homography_matrix.json (3×3)
      2) homography_points.json (список точек для отображения/отладки)
    
    :param points_data: список словарей вида:
        [
          {"x": int/float, "y": int/float, "realX": float, "realY": float},
          ...
        ]
        Минимум 4 пары для findHomography.
    :param matrix_file: имя файла, куда писать матрицу (3×3)
    :param points_file: имя файла, куда писать сам список (для фронтенда)
    """
    if len(points_data) < 4:
        raise ValueError("Нужно как минимум 4 точки для вычисления гомографии.")

    # Разделяем пиксельные и реальные координаты
    pixel_points = []
    real_points = []
    for p in points_data:
        pixel_points.append([p["x"], p["y"]])
        real_points.append([p["realX"], p["realY"]])

    pixel_points = np.array(pixel_points, dtype=np.float32)
    real_points = np.array(real_points, dtype=np.float32)

    # Находим матрицу гомографии
    homography_matrix, status = cv2.findHomography(pixel_points, real_points)
    if status is None or not status.any():
        raise ValueError("cv2.findHomography не смог вычислить матрицу гомографии.")

    # Сохраняем саму матрицу
    with open(matrix_file, "w", encoding="utf-8") as f:
        json.dump(homography_matrix.tolist(), f, indent=2, ensure_ascii=False)

    # Сохраняем точки (на будущее)
    with open(points_file, "w", encoding="utf-8") as f:
        json.dump(points_data, f, indent=2, ensure_ascii=False)

    print(f"[calculate_and_save_homography] Матрица сохранена в {matrix_file}")
    print(f"[calculate_and_save_homography] Точки сохранены в {points_file}")
