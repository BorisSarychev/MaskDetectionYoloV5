import os
import json
import signal
import cv2
import numpy as np
import subprocess

from flask import Flask, render_template, request, jsonify
from homography_service import calculate_and_save_homography

app = Flask(__name__)

CONFIG_FILE = "config.json"
SNAPSHOT_PATH = "static/images/snapshot.jpg"

DETECTION_PROCESS = None  # Глобальная переменная, чтобы хранить Subprocess
POINTS_PROCESS = None

########################################
# 1) Домашняя страница
########################################
@app.route("/")
def home_page():
    """Домашняя страница (home.html)."""
    return render_template("home.html")

########################################
# 2) Страницы points / homography / config
########################################
@app.route("/points")
def points_page():
    """Страница с точками интереса."""
    return render_template("points.html")

@app.route("/homography")
def homography_page():
    """Страница гомографии."""
    return render_template("homography.html")

@app.route("/config")
def config_page():
    """Страница настроек (config.html)."""
    return render_template("config.html")

########################################
# 3) API: status, start/stop detection, start/stop points
########################################
@app.route("/api/status", methods=["GET"])
def status():
    """Возвращает запущены ли процессы."""
    return jsonify({
        "detection_running": (DETECTION_PROCESS is not None),
        "points_running": (POINTS_PROCESS is not None)
    })

@app.route("/api/start_detection", methods=["POST"])
def start_detection():
    global DETECTION_PROCESS
    if DETECTION_PROCESS is not None:
        return jsonify({"status": "error", "message": "Detection already running"}), 400

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    cmd = [
        "python", "object_detection.py",
        f"--weights={cfg.get('weights','weights/crowdhuman.onnx')}",
        f"--source={cfg.get('rtspUrl','0')}",
        f"--conf-thres={cfg.get('conf_thres',0.45)}",
        f"--iou-thres={cfg.get('iou_thres',0.45)}",
        f"--max-det={cfg.get('max_det',1000)}",
        f"--img-size={cfg.get('img_size',640)}"
    ]
    max_fps = cfg.get("max_fps")
    if max_fps is not None and max_fps > 0:
        cmd.append(f"--max-fps={max_fps}")
    if cfg.get("view", False):
        cmd.append("--view")
    if cfg.get("save", False):
        cmd.append("--save")

    img_size = cfg.get("img_size", 640)

    if isinstance(img_size, list):
        # Напр. img_size = [640, 480]
        # Нужно сгенерировать:  ["--img-size", "640", "480"]
        cmd.append("--img-size")
        cmd.extend(str(i) for i in img_size)
    else:
        # Если только одно число, например 640
        # Нужно сгенерировать: ["--img-size", "640"]
        cmd.extend(["--img-size", str(img_size)])

    try:
        #subprocess.Popen(['./venv/Scripts/activate.bat'])
        DETECTION_PROCESS = subprocess.Popen(cmd)
        return jsonify({"status": "ok", "message": "Detection started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stop_detection", methods=["POST"])
def stop_detection():
    global DETECTION_PROCESS
    if DETECTION_PROCESS is None:
        return jsonify({"status": "error", "message": "No detection running"}), 400

    try:
        DETECTION_PROCESS.terminate()
        DETECTION_PROCESS.wait(timeout=5)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to stop detection: {e}"}), 500
    finally:
        DETECTION_PROCESS = None

    return jsonify({"status": "ok", "message": "Detection stopped"})

@app.route("/api/start_process_points", methods=["POST"])
def start_process_points():
    global POINTS_PROCESS
    if POINTS_PROCESS is not None:
        return jsonify({"status": "error", "message": "Points process already running"}), 400

    try:
        POINTS_PROCESS = subprocess.Popen(["python", "process_points.py"])
        return jsonify({"status": "ok", "message": "Points process started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stop_process_points", methods=["POST"])
def stop_process_points():
    global POINTS_PROCESS
    if POINTS_PROCESS is None:
        return jsonify({"status": "error", "message": "No points process running"}), 400

    try:
        POINTS_PROCESS.terminate()
        POINTS_PROCESS.wait(timeout=5)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to stop points: {e}"}), 500
    finally:
        POINTS_PROCESS = None

    return jsonify({"status": "ok", "message": "Points process stopped"})

########################################
# 4) API: load_config, save_config, get_snapshot, etc.
########################################
@app.route("/api/load_config", methods=["GET"])
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "rtspUrl": "rtsp://127.0.0.1:8554/mystream",
            "standIp": "127.0.0.1",
            "standPort": 7777,
            "weights": "weights/crowdhuman.onnx",
            "conf_thres": 0.45,
            "iou_thres": 0.45,
            "max_det": 1000,
            "view": False
        }
    return jsonify(data)

@app.route("/api/save_config", methods=["POST"])
def save_config():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No config data provided"}), 400

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "ok", "message": "Config saved"})

@app.route("/api/get_snapshot", methods=["GET"])
def get_snapshot():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {"rtspUrl": "rtsp://127.0.0.1:8554/mystream"}

    rtsp_url = config.get("rtspUrl", "rtsp://127.0.0.1:8554/mystream")

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return jsonify({"error": f"Не удалось открыть RTSP-поток: {rtsp_url}"}), 500

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Не удалось считать кадр из RTSP."}), 500

    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    cv2.imwrite(SNAPSHOT_PATH, frame)

    return jsonify({"snapshotUrl": "/" + SNAPSHOT_PATH})

@app.route("/api/load_points", methods=["GET"])
def load_points():
    file_path = "points.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            points = json.load(f)
    else:
        points = []
    return jsonify(points)

@app.route("/api/save_points", methods=["POST"])
def save_points():
    data = request.get_json()
    pts = data.get("points", [])
    file_path = "points.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(pts, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "ok", "message": "Points saved"})

@app.route("/api/load_homography_points", methods=["GET"])
def load_homography_points():
    file_path = "homography_points.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    return jsonify(data)

from homography_service import calculate_and_save_homography

@app.route("/api/apply_homography", methods=["POST"])
def apply_homography_api():
    data = request.get_json()
    if not data or "points" not in data:
        return jsonify({"status": "error", "message": "No 'points' in request"}), 400

    points = data["points"]
    try:
        calculate_and_save_homography(
            points,
            matrix_file="homography_matrix.json",
            points_file="homography_points.json"
        )
        return jsonify({"status": "ok", "message": "Homography updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

########################################
# 5) Запуск приложения
########################################
if __name__ == "__main__":
    app.run(debug=True)
