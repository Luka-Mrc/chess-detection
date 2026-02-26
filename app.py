from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

from chess_detection.board import CannyBoardDetector, HoughBoardDetector, DNNBoardDetector
from chess_detection.pieces.yolo import YOLOPieceDetector
from chess_detection.pipeline import ChessPositionPipeline

app = Flask(__name__)

BOARD_DETECTORS = {
    "canny": lambda: CannyBoardDetector.from_config("config/board_detection.yaml"),
    "hough": lambda: HoughBoardDetector.from_config("config/board_detection.yaml"),
    "dnn":   lambda: DNNBoardDetector.from_config("config/board_detection.yaml"),
}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/detect")
def detect():
    method = request.form.get("method", "canny")
    if method not in BOARD_DETECTORS:
        return jsonify({"error": f"Unknown method: {method}"}), 400

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    board_det = BOARD_DETECTORS[method]()
    piece_det = YOLOPieceDetector.from_config("config/piece_detection.yaml")
    pipeline = ChessPositionPipeline(board_det, piece_det)
    result = pipeline.run(image)

    fen = result.fen or ""
    success = bool(result.board.success) if hasattr(result.board, "success") else (result.board.homography is not None)
    return jsonify({"fen": fen, "success": success})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
