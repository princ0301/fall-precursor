import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path("models/pose_landmarker_lite.task")


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"model already present at {MODEL_PATH}")
        return

    print(f"downloading pose landmarker model to {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("done")


if __name__ == "__main__":
    main()