from pathlib import Path

PROJECT_PATH = Path(__file__).parent
BETELGEUSE_DATA_PATH = (
    PROJECT_PATH / "data" / "betelguese" / "aavsodata_5e4b04f969ce3.txt"
)

BETELGEUSE_PERCENTAGE_POINTS_TO_OMMIT = 0.7
BETELGEUSE_TAKE_EVERY_NTH_POINT = 4

TRAIN_RATIO = 0.7
