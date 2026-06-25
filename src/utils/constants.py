import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTO    = os.path.join(BASE_DIR, "models", "mobilenet_ssd.prototxt")
WEIGHTS  = os.path.join(BASE_DIR, "models", "mobilenet_ssd.caffemodel")

SSD_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow","diningtable","dog","horse",
    "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

C = {
    "bg":        "#0F1E14",
    "panel":     "#162219",
    "card":      "#1C2E22",
    "border":    "#243B2C",
    "green":     "#22C55E",
    "green_dim": "#15532E",
    "amber":     "#F59E0B",
    "amber_dim": "#78350F",
    "red":       "#EF4444",
    "red_dim":   "#7F1D1D",
    "text":      "#E2F5E8",
    "text_mid":  "#86A891",
    "text_dim":  "#4B7055",
    "white":     "#FFFFFF",
}

BOVINO_IDS = {10: "Boi/Vaca"}

MODEL_URLS = {
    "mobilenet_ssd.prototxt": (
        "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/"
        "master/MobileNetSSD_deploy.prototxt"
    ),
    "mobilenet_ssd.caffemodel": (
        "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/"
        "master/MobileNetSSD_deploy.caffemodel"
    ),
}
