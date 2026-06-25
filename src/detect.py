import cv2
import os
import urllib.request
import numpy as np
from src.utils.constants import PROTO, WEIGHTS, C, BASE_DIR, MODEL_URLS, BOVINO_IDS

class Detector:
    _TABELA_PESO = [
        ( 70,  50,   80,  110),
        ( 80,  80,  120,  170),
        ( 90, 120,  180,  250),
        (100, 170,  250,  340),
        (110, 240,  330,  430),
        (120, 320,  420,  530),
        (130, 390,  500,  620),
        (140, 450,  570,  700),
        (150, 500,  630,  750),
        (162, 540,  670,  790),
    ]

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(PROTO, WEIGHTS)

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.40):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        dets = self.net.forward()

        results = []
        for i in range(dets.shape[2]):
            conf     = float(dets[0, 0, i, 2])
            class_id = int(dets[0, 0, i, 1])
            if conf < conf_thresh or class_id not in BOVINO_IDS:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            results.append({
                "class_id":    class_id,
                "label":       BOVINO_IDS[class_id],
                "confidence":  conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "bbox_w_px":   x2 - x1,
                "bbox_h_px":   y2 - y1,
                "bbox_w_frac": (x2 - x1) / w,
                "bbox_h_frac": (y2 - y1) / h,
            })
        return sorted(results, key=lambda r: -r["confidence"])

    @staticmethod
    def estimate_metrics(det: dict, frame_h: int, frame_w: int) -> dict:
        bh_frac = det["bbox_h_frac"]
        bw_frac = det["bbox_w_frac"]

        # ── Altura estimada ───────────────────────────────────────────────────
        # Expoente 0.45 suaviza o efeito de proximidade/afastamento da câmera.
        # Âncora: bh_frac=0.45 → ~105 cm (novilha jovem típica no frame médio).
        altura_cm = 105.0 * (bh_frac / 0.45) ** 0.45
        altura_cm = max(70.0, min(162.0, round(altura_cm, 1)))

        # ── Índice de robustez corporal (wh_ratio) ────────────────────────────
        # Razão largura/altura do bbox; invariante à distância da câmera.
        # Boi gordo de lado: wh > 1.0; bezerro/magro: wh < 0.80.
        wh_ratio = bw_frac / max(bh_frac, 0.01)

        # ── Comprimento estimado ──────────────────────────────────────────────
        comprimento_cm = altura_cm * 1.35 * (bw_frac / max(bh_frac * 1.35, 0.01))
        comprimento_cm = max(80.0, min(300.0, round(comprimento_cm, 1)))

        # ── Peso via tabela Embrapa + ajuste wh_ratio ─────────────────────────
        tab = Detector._TABELA_PESO
        if altura_cm <= tab[0][0]:
            p_min, p_max = tab[0][1], tab[0][3]
        elif altura_cm >= tab[-1][0]:
            p_min, p_max = tab[-1][1], tab[-1][3]
        else:
            for i in range(len(tab) - 1):
                if tab[i][0] <= altura_cm <= tab[i+1][0]:
                    t = (altura_cm - tab[i][0]) / (tab[i+1][0] - tab[i][0])
                    p_min = tab[i][1] + t * (tab[i+1][1] - tab[i][1])
                    p_max = tab[i][3] + t * (tab[i+1][3] - tab[i][3])
                    break

        # fator: 0.0 = muito magro (wh≤0.55), 1.0 = muito gordo (wh≥1.30)
        fator_cc = (wh_ratio - 0.55) / (1.30 - 0.55)
        fator_cc = max(0.0, min(1.0, fator_cc))
        peso_kg  = round(p_min + fator_cc * (p_max - p_min), 1)
        peso_kg  = max(50.0, min(800.0, peso_kg))

        # ── IMC bovino ────────────────────────────────────────────────────────
        altura_m = altura_cm / 100.0
        imc = round((peso_kg / (altura_m ** 2)) * 0.01, 2)

        return {
            "altura_cm":      altura_cm,
            "comprimento_cm": comprimento_cm,
            "peso_kg":        peso_kg,
            "imc":            imc,
            "wh_ratio":       round(wh_ratio, 2),
        }

    @staticmethod
    def diagnostico(metrics: dict) -> dict:
        imc  = metrics["imc"]
        alertas = []
        score   = 100

        if imc < 1.70:
            alertas.append(("CRÍTICO", f"IMC {imc:.2f} — Subnutrição severa"))
            score -= 40
        elif imc < 2.00:
            alertas.append(("ATENÇÃO",  f"IMC {imc:.2f} — Abaixo do ideal (< 2.00)"))
            score -= 20
        elif imc > 3.20:
            alertas.append(("ATENÇÃO",  f"IMC {imc:.2f} — Sobrepeso (> 3.20)"))
            score -= 15
        elif imc > 2.80:
            alertas.append(("AVISO",    f"IMC {imc:.2f} — Levemente acima do ideal"))
            score -= 8

        if metrics["peso_kg"] < 150:
            alertas.append(("CRÍTICO", f"Peso {metrics['peso_kg']} kg — muito baixo"))
            score -= 20
        elif metrics["peso_kg"] < 250:
            alertas.append(("ATENÇÃO",  f"Peso {metrics['peso_kg']} kg — abaixo do esperado"))
            score -= 10

        score = max(0, score)

        if score >= 80:   status = "SAUDÁVEL"
        elif score >= 55: status = "ATENÇÃO"
        else:             status = "CRÍTICO"

        return {"status": status, "score": score, "alertas": alertas}

def ensure_model(progress_cb=None):
    for fname, url in MODEL_URLS.items():
        dest = os.path.join(BASE_DIR, 'models', fname)
        if os.path.exists(dest):
            continue
        if progress_cb:
            progress_cb(f"Baixando {fname}…")
        req = urllib.request.Request(url, headers={"User-Agent": "BovIMC/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    if progress_cb:
        progress_cb("Modelo pronto.")
