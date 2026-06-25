from src.detect import Detector, ensure_model
from src.report import gerar_relatorio
from src.utils.constants import C
import src.utils.colors

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import datetime
import io

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# DESENHO NO FRAME
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame: np.ndarray, det: dict, metrics: dict, diag: dict) -> np.ndarray:
    img    = frame.copy()
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    status = diag["status"]
    col    = src.utils.colors.GREEN_RGB if status == "SAUDÁVEL" else (src.utils.colors.AMBER_RGB if status == "ATENÇÃO" else src.utils.colors.RED_RGB)
    col_bgr = col[::-1]

    cv2.rectangle(img, (x1, y1), (x2, y2), col_bgr, 2)

    sz = 18
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx, cy), (cx + dx*sz, cy), col_bgr, 3)
        cv2.line(img, (cx, cy), (cx, cy + dy*sz), col_bgr, 3)

    label_txt = (
        f"{det['label'].upper()}  {det['confidence']*100:.0f}%  |  "
        f"{metrics['peso_kg']} kg  |  {metrics['altura_cm']} cm  |  "
        f"IMC {metrics['imc']:.2f}"
    )
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thick = 1
    (tw, th), _ = cv2.getTextSize(label_txt, font, scale, thick)
    pad = 5
    lbl_y1 = max(0, y1 - th - 2*pad - 2)
    cv2.rectangle(img, (x1, lbl_y1), (x1 + tw + 2*pad, y1), col_bgr, -1)
    cv2.putText(img, label_txt, (x1 + pad, y1 - pad),
                font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    h_img, w_img = img.shape[:2]
    hud_lines = [
        "BOVIMC VISION  v1.0",
        f"PESO EST: {metrics['peso_kg']} kg",
        f"ALTURA EST: {metrics['altura_cm']} cm",
        f"IMC: {metrics['imc']:.2f}",
    ]
    for i, line in enumerate(reversed(hud_lines)):
        cv2.putText(img, line, (w_img - 210, h_img - 8 - i * 16),
                    font, 0.38, col_bgr, 1, cv2.LINE_AA)

    return img


def frame_to_photoimage(frame_bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil)

# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE GRÁFICA
# ══════════════════════════════════════════════════════════════════════════════

class BovIMCApp(tk.Tk):

    PREVIEW_W = 680
    PREVIEW_H = 460

    def __init__(self):
        super().__init__()
        self.title("Detector de Tamanho, Peso e IMC Bovino")
        self.configure(bg=C["bg"])
        self.resizable(True, True)
        self.minsize(1100, 700)

        self._detector    = None
        self._cap         = None
        self._video_job   = None
        self._current_results = []

        # Sessão de análise: acumula dados para o relatório
        self._session: list = []          # lista de {frame_bgr, results, ts, source}
        self._last_annotated: np.ndarray | None = None  # último frame anotado (sem piscar)
        self._last_results: list = []     # últimos resultados (para re-renderizar frames sem análise)
        self._source_name: str = ""
        self._video_start_time: float = 0

        self._build_ui()
        self._load_model_async()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)
        self._build_header()
        self._build_preview_panel()
        self._build_results_panel()
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["panel"], pady=10)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        tk.Label(hdr, text="🐄",
                 bg=C["panel"], fg=C["green"],
                 font=("Courier New", 18, "bold")).grid(row=0, column=0, padx=20, sticky="w")
        tk.Label(hdr, text="Detecção Automática de Tamanho • Peso • IMC Bovino",
                 bg=C["panel"], fg=C["text_mid"],
                 font=("Courier New", 9)).grid(row=1, column=0, padx=20, sticky="w")

        btn_frame = tk.Frame(hdr, bg=C["panel"])
        btn_frame.grid(row=0, column=2, rowspan=2, padx=20, sticky="e")

        self._btn_img = self._make_btn(btn_frame, "📷  Abrir Imagem",
                                       self._open_image, C["green_dim"], C["green"])
        self._btn_img.pack(side="left", padx=6)

        self._btn_vid = self._make_btn(btn_frame, "🎬  Abrir Vídeo",
                                       self._open_video, C["green_dim"], C["green"])
        self._btn_vid.pack(side="left", padx=6)

        self._btn_stop = self._make_btn(btn_frame, "⏹  Parar e Gerar Relatório",
                                        self._stop_and_report, C["red_dim"], C["red"])
        self._btn_stop.pack(side="left", padx=6)
        self._btn_stop.configure(state="disabled")

    def _make_btn(self, parent, text, cmd, bg, fg):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, activebackground=fg, activeforeground=C["bg"],
                         font=("Courier New", 10, "bold"),
                         relief="flat", bd=0, padx=14, pady=7, cursor="hand2")

    def _build_preview_panel(self):
        frame = tk.Frame(self, bg=C["bg"])
        frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=12)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(frame, bg=C["panel"],
                                  highlightthickness=1, highlightbackground=C["border"],
                                  width=self.PREVIEW_W, height=self.PREVIEW_H)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._show_placeholder()

    def _show_placeholder(self):
        self._canvas.delete("all")
        cw = self._canvas.winfo_width() or self.PREVIEW_W
        ch = self._canvas.winfo_height() or self.PREVIEW_H
        cx, cy = (cw // 2 + 250), ch // 2
        self._canvas.create_text(cx, cy - 30, text="📁",
                                  font=("Arial", 48), fill=C["text_dim"])
        self._canvas.create_text(cx, cy + 30,
                                  text="Selecione uma imagem ou vídeo para iniciar a análise",
                                  font=("Courier New", 11), fill=C["text_dim"])
        self._canvas.create_text(cx, cy + 55,
                                  text="Formatos aceitos: JPG · PNG · BMP · MP4 · AVI · MOV · MKV",
                                  font=("Courier New", 9), fill=C["text_dim"])

    def _build_results_panel(self):
        outer = tk.Frame(self, bg=C["bg"])
        outer.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=12)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text="LAUDO AUTOMÁTICO",
                 bg=C["bg"], fg=C["text_dim"],
                 font=("Courier New", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

        scroll_frame = tk.Frame(outer, bg=C["bg"])
        scroll_frame.grid(row=1, column=0, sticky="nsew")
        scroll_frame.rowconfigure(0, weight=1)
        scroll_frame.columnconfigure(0, weight=1)

        self._results_text = tk.Text(
            scroll_frame,
            bg=C["card"], fg=C["text"],
            font=("Courier New", 10),
            relief="flat", bd=0, wrap="word",
            state="disabled", padx=10, pady=8, spacing3=4,
        )
        sb = ttk.Scrollbar(scroll_frame, command=self._results_text.yview)
        self._results_text.configure(yscrollcommand=sb.set)
        self._results_text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self._results_text.tag_configure("title",    foreground=C["green"],   font=("Courier New", 11, "bold"))
        self._results_text.tag_configure("section",  foreground=C["text_mid"],font=("Courier New", 9, "bold"))
        self._results_text.tag_configure("value",    foreground=C["white"],   font=("Courier New", 10, "bold"))
        self._results_text.tag_configure("ok",       foreground=C["green"],   font=("Courier New", 10, "bold"))
        self._results_text.tag_configure("warn",     foreground=C["amber"],   font=("Courier New", 10, "bold"))
        self._results_text.tag_configure("critical", foreground=C["red"],     font=("Courier New", 10, "bold"))
        self._results_text.tag_configure("dim",      foreground=C["text_dim"])
        self._results_text.tag_configure("normal",   foreground=C["text"])

        self._write_placeholder_results()

    def _write_placeholder_results(self):
        t = self._results_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", "  Nenhuma análise realizada ainda.\n\n", "dim")
        t.insert("end", "  Abra uma imagem ou vídeo para\n  detectar automaticamente:\n\n", "dim")
        for item in ["Presença de bovino", "Tamanho estimado (altura/comprimento)",
                     "Peso corporal estimado", "IMC Bovino",
                     "Score de saúde", "Alertas automáticos"]:
            t.insert("end", f"  • {item}\n", "dim")
        t.configure(state="disabled")

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C["panel"], height=26)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(1, weight=1)

        self._lbl_status = tk.Label(bar, text="● Carregando modelo…",
                                    bg=C["panel"], fg=C["amber"],
                                    font=("Courier New", 9), anchor="w")
        self._lbl_status.grid(row=0, column=0, padx=10, sticky="w")

        self._lbl_fps = tk.Label(bar, text="",
                                  bg=C["panel"], fg=C["text_dim"],
                                  font=("Courier New", 9), anchor="e")
        self._lbl_fps.grid(row=0, column=2, padx=10, sticky="e")

        tk.Label(bar, text="UFSC Araranguá · CIT7596 · 2026.1",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Courier New", 8)).grid(row=0, column=1, sticky="e", padx=10)

    # ── Modelo ────────────────────────────────────────────────────────────────

    def _load_model_async(self):
        def _run():
            try:
                ensure_model(lambda m: self._set_status(f"● {m}", C["amber"]))
                self._detector = Detector()
                self._set_status("● Modelo pronto — aguardando arquivo", C["green"])
                self._btn_img.configure(state="normal")
                self._btn_vid.configure(state="normal")
            except Exception as e:
                self._set_status(f"● Erro ao carregar modelo: {e}", C["red"])

        self._btn_img.configure(state="disabled")
        self._btn_vid.configure(state="disabled")
        threading.Thread(target=_run, daemon=True).start()

    # ── Abertura de arquivo ───────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("Todos", "*.*")]
        )
        if not path:
            return
        self._stop_video(report=False)
        self._session.clear()
        self._source_name = os.path.basename(path)
        self._set_status(f"● Analisando: {self._source_name}", C["amber"])
        threading.Thread(target=self._process_image, args=(path,), daemon=True).start()

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Selecionar vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("Todos", "*.*")]
        )
        if not path:
            return
        self._stop_video(report=False)
        self._session.clear()
        self._source_name = os.path.basename(path)
        self._start_video(path)

    # ── Processamento de imagem ───────────────────────────────────────────────

    def _process_image(self, path: str):
        frame = cv2.imread(path)
        if frame is None:
            self._set_status("● Erro: não foi possível ler a imagem.", C["red"])
            return

        h, w  = frame.shape[:2]
        dets  = self._detector.detect(frame)

        results    = []
        annotated  = frame.copy()
        for det in dets:
            metrics = Detector.estimate_metrics(det, h, w)
            diag    = Detector.diagnostico(metrics)
            results.append({"det": det, "metrics": metrics, "diag": diag})
            annotated = draw_overlay(annotated, det, metrics, diag)

        self._last_annotated = annotated
        self._last_results   = results

        # Guarda na sessão para o relatório
        ts = time.strftime("%H:%M:%S")
        self._session.append({
            "frame_bgr": annotated,
            "results":   results,
            "ts":        ts,
            "source":    self._source_name,
        })

        cw = self._canvas.winfo_width()  or self.PREVIEW_W
        ch = self._canvas.winfo_height() or self.PREVIEW_H
        photo = frame_to_photoimage(annotated, cw, ch)

        self.after(0, lambda: self._update_canvas(photo))
        self.after(0, lambda: self._update_results(results, self._source_name))
        self.after(0, lambda: self._btn_stop.configure(state="normal"))

        n  = len(results)
        st = results[0]["diag"]["status"] if results else "—"
        col = C["green"] if st == "SAUDÁVEL" else (C["amber"] if st == "ATENÇÃO" else C["red"])
        self._set_status(
            f"● {n} bovino(s) detectado(s)  |  Status: {st}  |  {self._source_name}",
            col if n > 0 else C["amber"]
        )

    # ── Vídeo ─────────────────────────────────────────────────────────────────

    def _start_video(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo.")
            return
        self._btn_stop.configure(state="normal")
        self._set_status(f"● Reproduzindo: {self._source_name}", C["green"])
        self._t_last_frame    = time.time()
        self._video_start_time = time.time()
        self._frame_count     = 0
        self._last_annotated  = None
        self._last_results    = []
        self._video_loop()

    def _video_loop(self):
        if self._cap is None or not self._cap.isOpened():
            return

        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_job = self.after(30, self._video_loop)
            return

        self._frame_count += 1

        # Detecta a cada 5 frames; nos demais reutiliza os overlays anteriores
        if self._frame_count % 5 == 0:
            h, w  = frame.shape[:2]
            dets  = self._detector.detect(frame)
            results = []
            annotated = frame.copy()
            for det in dets:
                metrics = Detector.estimate_metrics(det, h, w)
                diag    = Detector.diagnostico(metrics)
                results.append({"det": det, "metrics": metrics, "diag": diag})
                annotated = draw_overlay(annotated, det, metrics, diag)

            self._last_annotated = annotated
            self._last_results   = results

            # Salva 1 frame por segundo para o relatório (a cada ~30 frames)
            if self._frame_count % 30 == 0:
                novo_item = {
                    "frame_bgr": annotated.copy(),
                    "results": results,
                    "ts": time.strftime("%H:%M:%S"),
                    "source": self._source_name,
                }

                if not self._session:
                    self._session.append(novo_item)
                else:
                    ultimo = self._session[-1]

                    if not self._same_results(
                        ultimo["results"],
                        results
                    ):
                        self._session.append(novo_item)

            self.after(0, lambda r=results: self._update_results(r, self._source_name, video=True))
            display_frame = annotated
        else:
            # Reutiliza o último frame anotado → sem piscar
            display_frame = self._last_annotated if self._last_annotated is not None else frame

        # FPS
        now = time.time()
        fps = 1.0 / max(now - self._t_last_frame, 1e-6)
        self._t_last_frame = now
        self._lbl_fps.configure(text=f"FPS: {fps:.1f}  |  Amostras: {len(self._session)}")

        cw = self._canvas.winfo_width()  or self.PREVIEW_W
        ch = self._canvas.winfo_height() or self.PREVIEW_H
        photo = frame_to_photoimage(display_frame, cw, ch)
        self._update_canvas(photo)

        self._video_job = self.after(16, self._video_loop)

    def _same_results(self, r1, r2):
        if len(r1) != len(r2):
            return False

        for a, b in zip(r1, r2):
            m1 = a["metrics"]
            m2 = b["metrics"]

            if (
                m1["peso_kg"] != m2["peso_kg"] or
                m1["altura_cm"] != m2["altura_cm"] or
                m1["imc"] != m2["imc"]
            ):
                return False

        return True

    def _stop_video(self, report=True):
        """Para o vídeo. Se report=True e houver sessão, pergunta e gera PDF."""
        if self._video_job:
            self.after_cancel(self._video_job)
            self._video_job = None

        duracao = time.time() - self._video_start_time if self._video_start_time else 0

        if self._cap:
            self._cap.release()
            self._cap = None

        self._lbl_fps.configure(text="")

        if report and self._session:
            self._ask_and_generate_report(duracao)
        else:
            self._btn_stop.configure(state="disabled")

    def _stop_and_report(self):
        """Chamado pelo botão Parar e Gerar Relatório."""
        if self._cap:
            # Estava em vídeo → para e gera
            self._stop_video(report=True)
        elif self._session:
            # Era imagem → gera direto
            self._ask_and_generate_report(0)
        self._btn_stop.configure(state="disabled")

    def _ask_and_generate_report(self, duracao: float):
        """Pergunta onde salvar e gera o PDF em background."""
        if not self._session:
            messagebox.showinfo("Relatório", "Nenhuma análise disponível para gerar relatório.")
            return

        default_name = f"bovimc_relatorio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path = filedialog.asksaveasfilename(
            title="Salvar Relatório PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF", "*.pdf"), ("Todos", "*.*")],
        )
        if not path:
            return

        self._set_status("● Gerando relatório PDF…", C["amber"])

        session_snapshot = list(self._session)
        source           = self._source_name

        def _gen():
            try:
                gerar_relatorio(
                    session_data=session_snapshot,
                    output_path=path,
                    fonte=source,
                    duracao_s=duracao,
                )
                self._set_status(f"● Relatório salvo: {os.path.basename(path)}", C["green"])
                self.after(0, lambda: messagebox.showinfo(
                    "Relatório Gerado",
                    f"PDF salvo com sucesso!\n\n{path}\n\n"
                    f"{len(session_snapshot)} frames analisados."
                ))
            except Exception as e:
                self._set_status(f"● Erro ao gerar PDF: {e}", C["red"])
                self.after(0, lambda: messagebox.showerror("Erro", f"Falha ao gerar PDF:\n{e}"))

        threading.Thread(target=_gen, daemon=True).start()

    # ── Canvas / Resultados ───────────────────────────────────────────────────

    def _update_canvas(self, photo):
        self._canvas.delete("all")
        cw = self._canvas.winfo_width()  or self.PREVIEW_W
        ch = self._canvas.winfo_height() or self.PREVIEW_H
        self._canvas_photo = photo
        self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)

    def _update_results(self, results: list, source: str, video: bool = False):
        t = self._results_text
        t.configure(state="normal")
        t.delete("1.0", "end")

        ts = time.strftime("%H:%M:%S")
        t.insert("end", f"  {'[VÍDEO]' if video else '[IMAGEM]'}  {source}\n", "section")
        t.insert("end", f"  Análise: {ts}\n\n", "dim")

        if not results:
            t.insert("end", "  ✗  Nenhum bovino detectado.\n\n", "warn")
            t.insert("end", "  Dicas:\n", "dim")
            t.insert("end", "  • Use imagem lateral do animal\n", "dim")
            t.insert("end", "  • Boa iluminação e fundo contrastante\n", "dim")
            t.insert("end", "  • Animal deve estar em destaque\n", "dim")
            t.configure(state="disabled")
            return

        for idx, r in enumerate(results, 1):
            det     = r["det"]
            metrics = r["metrics"]
            diag    = r["diag"]
            status  = diag["status"]
            score   = diag["score"]
            col_tag = "ok" if status == "SAUDÁVEL" else ("warn" if status == "ATENÇÃO" else "critical")

            t.insert("end", f"  ── Animal #{idx}  {det['label'].upper()}", "title")
            t.insert("end", f"  confiança {det['confidence']*100:.0f}%\n\n", "dim")

            t.insert("end",  "  SCORE DE SAÚDE\n", "section")
            t.insert("end", f"  {score}/100  —  ", "value")
            t.insert("end", f"{status}\n\n", col_tag)

            filled = round(score / 5)
            bar    = "█" * filled + "░" * (20 - filled)
            t.insert("end", f"  [{bar}]\n\n", col_tag)

            t.insert("end", "  MEDIDAS ESTIMADAS\n", "section")
            imc = metrics["imc"]
            rows = [
                ("Peso (±30%)",        f"{metrics['peso_kg']} kg",       ""),
                ("Altura (cernelha)", f"{metrics['altura_cm']} cm",     ""),
                ("Comprimento",       f"{metrics['comprimento_cm']} cm",""),
                ("Robustez (w/h)",    f"{metrics['wh_ratio']:.2f}",     "  (gordo>1.05, magro<0.80)"),
                ("IMC Bovino",        f"{imc:.2f}",                     "  (normal: 2.00–2.80)"),
            ]
            for label, val, note in rows:
                t.insert("end", f"  {label:<20}", "dim")
                t.insert("end", val, "value")
                t.insert("end", f"{note}\n", "dim")

            t.insert("end", "\n  INTERPRETAÇÃO IMC\n", "section")
            if imc < 1.70:   interp, itag = "Subnutrição severa",          "critical"
            elif imc < 2.00: interp, itag = "Abaixo do ideal",             "warn"
            elif imc <= 2.80:interp, itag = "Faixa ideal",                 "ok"
            elif imc <= 3.20:interp, itag = "Levemente acima do ideal",    "warn"
            else:             interp, itag = "Sobrepeso",                   "warn"
            t.insert("end", f"  {interp}\n\n", itag)

            if diag["alertas"]:
                t.insert("end", f"  ALERTAS ({len(diag['alertas'])})\n", "section")
                for level, msg in diag["alertas"]:
                    sym  = "🔴" if level == "CRÍTICO" else "🟡"
                    atag = "critical" if level == "CRÍTICO" else "warn"
                    t.insert("end", f"  {sym} ", "normal")
                    t.insert("end", f"{level}  ", atag)
                    t.insert("end", f"{msg}\n", "normal")
                t.insert("end", "\n")
            else:
                t.insert("end", "  ✓ Sem alertas\n\n", "ok")

            t.insert("end", "  " + "─" * 36 + "\n\n", "dim")

        amostras = len(self._session)
        if amostras:
            t.insert("end", f"  📊 Amostras na sessão: {amostras}\n", "dim")
            t.insert("end",  "  Clique em Parar e Gerar Relatório para exportar PDF.\n\n", "dim")

        t.insert("end", "\n  ⚠ Estimativas por visão computacional.\n", "dim")
        t.insert("end",   "  Consulte um médico veterinário.\n", "dim")
        t.insert("end",   "  BovIMC · UFSC · CIT7596 · 2026.1\n", "dim")

        t.see("1.0")
        t.configure(state="disabled")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str):
        self.after(0, lambda: self._lbl_status.configure(text=msg, fg=color))

    def on_close(self):
        self._stop_video(report=False)
        self.destroy()

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = BovIMCApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
