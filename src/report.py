import cv2
import math
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import PageBreak
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
import src.utils.colors

def _status_color(status):
    return src.utils.colors.COL_GREEN if status == "SAUDÁVEL" else (src.utils.colors.COL_AMBER if status == "ATENÇÃO" else src.utils.colors.COL_RED)

def _gauge_image(score: int, status: str, size=140) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(size/72, size/72*0.6), dpi=72)
    fig.patch.set_facecolor("#1C2E22")
    ax.set_facecolor("#1C2E22")
    col = "#22C55E" if status == "SAUDÁVEL" else ("#F59E0B" if status == "ATENÇÃO" else "#EF4444")

    theta_full = np.linspace(math.pi, 0, 200)
    ax.plot(np.cos(theta_full), np.sin(theta_full), color="#243B2C", lw=14, solid_capstyle="round")
    frac = score / 100.0
    theta_val = np.linspace(math.pi, math.pi - frac * math.pi, 200)
    ax.plot(np.cos(theta_val), np.sin(theta_val), color=col, lw=14, solid_capstyle="round")

    ax.text(0, -0.1, str(score), ha="center", va="center",
            fontsize=22, fontweight="bold", color=col)
    ax.text(0, -0.5, status, ha="center", va="center",
            fontsize=7, fontweight="bold", color=col)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.65, 1.15)
    ax.axis("off")
    fig.tight_layout(pad=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight", facecolor="#1C2E22")
    plt.close(fig)
    buf.seek(0)
    return buf

def _frame_thumb(frame_bgr: np.ndarray, max_w=400, max_h=240) -> io.BytesIO:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


def gerar_relatorio(
    session_data: list,   # lista de {frame_bgr, results: [{det, metrics, diag}], ts, source}
    output_path: str,
    fonte: str = "",
    duracao_s: float = 0,
):
    """Gera PDF de relatório com todos os frames/imagens analisados."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="Relatório",
    )

    W, H = A4
    content_w = W - 4*cm

    # ── Estilos ────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle    = S("sTitle",    fontSize=22, fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_GREEN, spaceAfter=4, alignment=TA_CENTER)
    sSubtitle = S("sSub",     fontSize=10, fontName="Helvetica",
                  textColor=src.utils.colors.COL_DIM,   spaceAfter=2, alignment=TA_CENTER)
    sSection  = S("sSect",    fontSize=12, fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_GREEN, spaceBefore=10, spaceAfter=4)
    sLabel    = S("sLbl",     fontSize=9,  fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_DIM)
    sValue    = S("sVal",     fontSize=10, fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_WHITE)
    sNormal   = S("sNorm",    fontSize=9,  fontName="Helvetica",
                  textColor=src.utils.colors.COL_LIGHT)
    sAlert    = S("sAlt",     fontSize=9,  fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_RED,   spaceBefore=2)
    sAlertWarn= S("sAltW",    fontSize=9,  fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_AMBER, spaceBefore=2)
    sFooter   = S("sFoot",    fontSize=7,  fontName="Helvetica",
                  textColor=src.utils.colors.COL_DIM,   alignment=TA_CENTER)
    sOk       = S("sOk",      fontSize=9,  fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_GREEN)
    sCaution  = S("sCaut",    fontSize=9,  fontName="Helvetica-Bold",
                  textColor=src.utils.colors.COL_AMBER)

    def dark_table_style(extra=None):
        base_cmds = [
            ("BACKGROUND", (0,0), (-1,0), src.utils.colors.COL_MID),
            ("BACKGROUND", (0,1), (-1,-1), src.utils.colors.COL_DARK),
            ("TEXTCOLOR",  (0,0), (-1,-1), src.utils.colors.COL_LIGHT),
            ("FONTNAME",   (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.5, src.utils.colors.COL_BORDER),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [src.utils.colors.COL_DARK, src.utils.colors._rl_color("#13251A")]),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ]
        if extra:
            base_cmds.extend(extra)
        return TableStyle(base_cmds)

    story = []

    sub_data = [[
        Paragraph("Relatório de Detecção Automática — Tamanho • Peso • IMC Bovino", sSubtitle),
    ]]
    if fonte:
        sub_data.append([Paragraph(f"Fonte: {fonte}", sSubtitle)])
    if duracao_s > 0:
        sub_data.append([Paragraph(f"Duração analisada: {duracao_s:.1f} s", sSubtitle)])

    sub_tbl = Table(sub_data, colWidths=[content_w])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), src.utils.colors.COL_MID),
        ("BOX",         (0,0), (-1,-1), 1, src.utils.colors.COL_GREEN),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING",(0,0), (-1,-1), 12),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 12),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 0.4*cm))

    # Linha separadora
    story.append(HRFlowable(width="100%", thickness=1, color=src.utils.colors.COL_GREEN, spaceAfter=8))

    # ── Rodapé de identificação ─────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))

    # ── Resumo executivo ─────────────────────────────────────────────────────
    total_frames    = len(session_data)
    total_animais   = sum(len(s["results"]) for s in session_data)
    frames_c_det    = sum(1 for s in session_data if s["results"])

    all_results = []
    for s in session_data:
        all_results.extend(s["results"])

    n_saud  = sum(1 for r in all_results if r["diag"]["status"] == "SAUDÁVEL")
    n_aten  = sum(1 for r in all_results if r["diag"]["status"] == "ATENÇÃO")
    n_crit  = sum(1 for r in all_results if r["diag"]["status"] == "CRÍTICO")
    score_med = (sum(r["diag"]["score"] for r in all_results) / len(all_results)) if all_results else 0
    imc_med   = (sum(r["metrics"]["imc"] for r in all_results) / len(all_results)) if all_results else 0
    peso_med  = (sum(r["metrics"]["peso_kg"] for r in all_results) / len(all_results)) if all_results else 0

    story.append(Paragraph("RESUMO EXECUTIVO", sSection))
    story.append(HRFlowable(width="100%", thickness=0.5, color=src.utils.colors.COL_BORDER, spaceAfter=6))

    res_data = [
        ["Métrica", "Valor"],
        ["Frames/imagens analisados",        str(total_frames)],
        ["Frames com detecção",              str(frames_c_det)],
        ["Total de animais detectados",      str(total_animais)],
        ["Animais SAUDÁVEIS",                str(n_saud)],
        ["Animais em ATENÇÃO",               str(n_aten)],
        ["Animais em estado CRÍTICO",        str(n_crit)],
        ["Score médio de saúde",             f"{score_med:.1f} / 100"],
        ["IMC médio estimado",               f"{imc_med:.2f}"],
        ["Peso médio estimado",              f"{peso_med:.1f} kg"],
    ]
    res_tbl = Table(res_data, colWidths=[content_w*0.6, content_w*0.4])
    res_tbl.setStyle(dark_table_style([
        ("FONTNAME", (1,1), (1,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, res_data.index(["Animais SAUDÁVEIS", str(n_saud)])),
                      (1, res_data.index(["Animais SAUDÁVEIS", str(n_saud)])), src.utils.colors.COL_GREEN),
        ("TEXTCOLOR", (1, res_data.index(["Animais em ATENÇÃO", str(n_aten)])),
                      (1, res_data.index(["Animais em ATENÇÃO", str(n_aten)])), src.utils.colors.COL_AMBER),
        ("TEXTCOLOR", (1, res_data.index(["Animais em estado CRÍTICO", str(n_crit)])),
                      (1, res_data.index(["Animais em estado CRÍTICO", str(n_crit)])), src.utils.colors.COL_RED),
    ]))
    story.append(res_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Detalhes por frame ──────────────────────────────────────────────────
    story.append(Paragraph("ANÁLISES INDIVIDUAIS", sSection))
    story.append(HRFlowable(width="100%", thickness=0.5, color=src.utils.colors.COL_BORDER, spaceAfter=6))

    for s_idx, sess in enumerate(session_data, 1):
        frame_bgr = sess["frame_bgr"]
        results   = sess["results"]
        ts        = sess.get("ts", "—")
        source    = sess.get("source", "—")

        # Cabeçalho do frame
        frame_header = Table(
            [[Paragraph(f"Frame #{s_idx}  —  {source}  —  {ts}", sLabel)]],
            colWidths=[content_w]
        )
        frame_header.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), src.utils.colors.COL_MID),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("BOX",          (0,0), (-1,-1), 0.5, src.utils.colors.COL_BORDER),
        ]))
        story.append(frame_header)

        if not results:
            story.append(Paragraph("  Nenhum bovino detectado neste frame.", sCaution))
            story.append(Spacer(1, 0.2*cm))
            continue

        for a_idx, r in enumerate(results, 1):
            det     = r["det"]
            metrics = r["metrics"]
            diag    = r["diag"]
            status  = diag["status"]
            score   = diag["score"]
            s_col   = _status_color(status)

            # Thumb + gauge lado a lado
            thumb_buf = _frame_thumb(frame_bgr, 280, 180)
            gauge_buf = _gauge_image(score, status, size=130)

            thumb_img = RLImage(thumb_buf, width=7*cm, height=4.5*cm)
            gauge_img = RLImage(gauge_buf, width=4.5*cm, height=2.7*cm)

            side_w = content_w - 7*cm - 0.4*cm
            animal_label = Table(
                [[Paragraph(f"Animal #{a_idx} — {det['label']}  ({det['confidence']*100:.0f}% conf.)", sValue)],
                 [gauge_img],
                 [Paragraph(f"Score: {score}/100", sValue)]],
                colWidths=[side_w]
            )
            animal_label.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,-1), src.utils.colors.COL_MID),
                ("ALIGN",        (0,0), (-1,-1), "CENTER"),
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",   (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ("BOX",          (0,0), (-1,-1), 0.5, s_col),
            ]))

            vis_row = Table(
                [[thumb_img, animal_label]],
                colWidths=[7*cm, side_w]
            )
            vis_row.setStyle(TableStyle([
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING",  (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 0),
                ("TOPPADDING",   (0,0), (-1,-1), 0),
                ("BOTTOMPADDING",(0,0), (-1,-1), 0),
            ]))
            story.append(Spacer(1, 0.2*cm))
            story.append(vis_row)
            story.append(Spacer(1, 0.15*cm))

            # Tabela de métricas
            imc = metrics["imc"]
            if imc < 1.70:   interp = "Subnutrição severa"
            elif imc < 2.00: interp = "Abaixo do ideal"
            elif imc <= 2.80:interp = "Faixa ideal (2.00–2.80)"
            elif imc <= 3.20:interp = "Levemente acima do ideal"
            else:             interp = "Sobrepeso"

            met_data = [
                ["Métrica",              "Valor",                     "Referência"],
                ["Peso estimado (±30%)",  f"{metrics['peso_kg']} kg", "350–700 kg (adulto)"],
                ["Altura (cernelha)",     f"{metrics['altura_cm']} cm","120–160 cm (adulto)"],
                ["Comprimento",           f"{metrics['comprimento_cm']} cm","160–220 cm"],
                ["IMC Bovino",            f"{imc:.2f}",               "2.00–2.80 (ideal)"],
                ["Interpretação IMC",     interp,                     "—"],
                ["Cond. Corporal (BCS)",  f"{metrics['bcs']}/5",      "3/5 (ideal)"],
                ["Robustez visual (w/h)", f"{metrics['wh_ratio']:.2f}","<0.80 magro | >1.05 gordo"],
                ["Status geral",          status,                     "—"],
            ]
            extra_style = [
                # Colorir linha IMC
                ("TEXTCOLOR", (1,4), (1,4),
                    src.utils.colors.COL_GREEN if 2.0<=imc<=2.8 else (src.utils.colors.COL_AMBER if imc<2.0 or imc<=3.2 else src.utils.colors.COL_RED)),
                # Colorir linha status
                ("TEXTCOLOR", (1,8), (1,8), s_col),
                ("FONTNAME",  (1,8), (1,8), "Helvetica-Bold"),
                # Highlight interpretação
                ("TEXTCOLOR", (1,5), (1,5),
                    src.utils.colors.COL_GREEN if "ideal" in interp.lower() and "acima" not in interp.lower()
                    else src.utils.colors.COL_AMBER),
            ]
            met_tbl = Table(met_data, colWidths=[content_w*0.38, content_w*0.3, content_w*0.32])
            met_tbl.setStyle(dark_table_style(extra_style))
            story.append(met_tbl)

            # Alertas
            story.append(Spacer(1, 0.15*cm))
            if diag["alertas"]:
                al_data = [["Alertas Automáticos"]]
                for level, msg in diag["alertas"]:
                    al_data.append([f"{'🔴' if level=='CRÍTICO' else '🟡'}  {level}: {msg}"])
                al_tbl = Table(al_data, colWidths=[content_w])
                al_cmds = [
                    ("BACKGROUND",   (0,0), (-1,0),  src.utils.colors.COL_MID),
                    ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
                    ("TEXTCOLOR",    (0,0), (-1,0),  src.utils.colors.COL_AMBER),
                    ("BACKGROUND",   (0,1), (-1,-1), src.utils.colors._rl_color("#1A0A0A")),
                    ("FONTNAME",     (0,1), (-1,-1), "Helvetica-Bold"),
                    ("TEXTCOLOR",    (0,1), (-1,-1), src.utils.colors.COL_RED),
                    ("GRID",         (0,0), (-1,-1), 0.5, src.utils.colors.COL_BORDER),
                    ("LEFTPADDING",  (0,0), (-1,-1), 8),
                    ("TOPPADDING",   (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ]
                # Alertas de atenção em âmbar
                for i, (level, _) in enumerate(diag["alertas"], 1):
                    if level != "CRÍTICO":
                        al_cmds.append(("TEXTCOLOR", (0,i), (0,i), src.utils.colors.COL_AMBER))
                al_tbl.setStyle(TableStyle(al_cmds))
                story.append(al_tbl)
            else:
                ok_data = [["✓  Sem alertas — animal dentro dos parâmetros normais"]]
                ok_tbl  = Table(ok_data, colWidths=[content_w])
                ok_tbl.setStyle(TableStyle([
                    ("BACKGROUND",   (0,0), (-1,-1), src.utils.colors._rl_color("#0D2A15")),
                    ("TEXTCOLOR",    (0,0), (-1,-1), src.utils.colors.COL_GREEN),
                    ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
                    ("GRID",         (0,0), (-1,-1), 0.5, src.utils.colors.COL_BORDER),
                    ("LEFTPADDING",  (0,0), (-1,-1), 8),
                    ("TOPPADDING",   (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ]))
                story.append(ok_tbl)

            story.append(Spacer(1, 0.3*cm))

        story.append(HRFlowable(width="100%", thickness=0.3, color=src.utils.colors.COL_BORDER, spaceAfter=4))

    # ── Rodapé legal ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=src.utils.colors.COL_GREEN))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠  Este relatório é gerado automaticamente por visão computacional e deve ser usado "
        "como ferramenta auxiliar. As estimativas de peso, altura e IMC são baseadas em proporções "
        "do bounding box e podem apresentar variações conforme distância da câmera, ângulo e "
        "iluminação. Consulte um médico veterinário habilitado para diagnóstico clínico definitivo.",
        sFooter
    ))
    story.append(Spacer(1, 0.15*cm))

    doc.build(story)
