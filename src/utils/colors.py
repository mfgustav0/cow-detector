from src.utils.constants import C
from reportlab.lib import colors

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

GREEN_RGB = hex2rgb(C["green"])
AMBER_RGB = hex2rgb(C["amber"])
RED_RGB   = hex2rgb(C["red"])

def _rl_color(hex_str):
    r, g, b = hex2rgb(hex_str)
    return colors.Color(r/255, g/255, b/255)

COL_GREEN  = _rl_color("#22C55E")
COL_AMBER  = _rl_color("#F59E0B")
COL_RED    = _rl_color("#EF4444")
COL_DARK   = _rl_color("#0F1E14")
COL_MID    = _rl_color("#162219")
COL_BORDER = _rl_color("#243B2C")
COL_LIGHT  = _rl_color("#E2F5E8")
COL_DIM    = _rl_color("#86A891")
COL_WHITE  = colors.white
COL_BLACK  = colors.black
