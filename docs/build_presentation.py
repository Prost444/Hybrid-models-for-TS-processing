#!/usr/bin/env python3
"""ВКР defense presentation — post-предзащита revision.

Addresses committee feedback:
  * dedicated model-architecture slides (pipelines + formulas, Lykov style);
  * explicit MEANING of learnable vs fixed decomposition;
  * explanation of what "ранг" is and how it is computed;
  * forecast charts connect to the train end (no 1-step gap);
  * a standalone large forecast slide;
  * every chart legend duplicated as text on free slide space;
  * a "when to build complex models" recommendation matrix;
  * 3 economic slides (shared startup «ГМ-ПВР»).
Calibri — portable to PowerPoint.
"""
import struct
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "thesis" / "figures"
FRM = ROOT / "docs" / "assets" / "formulas"
OUT = ROOT / "docs" / "Презентация_Лазарев_ВКР.pptx"

NAVY    = RGBColor(0x1B, 0x26, 0x49)
NAVY2   = RGBColor(0x2E, 0x3D, 0x6B)
ACCENT  = RGBColor(0xE0, 0x7A, 0x5F)
ACC_DK  = RGBColor(0xC2, 0x5A, 0x40)
INK     = RGBColor(0x29, 0x2E, 0x3A)
MUTE    = RGBColor(0x66, 0x6E, 0x82)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
PANEL   = RGBColor(0xEE, 0xF1, 0xF7)
PANEL2  = RGBColor(0xF6, 0xF7, 0xFB)
LBLUE   = RGBColor(0xB7, 0xC2, 0xDE)
RULE    = RGBColor(0xD7, 0xDD, 0xEA)
GREEN   = RGBColor(0x2E, 0x7D, 0x6B)

# chart line colours (match make_defense_charts.py)
C_PATCH = RGBColor(0x27, 0xAE, 0x60)
C_FED   = RGBColor(0x16, 0xA0, 0x85)
C_HELF  = RGBColor(0xE9, 0x1E, 0x63)
C_AUTO  = RGBColor(0xE6, 0x7E, 0x22)
C_DLIN  = RGBColor(0x34, 0x98, 0xDB)
C_NBE   = RGBColor(0x8E, 0x44, 0xAD)
C_TIMES = RGBColor(0x9B, 0x59, 0xB6)
C_ARIMA = RGBColor(0xC0, 0x39, 0x2B)
C_ETS   = RGBColor(0xD4, 0x88, 0x1A)
C_GREY  = RGBColor(0x95, 0xA3, 0xB2)
C_BLACK = RGBColor(0x16, 0x18, 0x1D)

FONT = "Calibri"
SW, SH = Inches(13.333), Inches(7.5)
TOTAL = 22


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    return 1600, 900


def solid(shape, color, line=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False


def rect(slide, x, y, w, h, color, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    solid(sh, color, line)
    return sh


def tb(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, ls=1.0, sa=0):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = ls
        if sa:
            p.space_after = Pt(sa)
        for (t, sz, c, b, it) in para:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(sz)
            r.font.color.rgb = c
            r.font.bold = b
            r.font.italic = it
    return box


def header(slide, title, subtitle, page):
    rect(slide, 0, 0, SW, SH, WHITE)
    rect(slide, 0, 0, SW, Inches(1.46), NAVY)
    rect(slide, 0, Inches(1.46), SW, Pt(3), ACCENT)
    tb(slide, Inches(0.7), Inches(0.26), Inches(10.6), Inches(0.72),
       [[(title, 26, WHITE, True, False)]], ls=1.0)
    if subtitle:
        tb(slide, Inches(0.72), Inches(0.99), Inches(10.4), Inches(0.4),
           [[(subtitle, 14, LBLUE, False, True)]])
    chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(11.78), Inches(0.42), Inches(1.12),
                                  Inches(0.5))
    solid(chip, NAVY2)
    txt = page if isinstance(page, str) else f"{page:02d} / {TOTAL}"
    tb(slide, Inches(11.78), Inches(0.52), Inches(1.12), Inches(0.34),
       [[(txt, 12.5, WHITE, True, False)]], align=PP_ALIGN.CENTER)
    rect(slide, Inches(0.7), Inches(7.04), Inches(11.93), Pt(0.75), RULE)
    tb(slide, Inches(0.7), Inches(7.1), Inches(9.0), Inches(0.3),
       [[("Прогнозирование временных рядов · ВКР · Лазарев А.К. · 2026",
          10, MUTE, False, False)]])


def panel(slide, x, y, w, h, title, items, fill=PANEL, size=14, gap=8,
          tcolor=NAVY):
    rect(slide, x, y, w, h, fill)
    rect(slide, x, y, Pt(4), h, ACCENT)
    tb(slide, x + Inches(0.28), y + Inches(0.16), w - Inches(0.4),
       Inches(0.4), [[(title, 15, tcolor, True, False)]])
    runs = []
    for it in items:
        if isinstance(it, tuple):
            runs.append([("▪  ", size, ACCENT, True, False),
                         (it[0], size, INK, True, False),
                         (it[1], size, INK, False, False)])
        else:
            runs.append([("▪  ", size, ACCENT, True, False),
                         (it, size, INK, False, False)])
    tb(slide, x + Inches(0.28), y + Inches(0.64), w - Inches(0.5),
       h - Inches(0.78), runs, ls=1.1, sa=gap)


def conclusion_band(slide, text, y=Inches(6.32)):
    rect(slide, Inches(0.7), y, Inches(11.93), Inches(0.62), PANEL)
    rect(slide, Inches(0.7), y, Pt(4), Inches(0.62), ACCENT)
    tb(slide, Inches(1.0), y + Inches(0.08), Inches(1.7), Inches(0.46),
       [[("ВЫВОД", 13, ACC_DK, True, False)]], anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, Inches(2.15), y, Inches(10.3), Inches(0.62),
       [[(text, 14, INK, False, False)]], anchor=MSO_ANCHOR.MIDDLE, ls=1.05)


def keystat(slide, x, y, number, label, w=Inches(3.5), col=ACCENT, ns=34):
    tb(slide, x, y, w, Inches(0.8), [[(number, ns, col, True, False)]])
    tb(slide, x, y + Inches(0.66), w, Inches(0.8),
       [[(label, 12.5, MUTE, False, False)]], ls=1.06)


def img_fit(slide, path, x, y, mw, mh):
    w, h = png_size(path)
    ar = w / h
    if ar > mw / mh:
        nw, nh = mw, int(mw / ar)
    else:
        nh, nw = mh, int(mh * ar)
    slide.shapes.add_picture(str(path), x + (mw - nw) // 2,
                             y + (mh - nh) // 2, nw, nh)


def formula(slide, name, x, y, h):
    """Embed a formula PNG scaled to height h; returns its rendered width."""
    path = FRM / f"{name}.png"
    w_px, h_px = png_size(path)
    nw = int(h * (w_px / h_px))
    slide.shapes.add_picture(str(path), x, y, height=h)
    return nw


def pipeline(slide, x, y, w, boxes, accent_idx=(), bh=Inches(0.92), fs=11):
    """Native horizontal box-and-arrow pipeline (Lykov style)."""
    n = len(boxes)
    gap = Inches(0.16)
    bw = int((w - gap * (n - 1)) / n)
    for i, label in enumerate(boxes):
        bx = x + i * (bw + gap)
        color = ACCENT if i in accent_idx else NAVY
        rect(slide, bx, y, bw, bh, color)
        lines = label.split("\n")
        runs = [[(ln, fs, WHITE, True, False)] for ln in lines]
        tb(slide, bx + Inches(0.04), y, bw - Inches(0.08), bh, runs,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, ls=1.0)
        if i < n - 1:
            ar = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                        bx + bw + Emu(int(gap * 0.12)),
                                        y + bh / 2 - Inches(0.08),
                                        Emu(int(gap * 0.76)), Inches(0.16))
            solid(ar, NAVY2)


def legend_text(slide, x, y, w, h, title, entries):
    """Duplicate a chart legend as text (committee asked for this)."""
    rect(slide, x, y, w, h, PANEL2)
    rect(slide, x, y, Pt(4), h, ACCENT)
    tb(slide, x + Inches(0.26), y + Inches(0.16), w - Inches(0.4), Inches(0.4),
       [[(title, 14.5, NAVY, True, False)]])
    runs = []
    for col, bold, rest in entries:
        runs.append([("▬  ", 15, col, True, False),
                     (bold, 13, INK, True, False),
                     (rest, 13, INK, False, False)])
    tb(slide, x + Inches(0.26), y + Inches(0.62), w - Inches(0.46),
       h - Inches(0.74), runs, ls=1.08, sa=7)


prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def new():
    return prs.slides.add_slide(BLANK)


# ===== 1 — Title ===========================================================
s = new()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 0, Inches(0.22), SH, ACCENT)
rect(s, Inches(0.9), Inches(4.62), Inches(4.4), Pt(2.5), ACCENT)
tb(s, Inches(0.95), Inches(0.66), Inches(11.4), Inches(0.4),
   [[("МИРЭА — Российский технологический университет", 15, WHITE, False, False)]])
tb(s, Inches(0.95), Inches(1.04), Inches(11.4), Inches(0.4),
   [[("Институт искусственного интеллекта · Кафедра высшей математики",
      13, LBLUE, False, True)]])
tb(s, Inches(0.95), Inches(1.95), Inches(11.4), Inches(0.4),
   [[("ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА", 14, ACCENT, True, False)]])
tb(s, Inches(0.92), Inches(2.5), Inches(11.7), Inches(2.0),
   [[("Прогнозирование временных рядов за счёт", 30, WHITE, True, False)],
    [("обучаемой декомпозиции и нейросетевых механизмов", 30, WHITE, True, False)],
    [("учёта долгосрочных зависимостей", 30, WHITE, True, False)]], ls=1.06)
tb(s, Inches(0.95), Inches(4.95), Inches(11.4), Inches(0.42),
   [[("Выполнил: ", 15, LBLUE, False, False),
     ("студент группы КМБО-03-22 — Лазарев А.К.", 15, WHITE, True, False)]])
tb(s, Inches(0.95), Inches(5.42), Inches(11.4), Inches(0.42),
   [[("Научный руководитель: ", 15, LBLUE, False, False),
     ("к.ф.-м.н., доцент кафедры ВМ — Петрусевич Д.А.", 15, WHITE, True, False)]])
tb(s, Inches(0.95), Inches(6.55), Inches(5.0), Inches(0.4),
   [[("Москва · 2026", 13, LBLUE, False, False)]])

# ===== 2 — Актуальность и цель =============================================
s = new()
header(s, "Актуальность и цель исследования", "Раздел 1. Постановка задачи", 2)
panel(s, Inches(0.7), Inches(1.78), Inches(6.05), Inches(4.35), "Проблематика", [
    ("Прикладная значимость. ", "Краткосрочный прогноз — основа планирования спроса, трафика, нагрузки и ресурсов."),
    ("Нет универсальной модели. ", "Классика устойчива на коротких рядах, нейросети — на длинных и сложных."),
    ("Открытый вопрос. ", "Оправдывают ли обучаемая декомпозиция и механизмы долгосрочных зависимостей свою сложность?"),
], size=15, gap=12)
rect(s, Inches(7.0), Inches(1.78), Inches(5.63), Inches(4.35), NAVY)
rect(s, Inches(7.0), Inches(1.78), Pt(4), Inches(4.35), ACCENT)
tb(s, Inches(7.3), Inches(2.0), Inches(5.1), Inches(0.4),
   [[("ЦЕЛЬ РАБОТЫ", 14, ACCENT, True, False)]])
tb(s, Inches(7.3), Inches(2.55), Inches(5.05), Inches(3.4),
   [[("Установить условия, при которых обучаемая декомпозиция и нейросетевые "
      "механизмы учёта долгосрочных зависимостей дают практическое "
      "преимущество над классическими моделями — и определить, где более "
      "простые методы остаются предпочтительными.", 16, WHITE, False, False)]],
   ls=1.22)

# ===== 3 — Постановка задачи и метрики =====================================
s = new()
header(s, "Постановка задачи и метрики качества",
       "Раздел 2. Что прогнозируем и как измеряем", 3)
panel(s, Inches(0.7), Inches(1.78), Inches(5.0), Inches(4.35),
      "Задача прогноза", [
    ("Дано: ", "ряд наблюдений y₁, y₂, …, y_T."),
    ("Найти: ", "значения ŷ_{T+1}, …, ŷ_{T+H} на горизонте H."),
    ("Условия. ", "Одномерный краткосрочный прогноз, H ≪ T, минимум допущений о структуре."),
    ("Оценка. ", "Out-of-sample: модель не видит тестовый отрезок при обучении."),
], size=14.5, gap=12)
metrics = [("RMSE", "rmse", "абсолютная ошибка, тот же масштаб"),
           ("MASE", "mase", "относительно Seasonal Naive, безразмерна"),
           ("sMAPE", "smape", "симметричная, %, стандарт протоколов M3/M4")]
yk = Inches(1.78)
for nm, key, cap in metrics:
    rect(s, Inches(5.95), yk, Inches(6.68), Inches(1.32), PANEL)
    rect(s, Inches(5.95), yk, Pt(4), Inches(1.32), ACCENT)
    tb(s, Inches(6.2), yk + Inches(0.12), Inches(2.0), Inches(0.4),
       [[(nm, 16, NAVY, True, False)]])
    formula(s, key, Inches(6.2), yk + Inches(0.52), Inches(0.46))
    tb(s, Inches(6.2), yk + Inches(1.02), Inches(6.2), Inches(0.3),
       [[(cap, 12, MUTE, False, True)]])
    yk += Inches(1.5)

# ===== 4 — Классические модели (formula cards) =============================
s = new()
header(s, "Классические модели сравнения",
       "Раздел 2. Эталон: интерпретируемы и устойчивы на коротких рядах", 4)
cards = [("ARIMA", "arima", "AR + интегрирование + MA; порядки (p,d,q) по ACF/PACF.",
          "устойчивость на коротких рядах"),
         ("ETS", "ets", "Экспоненциальное сглаживание уровня, тренда и сезона.",
          "плавная динамика и сезонность"),
         ("Prophet", "prophet", "Аддитивная модель: тренд + сезон Фурье + календарь.",
          "гибкая настройка сезонных компонент")]
xc = Inches(0.7); cw = Inches(3.87); gap = Inches(0.16)
for i, (nm, key, desc, role) in enumerate(cards):
    x = xc + i * (cw + gap)
    rect(s, x, Inches(1.78), cw, Inches(3.55), PANEL)
    rect(s, x, Inches(1.78), cw, Inches(0.56), NAVY if i < 2 else ACCENT)
    tb(s, x, Inches(1.88), cw, Inches(0.4), [[(nm, 16, WHITE, True, False)]],
       align=PP_ALIGN.CENTER)
    rect(s, x + Inches(0.2), Inches(2.55), cw - Inches(0.4), Inches(0.86), WHITE)
    fh = Inches(0.3) if key == "arima" else Inches(0.36)
    formula(s, key, x + Inches(0.34), Inches(2.84), fh)
    tb(s, x + Inches(0.24), Inches(3.62), cw - Inches(0.44), Inches(1.0),
       [[(desc, 13.5, INK, False, False)]], ls=1.16)
    rect(s, x + Inches(0.24), Inches(4.72), cw - Inches(0.48), Pt(1.2), RULE)
    tb(s, x + Inches(0.24), Inches(4.84), cw - Inches(0.44), Inches(0.46),
       [[("Сильная сторона: ", 12.5, ACC_DK, True, True), (role, 12.5, ACC_DK, False, True)]],
       ls=1.1)
panel(s, Inches(0.7), Inches(5.5), Inches(11.93), Inches(0.78),
      "Четвёртая модель", [
    ("Seasonal Naive — ", "прогноз = значение предыдущего сезонного периода; нижняя граница качества и база метрики MASE."),
], size=13.5)

# ===== 5 — Обучаемая vs фиксированная декомпозиция (СМЫСЛ) ==================
s = new()
header(s, "Обучаемая и фиксированная декомпозиция",
       "Раздел 2. В чём смысл различия и почему это главный вопрос работы", 5)
rect(s, Inches(0.7), Inches(1.78), Inches(5.85), Inches(3.0), PANEL)
rect(s, Inches(0.7), Inches(1.78), Pt(4), Inches(3.0), NAVY2)
tb(s, Inches(0.98), Inches(1.94), Inches(5.4), Inches(0.4),
   [[("Фиксированная декомпозиция", 15.5, NAVY, True, False)]])
tb(s, Inches(0.98), Inches(2.42), Inches(5.3), Inches(2.3),
   [[("Правило разложения задано человеком ", 14, INK, False, False),
     ("заранее", 14, INK, True, False),
     (": период сезонности, скользящее среднее, формулы Холта–Уинтерса. "
      "Прозрачно и устойчиво, но не подстраивается под данные.\n", 14, INK, False, False)],
    [("Примеры: ", 13.5, MUTE, True, False),
     ("ETS, DLinear, Helformer, классическая STL.", 13.5, MUTE, False, True)]], ls=1.16)
rect(s, Inches(6.78), Inches(1.78), Inches(5.85), Inches(3.0), PANEL)
rect(s, Inches(6.78), Inches(1.78), Pt(4), Inches(3.0), ACCENT)
tb(s, Inches(7.06), Inches(1.94), Inches(5.4), Inches(0.4),
   [[("Обучаемая декомпозиция", 15.5, ACC_DK, True, False)]])
tb(s, Inches(7.06), Inches(2.42), Inches(5.35), Inches(2.3),
   [[("Компоненты выделяет ", 14, INK, False, False),
     ("сама сеть", 14, INK, True, False),
     (" в процессе обучения: авто-корреляция, частотные фильтры, 2D-свёртки. "
      "Гибко и адаптивно, но сложнее и рискует переобучиться.\n", 14, INK, False, False)],
    [("Примеры: ", 13.5, MUTE, True, False),
     ("Autoformer, FEDformer, TimesNet, N-BEATS.", 13.5, MUTE, False, True)]], ls=1.16)
rect(s, Inches(0.7), Inches(4.95), Inches(11.93), Inches(1.33), NAVY)
rect(s, Inches(0.7), Inches(4.95), Pt(4), Inches(1.33), ACCENT)
tb(s, Inches(1.0), Inches(5.1), Inches(4.0), Inches(0.4),
   [[("ЗАЧЕМ ЭТО РАЗЛИЧИЕ", 13, ACCENT, True, False)]])
tb(s, Inches(1.0), Inches(5.5), Inches(11.3), Inches(0.8),
   [[("Обучаемость даёт гибкость, но повышает требования к объёму данных и "
      "вычислениям. Центральный вопрос работы — ", 15, WHITE, False, False),
     ("когда эта дополнительная свобода окупается, а когда фиксированное "
      "правило надёжнее.", 15, WHITE, True, False)]], ls=1.16)

# ===== 6 — Архитектуры I: DLinear + N-BEATS ================================
s = new()
header(s, "Архитектуры нейросетей · I", "Раздел 2. DLinear и N-BEATS — разложение прогноза", 6)
# DLinear
tb(s, Inches(0.7), Inches(1.66), Inches(11.9), Inches(0.34),
   [[("DLinear — ", 15, NAVY, True, False),
     ("линейная модель с явной декомпозицией", 14, MUTE, False, True)]])
pipeline(s, Inches(0.7), Inches(2.06), Inches(8.4),
         ["Вход\n(окно L)", "Декомпозиция\n(скольз. среднее)",
          "Тренд / Остаток", "Линейные\nпроекции", "Прогноз H"],
         accent_idx=[1])
formula(s, "dlinear", Inches(0.7), Inches(3.18), Inches(0.42))
tb(s, Inches(9.4), Inches(2.06), Inches(3.23), Inches(1.4),
   [[("Смысл: ", 13, ACC_DK, True, False),
     ("разделяет ряд на тренд и колебания и предсказывает их двумя простыми "
      "линейными слоями — сильный и дешёвый бейзлайн.", 13, INK, False, False)]], ls=1.12)
rect(s, Inches(0.7), Inches(3.86), Inches(11.93), Pt(1), RULE)
# N-BEATS
tb(s, Inches(0.7), Inches(4.0), Inches(11.9), Inches(0.34),
   [[("N-BEATS — ", 15, NAVY, True, False),
     ("обучаемое базисное разложение с двойным остатком", 14, MUTE, False, True)]])
pipeline(s, Inches(0.7), Inches(4.4), Inches(8.4),
         ["Вход", "Блок: FC ×4", "Базисное\nразложение",
          "Двойной\nостаток", "Σ прогнозов ŷ"], accent_idx=[2])
formula(s, "nbeats", Inches(0.7), Inches(5.52), Inches(0.42))
tb(s, Inches(9.4), Inches(4.4), Inches(3.23), Inches(1.5),
   [[("Смысл: ", 13, ACC_DK, True, False),
     ("стек блоков, каждый «объясняет» часть истории (backcast) и уточняет "
      "прогноз (forecast); итог — сумма вкладов.", 13, INK, False, False)]], ls=1.12)
conclusion_band(s, "Обе модели раскладывают прогноз на простые составляющие — "
                   "но DLinear делает это фиксированно, а N-BEATS обучает базис.")

# ===== 7 — Архитектуры II: TimesNet + Autoformer ===========================
s = new()
header(s, "Архитектуры нейросетей · II",
       "Раздел 2. TimesNet и Autoformer — периодичность и дальние зависимости", 7)
tb(s, Inches(0.7), Inches(1.66), Inches(11.9), Inches(0.34),
   [[("TimesNet — ", 15, NAVY, True, False),
     ("1D-ряд → 2D-представления по доминирующим периодам", 14, MUTE, False, True)]])
pipeline(s, Inches(0.7), Inches(2.06), Inches(8.4),
         ["Вход (1D)", "FFT:\nпериоды", "Reshape\n1D → 2D",
          "2D-свёртки", "Softmax-\nагрегация"], accent_idx=[2])
formula(s, "timesnet", Inches(0.7), Inches(3.18), Inches(0.42))
tb(s, Inches(9.4), Inches(2.06), Inches(3.23), Inches(1.4),
   [[("Смысл: ", 13, ACC_DK, True, False),
     ("находит периоды через Фурье и сворачивает ряд в 2D-«картинки» — "
      "видит и внутри-, и меж-периодные изменения.", 13, INK, False, False)]], ls=1.12)
rect(s, Inches(0.7), Inches(3.86), Inches(11.93), Pt(1), RULE)
tb(s, Inches(0.7), Inches(4.0), Inches(11.9), Inches(0.34),
   [[("Autoformer — ", 15, NAVY, True, False),
     ("обучаемая декомпозиция + авто-корреляция (ядро темы работы)", 14, ACC_DK, False, True)]])
pipeline(s, Inches(0.7), Inches(4.4), Inches(8.4),
         ["Вход", "Series Decomp\n(тренд+сезон)", "Auto-\nCorrelation",
          "Encoder–\nDecoder", "Прогноз"], accent_idx=[1, 2])
formula(s, "autoformer", Inches(0.7), Inches(5.52), Inches(0.42))
tb(s, Inches(9.4), Inches(4.4), Inches(3.23), Inches(1.5),
   [[("Смысл: ", 13, ACC_DK, True, False),
     ("раскладывает ряд внутри сети и заменяет внимание авто-корреляцией — "
      "ищет похожие участки на больших лагах.", 13, INK, False, False)]], ls=1.12)
conclusion_band(s, "Autoformer объединяет обе идеи темы: обучаемую декомпозицию "
                   "и механизм учёта долгосрочных зависимостей.")

# ===== 8 — Helformer (флагман) =============================================
s = new()
header(s, "Helformer — основная гибридная модель",
       "Раздел 2. Декомпозиция Холта–Уинтерса + механизм внимания", 8)
pipeline(s, Inches(0.7), Inches(2.0), Inches(11.93),
         ["Вход", "Holt-Winters\nдекомпозиция", "Нормализация",
          "Transformer\n+ LSTM-блок", "Реконструкция\n→ Прогноз"],
         accent_idx=[1, 3], bh=Inches(1.0), fs=12)
rect(s, Inches(0.7), Inches(3.4), Inches(7.4), Inches(1.0), WHITE)
formula(s, "helformer", Inches(0.84), Inches(3.62), Inches(0.5))
panel(s, Inches(0.7), Inches(4.62), Inches(5.85), Inches(1.66),
      "Что делает модель", [
    ("Холт–Уинтерс ", "снимает тренд и сезон — фиксированная база."),
    ("Внимание + LSTM ", "моделируют остаток и дальние зависимости."),
], size=13.5, gap=8)
rect(s, Inches(6.78), Inches(4.62), Inches(5.85), Inches(1.66), NAVY)
rect(s, Inches(6.78), Inches(4.62), Pt(4), Inches(1.66), ACCENT)
tb(s, Inches(7.06), Inches(4.78), Inches(5.3), Inches(0.4),
   [[("ПОЧЕМУ В НАЗВАНИИ РАБОТЫ", 12.5, ACCENT, True, False)]])
tb(s, Inches(7.06), Inches(5.22), Inches(5.35), Inches(1.0),
   [[("Helformer сочетает обе исследуемые идеи в одной модели — это прямой "
      "тест гипотезы работы на флагманской архитектуре.", 14.5, WHITE, False, False)]],
   ls=1.18)

# ===== 9 — Экспериментальная база ==========================================
s = new()
header(s, "Экспериментальная база и протокол",
       "Раздел 3. Данные, режимы обучения, тип декомпозиции", 9)
panel(s, Inches(0.7), Inches(1.78), Inches(3.8), Inches(4.35), "Данные", [
    ("M3 Competition. ", "Полный набор — 2829 рядов: годовые, квартальные, месячные."),
    ("M4 Competition. ", "3000 рядов: квартальные, месячные, ежедневные."),
    ("Итого ", "6 частотных категорий, 5829 рядов."),
    ("Метрики. ", "sMAPE, MASE, RMSE."),
], size=14, gap=13)
panel(s, Inches(4.66), Inches(1.78), Inches(3.8), Inches(4.35),
      "Три режима прогноза", [
    ("Прямой. ", "Весь горизонт за один проход модели."),
    ("Итеративный. ", "Пошагово, с подстановкой своих прогнозов; ошибка копится."),
    ("С предобучением. ", "Обучение на корпусе рядов + дообучение на целевом."),
], size=14, gap=13)
panel(s, Inches(8.62), Inches(1.78), Inches(4.0), Inches(4.35),
      "Тип декомпозиции", [
    ("Фиксированная. ", "ETS, DLinear, Helformer."),
    ("Обучаемая. ", "Autoformer, FEDformer, TimesNet."),
    ("Без декомпозиции. ", "Auto-ARIMA, PatchTST."),
], size=14, gap=13)
conclusion_band(s, "Анализ ведётся по трём осям — длина ряда, режим обучения и "
                   "тип декомпозиции.")

# ===== 10 — Прямое прогнозирование: ранг + ЧТО ТАКОЕ РАНГ ===================
s = new()
header(s, "Прямое прогнозирование: ранг моделей",
       "Раздел 3. Кто точнее в режиме обучения на одном ряде", 10)
img_fit(s, FIG / "heatmap_rank.png", Inches(0.7), Inches(1.7), Inches(7.5), Inches(4.4))
panel(s, Inches(8.4), Inches(1.7), Inches(4.23), Inches(2.4),
      "Что такое ранг", [
    "На каждом ряде модели сортируются по sMAPE: 1-е место — лучшая.",
    "Ранги усредняются по всем рядам категории → средний ранг модели.",
    ("Цвет: ", "зелёный — точнее, красный — хуже."),
], size=13, gap=7)
legend_text(s, Inches(8.4), Inches(4.22), Inches(4.23), Inches(1.88),
            "Как читать карту", [
    (GREEN, "Верхние строки — ", "классика (Auto-ARIMA, ETS, S. Naive) в топе."),
    (C_TIMES, "TimesNet ", "— единственная нейросеть рядом с лидерами."),
])
conclusion_band(s, "При обучении на одном ряде классика устойчиво лидирует во "
                   "всех категориях; среди нейросетей выделяется только TimesNet.")

# ===== 11 — Длина обучающей выборки =========================================
s = new()
header(s, "Длина обучающей выборки как фактор",
       "Раздел 3. Пороговая зависимость качества от длины ряда", 11)
img_fit(s, FIG / "bar_m4_daily_full.png", Inches(0.7), Inches(1.7), Inches(7.5), Inches(4.4))
panel(s, Inches(8.4), Inches(1.7), Inches(4.23), Inches(2.3),
      "Пороговый эффект (≈ 50–100 набл.)", [
    ("Короткие ряды. ", "Уверенно лидирует классика."),
    ("Длинные M4 daily. ", "Нейросети догоняют классику."),
], size=13, gap=8)
legend_text(s, Inches(8.4), Inches(4.12), Inches(4.23), Inches(1.98),
            "Значения sMAPE на диаграмме", [
    (C_ARIMA, "Auto-ARIMA 3.22 · ETS 3.46 ", "— классика (лидеры)."),
    (C_TIMES, "TimesNet 3.62 ", "— уже на уровне классики."),
    (C_PATCH, "PatchTST, N-BEATS, Helformer ", "≈ 4.5 — близкая группа."),
])
conclusion_band(s, "Есть пороговая длина выборки: ниже неё лидирует классика, "
                   "выше — нейросети выходят на её уровень.")

# ===== 12 — Предобучение: эффект ===========================================
s = new()
header(s, "Предобучение: эффект переноса знаний",
       "Раздел 3. Относительное улучшение sMAPE по моделям и частотам", 12)
img_fit(s, FIG / "pretrain_effect_all.png", Inches(0.7), Inches(1.7), Inches(7.7), Inches(4.35))
rect(s, Inches(8.6), Inches(1.7), Inches(4.03), Inches(4.35), NAVY)
rect(s, Inches(8.6), Inches(1.7), Pt(4), Inches(4.35), ACCENT)
tb(s, Inches(8.85), Inches(1.9), Inches(3.6), Inches(0.4),
   [[("СРЕДНИЙ ВЫИГРЫШ sMAPE", 12.5, ACCENT, True, False)]])
for i, (v, nm) in enumerate([("+31.5 %", "DLinear"), ("+25.5 %", "Autoformer"),
                              ("+18.2 %", "Helformer")]):
    yy = Inches(2.4) + i * Inches(1.16)
    tb(s, Inches(8.85), yy, Inches(3.6), Inches(0.6), [[(v, 30, WHITE, True, False)]])
    tb(s, Inches(8.85), yy + Inches(0.56), Inches(3.6), Inches(0.34),
       [[(nm, 13, LBLUE, False, False)]])
conclusion_band(s, "Предобучение на корпусе рядов выводит ранее слабые "
                   "нейросетевые модели на конкурентоспособный уровень.")

# ===== 13 — Предобучение: эффект на прогнозе (FIXED chart) ==================
s = new()
header(s, "Предобучение: эффект на прогнозе ряда",
       "Раздел 3. Одна модель: обучение с нуля против предобучения", 13)
img_fit(s, FIG / "pretrain_forecast_autoformer_D2111.png",
        Inches(0.7), Inches(1.7), Inches(7.7), Inches(4.4))
legend_text(s, Inches(8.55), Inches(1.7), Inches(4.08), Inches(2.95),
            "Линии на графике", [
    (C_GREY, "Серая ", "— обучающая история."),
    (C_BLACK, "Чёрная ", "— фактические значения."),
    (C_AUTO, "Оранжевый пунктир ", "— Autoformer с нуля, sMAPE 3.8 %."),
    (C_AUTO, "Оранжевая сплошная ", "— Autoformer предобучен, sMAPE 1.0 %."),
])
panel(s, Inches(8.55), Inches(4.78), Inches(4.08), Inches(1.32),
      "Обратите внимание", [
    "Прогноз начинается от последней точки обучения — без разрыва.",
], size=13, gap=6)
conclusion_band(s, "Предобучение существенно повышает точность одной и той же "
                   "модели — прогноз плотно следует факту.")

# ===== 14 — Прогноз крупно (dedicated forecast slide) ======================
s = new()
header(s, "Прогноз крупно: предобученные нейросети",
       "Раздел 3. Ежедневный ряд D2111 — все модели на одном горизонте", 14)
img_fit(s, FIG / "forecast_large_D2111.png", Inches(0.7), Inches(1.66), Inches(8.4), Inches(4.5))
legend_text(s, Inches(9.25), Inches(1.66), Inches(3.38), Inches(4.5),
            "Модели и sMAPE", [
    (C_BLACK, "Факт ", "— чёрная жирная линия."),
    (C_PATCH, "PatchTST ", "— 0.3 %"),
    (C_FED, "FEDformer ", "— 0.6 %"),
    (C_HELF, "Helformer ", "— 0.7 %"),
    (C_AUTO, "Autoformer ", "— 1.1 %"),
    (C_ARIMA, "Auto-ARIMA ", "— 1.3 % (пунктир)"),
    (C_ETS, "ETS ", "— 1.3 % (пунктир)"),
])
conclusion_band(s, "Предобученные нейросети (сплошные) точнее классических "
                   "ARIMA и ETS (пунктир) и плотно повторяют факт.")

# ===== 15 — Сводная таблица ================================================
s = new()
header(s, "Сводное сравнение: sMAPE по категориям",
       "Раздел 3. Предобученные нейросети и классические модели", 15)
rows = [
    ("Модель", ["M3-Y", "M3-Q", "M3-M", "M4-Q", "M4-M", "M4-D", "Ср."], "head"),
    ("Классические модели (per-series)", [""] * 7, "group"),
    ("Auto-ARIMA",     ["17.9", "10.9", "18.1", "10.2", "14.7", "3.22", "12.5"], "classic"),
    ("ETS",            ["19.1", "10.8", "17.4", "9.7", "15.1", "3.46", "12.6"], "classic"),
    ("Seasonal Naive", ["17.9", "11.1", "17.2", "12.2", "15.9", "4.05", "13.0"], "classic"),
    ("Нейросетевые модели (с предобучением)", [""] * 7, "group"),
    ("DLinear",   ["23.7", "12.7", "14.0", "15.4", "17.2", "2.44", "14.2"], "neural"),
    ("TimesNet",  ["25.1", "12.4", "13.7", "22.5", "17.2", "2.46", "15.6"], "neural"),
    ("PatchTST",  ["23.1", "15.1", "15.2", "19.6", "17.5", "2.36", "15.5"], "neural"),
    ("Autoformer", ["23.4", "13.8", "15.1", "20.1", "16.3", "2.81", "15.3"], "neural"),
    ("FEDformer", ["22.8", "15.8", "16.3", "19.6", "15.6", "4.16", "15.7"], "neural"),
    ("N-BEATS",   ["23.8", "14.4", "17.3", "21.5", "18.3", "3.20", "16.4"], "neural"),
    ("Helformer", ["29.8", "21.4", "23.0", "20.5", "46.3", "2.87", "24.0"], "neural"),
]
gt = s.shapes.add_table(len(rows), 8, Inches(0.7), Inches(1.74),
                        Inches(11.93), Inches(0.3)).table
gt.first_row = False; gt.horz_banding = False
gt.columns[0].width = Inches(3.5)
for j in range(1, 8):
    gt.columns[j].width = Inches(1.205)
for i, (label, vals, kind) in enumerate(rows):
    for j, txt in enumerate([label] + list(vals)):
        c = gt.cell(i, j)
        c.margin_left = Pt(5); c.margin_right = Pt(4)
        c.margin_top = Pt(2); c.margin_bottom = Pt(2)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = p.add_run(); r.text = txt; r.font.name = FONT
        if kind == "head":
            c.fill.solid(); c.fill.fore_color.rgb = NAVY
            r.font.color.rgb = WHITE; r.font.bold = True; r.font.size = Pt(13)
        elif kind == "group":
            if j == 0:
                c.merge(gt.cell(i, 7))
            c.fill.solid(); c.fill.fore_color.rgb = NAVY2
            r.font.color.rgb = WHITE; r.font.bold = True
            r.font.italic = True; r.font.size = Pt(12.5)
        else:
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if i % 2 else PANEL2
            r.font.size = Pt(13)
            r.font.color.rgb = MUTE if kind == "classic" else INK
            if j == 0:
                r.font.bold = True
            if kind == "neural" and j in (3, 6):
                c.fill.fore_color.rgb = RGBColor(0xFC, 0xE9, 0xE2)
                r.font.color.rgb = ACC_DK; r.font.bold = True
conclusion_band(s, "На M3 monthly и M4 daily (выделено) предобученные нейросети "
                   "превосходят лучшую классику; на коротких рядах классика "
                   "сохраняет преимущество.", y=Inches(6.46))

# ===== 16 — Анализ отдельных рядов =========================================
s = new()
header(s, "Анализ отдельных временных рядов",
       "Раздел 3. Массовый характер преимущества нейросетей", 16)
img_fit(s, FIG / "forecast_cherry_D2052.png", Inches(0.7), Inches(1.66), Inches(7.7), Inches(4.5))
legend_text(s, Inches(8.55), Inches(1.66), Inches(4.08), Inches(2.7),
            "Линии на графике", [
    (C_BLACK, "Факт ", "— чёрная линия."),
    (C_NBE, "N-BEATS ", "— 3.1 %"),
    (C_PATCH, "PatchTST ", "— 4.5 %"),
    (C_DLIN, "DLinear ", "— 6.0 %"),
    (C_ARIMA, "Auto-ARIMA ", "— 11.2 % (пунктир)"),
])
rect(s, Inches(8.55), Inches(4.5), Inches(4.08), Inches(1.6), NAVY)
rect(s, Inches(8.55), Inches(4.5), Pt(4), Inches(1.6), ACCENT)
tb(s, Inches(8.8), Inches(4.64), Inches(1.6), Inches(0.9),
   [[("517", 38, WHITE, True, False)]])
tb(s, Inches(10.35), Inches(4.66), Inches(2.1), Inches(0.9),
   [[("из 1000 ежедневных рядов M4 — нейросеть точнее классики",
      11.5, LBLUE, False, False)]], ls=1.05)
tb(s, Inches(8.8), Inches(5.46), Inches(1.6), Inches(0.6),
   [[("295", 38, WHITE, True, False)]])
tb(s, Inches(10.35), Inches(5.5), Inches(2.1), Inches(0.6),
   [[("рядов — побеждают 3+ нейросетевые модели", 11.5, LBLUE, False, False)]], ls=1.05)
conclusion_band(s, "Преимущество нейросетей на длинных рядах носит массовый, а "
                   "не единичный характер.")

# ===== 17 — Итеративное: проблемы ==========================================
s = new()
header(s, "Итеративное прогнозирование",
       "Раздел 3. Устойчивость моделей к накоплению ошибки", 17)
panel(s, Inches(0.7), Inches(1.78), Inches(7.3), Inches(4.35),
      "Деградация при переходе от прямого режима к итеративному", [
    ("Обучаемая декомпозиция уязвима. ", "FEDformer и TimesNet теряют ~5 п.п. sMAPE — компоненты искажаются на собственных прогнозах."),
    ("PatchTST устойчив. ", "Наименьшая деградация: работает с патчами без явного разделения на компоненты."),
    ("Классика стабильна. ", "ETS и Auto-ARIMA почти нечувствительны — рекуррентность заложена в модель."),
    ("Закономерность. ", "Чем сложнее обучаемое разложение, тем сильнее копится ошибка."),
], size=14.5, gap=13)
rect(s, Inches(8.25), Inches(1.78), Inches(4.38), Inches(4.35), NAVY)
rect(s, Inches(8.25), Inches(1.78), Pt(4), Inches(4.35), ACCENT)
tb(s, Inches(8.5), Inches(2.02), Inches(3.9), Inches(0.4),
   [[("СЛЕДСТВИЕ", 13, ACCENT, True, False)]])
tb(s, Inches(8.5), Inches(2.5), Inches(3.95), Inches(2.5),
   [[("Сложная обучаемая декомпозиция повышает риск накопления ошибки при "
      "многошаговом прогнозе. Для итеративных задач простые и классические "
      "модели предпочтительнее.", 15, WHITE, False, False)]], ls=1.22)
rect(s, Inches(8.5), Inches(4.95), Inches(3.7), Pt(1), NAVY2)
tb(s, Inches(8.5), Inches(5.12), Inches(3.95), Inches(1.0),
   [[("Рекомендация: ", 14, ACCENT, True, False),
     ("выбор модели учитывает не только точность, но и режим её применения.",
      14, LBLUE, False, False)]], ls=1.18)
conclusion_band(s, "Тип декомпозиции должен соответствовать режиму прогнозирования.")

# ===== 18 — Итеративное: масштаб деградации ================================
s = new()
header(s, "Итеративное прогнозирование: масштаб деградации",
       "Раздел 3. Прямой vs итеративный sMAPE по моделям (M3)", 18)
img_fit(s, FIG / "rollout_degradation_bars.png", Inches(0.7), Inches(1.7), Inches(7.7), Inches(4.4))
legend_text(s, Inches(8.55), Inches(1.7), Inches(4.08), Inches(2.5),
            "Столбцы на диаграмме", [
    (NAVY, "Синий ", "— прямой прогноз."),
    (ACCENT, "Оранжевый ", "— итеративный прогноз."),
])
panel(s, Inches(8.55), Inches(4.32), Inches(4.08), Inches(1.78),
      "Что видно", [
    ("Классика ", "(S. Naive, ARIMA, ETS) — деградации нет."),
    ("FEDformer, Autoformer, TimesNet ", "теряют 4–6 п.п. sMAPE."),
], size=13, gap=8)
conclusion_band(s, "Чем сложнее обучаемая декомпозиция, тем сильнее накопление "
                   "ошибки; классика к режиму нечувствительна.")

# ===== 19 — Выводы =========================================================
s = new()
header(s, "Выводы по работе", "Раздел 4. Основные результаты исследования", 19)
findings = [
    ("Длина обучающей выборки", "Определяющий фактор. Ниже ≈ 50–100 наблюдений — ETS и Auto-ARIMA; выше — нейросети конкурентоспособны."),
    ("TimesNet — лучшая нейросеть", "В режиме обучения на одном ряде: обучаемая 2D-декомпозиция эффективна при достаточном объёме данных."),
    ("Предобучение", "Повышает качество нейросетей на 18–32 % и выводит их на уровень и выше классики на M3 monthly и M4 daily."),
    ("Тип декомпозиции и режим", "Фиксированная устойчивее, обучаемая адаптивнее, но хрупка при итеративном прогнозе."),
    ("Общий вывод", "Эффективность нейросетевых механизмов не абсолютна — её определяют длина ряда, режим обучения и объём данных."),
]
y = Inches(1.74); rh = Inches(1.0)
for i, (t, d) in enumerate(findings):
    rect(s, Inches(0.7), y, Inches(11.93), rh - Inches(0.06),
         PANEL if i % 2 == 0 else PANEL2)
    rect(s, Inches(0.7), y, Pt(4), rh - Inches(0.06), ACCENT)
    chip = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.95), y + Inches(0.21),
                              Inches(0.56), Inches(0.56))
    solid(chip, NAVY)
    tb(s, Inches(0.95), y + Inches(0.29), Inches(0.56), Inches(0.4),
       [[(str(i + 1), 19, WHITE, True, False)]], align=PP_ALIGN.CENTER)
    tb(s, Inches(1.75), y + Inches(0.13), Inches(3.5), Inches(0.7),
       [[(t, 15.5, NAVY, True, False)]], anchor=MSO_ANCHOR.MIDDLE, ls=1.0)
    tb(s, Inches(5.35), y + Inches(0.1), Inches(7.1), Inches(0.82),
       [[(d, 13.5, INK, False, False)]], anchor=MSO_ANCHOR.MIDDLE, ls=1.1)
    y += rh

# ===== 20 — Когда строить сложные модели (рекомендации) ====================
s = new()
header(s, "Когда строить сложные модели",
       "Раздел 4. Практические рекомендации по выбору модели", 20)
recs = [
    ("Сценарий", "Что выбрать", "Почему", "head"),
    ("Короткие ряды (годовые, месячные)", "Классика: ETS / Auto-ARIMA",
     "Нейросети не успевают обучиться на малой истории", "row"),
    ("Длинные ряды (ежедневные, недельные)", "Нейросети с предобучением",
     "Длина раскрывает обучаемую декомпозицию и дальние зависимости", "hot"),
    ("Есть корпус похожих рядов", "Предобучение + дообучение",
     "Перенос знаний даёт +18–32 % к качеству", "hot"),
    ("Итеративный (rollout) прогноз", "Простые / классические модели",
     "Сложная обучаемая декомпозиция копит ошибку на горизонте", "row"),
    ("Гладкий тренд без сезонности", "ETS",
     "Сглаживание устойчиво к шуму", "row"),
]
y = Inches(1.78)
widths = [Inches(4.2), Inches(3.6), Inches(4.13)]
for ri, (a, b, c, kind) in enumerate(recs):
    rh = Inches(0.5) if kind == "head" else Inches(0.74)
    x = Inches(0.7)
    if kind == "head":
        fill = NAVY
    elif kind == "hot":
        fill = RGBColor(0xFC, 0xE9, 0xE2)
    else:
        fill = PANEL if ri % 2 else PANEL2
    for ci, (txt, wd) in enumerate(zip((a, b, c), widths)):
        rect(s, x, y, wd, rh, fill)
        if kind == "head":
            tcol = WHITE; bold = True
        elif ci == 1:
            tcol = ACC_DK if kind == "hot" else NAVY; bold = True
        else:
            tcol = INK; bold = False
        tb(s, x + Inches(0.18), y, wd - Inches(0.32), rh,
           [[(txt, 13.5 if kind == "head" else 13, tcol, bold, False)]],
           anchor=MSO_ANCHOR.MIDDLE, ls=1.04)
        x += wd
    y += rh + Inches(0.02)
conclusion_band(s, "Сложная модель оправдана при длинных рядах, доступном "
                   "корпусе для предобучения и прямом (не итеративном) режиме.")

# ===== 21 — Список литературы ==============================================
s = new()
header(s, "Список литературы (избранное)",
       "Раздел 5. Полный список — 51 источник в ВКР", 21)
refs_left = [
    "Hyndman R.J., Athanasopoulos G. Forecasting: principles and practice. — OTexts, 2021.",
    "Makridakis S. et al. The M4 Competition // Int. J. Forecasting. — 2020. — Vol. 36(1).",
    "Makridakis S., Hibon M. The M3-Competition // Int. J. Forecasting. — 2000. — Vol. 16(4).",
    "Cleveland R.B. et al. STL: A Seasonal-Trend Decomposition Procedure. — 1990.",
    "Hyndman R.J. et al. A state space framework for automatic forecasting. — 2002.",
    "Taylor S.J., Letham B. Forecasting at scale // The American Statistician. — 2018.",
    "Zeng A. et al. Are Transformers Effective for Time Series Forecasting? (DLinear) // AAAI. — 2023.",
]
refs_right = [
    "Oreshkin B.N. et al. N-BEATS. — arXiv:1905.10437, 2020.",
    "Wu H. et al. TimesNet: Temporal 2D-Variation Modeling. — ICLR, 2023.",
    "Wu H. et al. Autoformer: Decomposition Transformers with Auto-Correlation. — NeurIPS, 2021.",
    "Zhou T. et al. FEDformer // ICML. — 2022.",
    "Nie Y. et al. A Time Series is Worth 64 Words (PatchTST). — ICLR, 2023.",
    "Kehinde T.O. et al. Helformer // J. Big Data. — 2025.",
    "Vaswani A. et al. Attention Is All You Need. — NeurIPS, 2017.",
]


def ref_column(slide, x, items, start):
    runs = []
    for k, txt in enumerate(items):
        runs.append([(f"{start + k}.  ", 11.5, ACC_DK, True, False),
                     (txt, 11.5, INK, False, False)])
    tb(slide, x, Inches(1.85), Inches(5.9), Inches(4.9), runs, ls=1.12, sa=12)


ref_column(s, Inches(0.7), refs_left, 1)
rect(s, Inches(6.66), Inches(1.95), Pt(0.75), Inches(4.5), RULE)
ref_column(s, Inches(6.9), refs_right, 8)
conclusion_band(s, "Полный пронумерованный список из 51 источника по ГОСТ "
                   "Р 7.0.5-2008 приведён в тексте ВКР.", y=Inches(6.45))

# ===== 22 — Заключение =====================================================
s = new()
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, 0, Inches(0.22), SH, ACCENT)
rect(s, Inches(0.95), Inches(4.05), Inches(4.4), Pt(2.5), ACCENT)
tb(s, Inches(0.95), Inches(2.55), Inches(11.4), Inches(1.0),
   [[("Спасибо за внимание", 42, WHITE, True, False)]])
tb(s, Inches(0.95), Inches(4.35), Inches(11.4), Inches(0.9),
   [[("Прогнозирование временных рядов за счёт обучаемой декомпозиции и "
      "нейросетевых механизмов учёта долгосрочных зависимостей",
      15, LBLUE, False, True)]], ls=1.2)
tb(s, Inches(0.95), Inches(5.55), Inches(11.4), Inches(0.4),
   [[("Лазарев А.К. · КМБО-03-22 · Научный руководитель — Петрусевич Д.А.",
      14, WHITE, False, False)]])
tb(s, Inches(0.95), Inches(6.05), Inches(11.4), Inches(0.4),
   [[("Готов ответить на вопросы комиссии", 13, LBLUE, False, True)]])

# ===== Д1 — Эконом: себестоимость ==========================================
s = new()
header(s, "Себестоимость проекта",
       "Экономическая часть · калькуляция разработки сервиса «ГМ-ПВР»", "Доп. 1")
erows = [
    ("Статья затрат", "Затраты, ₽", "Доля", "head"),
    ("Материалы и расходники", "5 210", "0,6 %", "r"),
    ("Спец. оборудование (GPU, хранение)", "27 000", "3,0 %", "r"),
    ("Основная заработная плата", "240 830", "27,1 %", "hot"),
    ("Дополнительная ЗП (20 %)", "48 166", "5,4 %", "r"),
    ("Страховые взносы (30,2 %)", "87 276,8", "9,8 %", "r"),
    ("Накладные расходы (200 %)", "481 660", "54,1 %", "hot"),
    ("Итого", "890 142,8", "100 %", "tot"),
]
y = Inches(1.78); ws = [Inches(4.1), Inches(2.1), Inches(1.5)]
for a, b, c, kind in erows:
    rh = Inches(0.5) if kind == "head" else Inches(0.52)
    x = Inches(0.7)
    fill = {"head": NAVY, "hot": RGBColor(0xFC, 0xE9, 0xE2),
            "tot": NAVY2}.get(kind, PANEL2)
    for ci, (txt, wd) in enumerate(zip((a, b, c), ws)):
        rect(s, x, y, wd, rh, fill)
        tcol = WHITE if kind in ("head", "tot") else (ACC_DK if kind == "hot" and ci != 0 else INK)
        tb(s, x + Inches(0.16), y, wd - Inches(0.28), rh,
           [[(txt, 12.5, tcol, kind in ("head", "tot", "hot"), False)]],
           align=PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER,
           anchor=MSO_ANCHOR.MIDDLE)
        x += wd
    y += rh + Inches(0.015)
# keystats
for i, (num, lab) in enumerate([("890 142,8 ₽", "полная себестоимость (метод калькулирования)"),
                                 ("60 000 ₽", "оклад руководителя · 25 раб. дней"),
                                 ("50 000 ₽", "оклад разработчика · 90 раб. дней")]):
    yy = Inches(1.82) + i * Inches(1.45)
    rect(s, Inches(8.65), yy, Inches(3.98), Inches(1.3),
         NAVY if i == 0 else (ACCENT if i == 1 else NAVY2))
    tb(s, Inches(8.9), yy + Inches(0.16), Inches(3.6), Inches(0.6),
       [[(num, 28, WHITE, True, False)]])
    tb(s, Inches(8.9), yy + Inches(0.82), Inches(3.6), Inches(0.4),
       [[(lab, 11.5, LBLUE, False, True)]], ls=1.04)
conclusion_band(s, "Большая часть бюджета — накладные расходы (54,1 %) и "
                   "основная зарплата (27,1 %): главный ресурс — труд исполнителей.")

# ===== Д2 — Эконом: окупаемость ============================================
s = new()
header(s, "Окупаемость и эффективность проекта",
       "Экономическая часть · NPV, PI, срок окупаемости при ставке 22 %", "Доп. 2")
ks = [("890 142,8 ₽", "IC — инвестиции", NAVY),
      ("≈ 48,0 млн ₽", "NPV — чистая прив. стоимость", NAVY2),
      ("≈ 54,93", "PI — индекс рентабельности", ACCENT),
      ("≈ 2,4 мес.", "DPP — диск. окупаемость", NAVY2)]
xx = Inches(0.7)
for num, lab, col in ks:
    rect(s, xx, Inches(1.78), Inches(2.86), Inches(1.4), col)
    tb(s, xx + Inches(0.2), Inches(1.96), Inches(2.5), Inches(0.6),
       [[(num, 24, WHITE, True, False)]])
    tb(s, xx + Inches(0.2), Inches(2.62), Inches(2.5), Inches(0.5),
       [[(lab, 11.5, LBLUE, False, True)]], ls=1.04)
    xx += Inches(2.98)
# discounted cashflow table
drows = [("Период", "CF, тыс. ₽", "k = 1/(1+r)ⁱ", "PV, тыс. ₽", "head"),
         ("Год 0 (инвестиции)", "−890,1", "1,0000", "−890,1", "r"),
         ("Год 1", "5 345,9", "0,8197", "4 381,9", "r"),
         ("Год 2", "26 239,8", "0,6719", "17 629,5", "r"),
         ("Год 3", "48 818,7", "0,5507", "26 884,8", "r"),
         ("NPV", "", "", "48 006,1", "tot")]
y = Inches(3.5); ws = [Inches(2.5), Inches(1.6), Inches(1.7), Inches(1.6)]
for a, b, c, d, kind in drows:
    rh = Inches(0.46)
    x = Inches(0.7)
    fill = {"head": NAVY, "tot": NAVY2}.get(kind, PANEL2)
    for ci, (txt, wd) in enumerate(zip((a, b, c, d), ws)):
        rect(s, x, y, wd, rh, fill)
        tcol = WHITE if kind in ("head", "tot") else INK
        tb(s, x + Inches(0.14), y, wd - Inches(0.24), rh,
           [[(txt, 12, tcol, kind in ("head", "tot"), False)]],
           align=PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER,
           anchor=MSO_ANCHOR.MIDDLE)
        x += wd
    y += rh + Inches(0.015)
# conservative scenario panel
rect(s, Inches(8.2), Inches(3.5), Inches(4.43), Inches(2.78), PANEL)
rect(s, Inches(8.2), Inches(3.5), Pt(4), Inches(2.78), ACCENT)
tb(s, Inches(8.46), Inches(3.64), Inches(4.0), Inches(0.4),
   [[("Консервативный сценарий", 14.5, NAVY, True, False)]])
tb(s, Inches(8.46), Inches(4.06), Inches(4.0), Inches(0.4),
   [[("Прирост клиентской базы по 1 в месяц:", 12.5, INK, False, False)]])
rect(s, Inches(8.46), Inches(4.5), Inches(3.9), Inches(0.66), WHITE)
formula(s, "payback", Inches(8.62), Inches(4.66), Inches(0.36))
tb(s, Inches(8.46), Inches(5.3), Inches(4.0), Inches(0.9),
   [[("N = 4 → 850 000 ₽ — не покрывает;  N = 5 → 1 275 000 ₽ — покрывает.\n",
      12.5, INK, False, False),
     ("Полная окупаемость — к концу 5-го месяца продаж.", 12.5, ACC_DK, True, False)]],
   ls=1.1)

# ===== Д3 — Эконом: план продаж и эффект ===================================
s = new()
header(s, "План продаж и эффект для клиента",
       "Экономическая часть · 3-летний P&L и экономия потребителя «ГМ-ПВР»", "Доп. 3")
img_fit(s, FIG / "econ_pnl.png", Inches(0.7), Inches(1.7), Inches(7.4), Inches(4.4))
rect(s, Inches(8.3), Inches(1.7), Inches(4.33), Inches(2.4), PANEL)
rect(s, Inches(8.3), Inches(1.7), Pt(4), Inches(2.4), ACCENT)
tb(s, Inches(8.56), Inches(1.86), Inches(3.9), Inches(0.4),
   [[("Экономия клиента, ₽/мес", 14, NAVY, True, False)]])
tb(s, Inches(8.56), Inches(2.3), Inches(3.9), Inches(0.7),
   [[("≈ 815 000", 36, ACCENT, True, False)]])
tb(s, Inches(8.56), Inches(3.04), Inches(3.9), Inches(0.4),
   [[("≈ 9,78 млн ₽ / год на одного клиента", 12, MUTE, False, True)]])
tb(s, Inches(8.56), Inches(3.42), Inches(3.9), Inches(0.6),
   [[("Снижение потерь (25 %) +750 000 · экономия на специалистах "
      "+150 000 · подписка −85 000", 12, INK, False, False)]], ls=1.1)
botcells = [("Модель", "B2B SaaS, REST API"), ("Тариф", "85 000 ₽/мес"),
            ("ОПФ", "ООО · УСН 6 %"), ("SOM", "≈ 60 млн ₽/год")]
x = Inches(8.3)
for i, (a, b) in enumerate(botcells):
    rect(s, Inches(8.3), Inches(4.3) + i * Inches(0.47), Inches(4.33),
         Inches(0.44), PANEL2 if i % 2 else PANEL)
    tb(s, Inches(8.5), Inches(4.3) + i * Inches(0.47), Inches(1.3), Inches(0.44),
       [[(a, 12, ACC_DK, True, False)]], anchor=MSO_ANCHOR.MIDDLE)
    tb(s, Inches(9.7), Inches(4.3) + i * Inches(0.47), Inches(2.8), Inches(0.44),
       [[(b, 12, INK, False, False)]], anchor=MSO_ANCHOR.MIDDLE)
conclusion_band(s, "Подписочная SaaS-модель: предельные издержки на нового "
                   "клиента малы, основные инвестиции уже понесены.")

prs.save(str(OUT))
print(f"Saved: {OUT}  ({len(prs.slides._sldIdLst)} slides)")
