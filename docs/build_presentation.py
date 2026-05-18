#!/usr/bin/env python3
"""ВКР defense presentation — rich, structured, academic register.

Design: full-width navy header band (white title + light subtitle + accent
underline), tinted content panels with accent left border, structured
conclusion band, redesigned takeaways. Calibri (portable to PowerPoint).
Academic nominal phrasing — no colloquial verb-phrases.
"""
import struct
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "thesis" / "figures"
OUT = ROOT / "docs" / "Презентация_Лазарев_ВКР.pptx"

NAVY    = RGBColor(0x1B, 0x26, 0x49)   # header band, titles
NAVY2   = RGBColor(0x2E, 0x3D, 0x6B)   # secondary navy
ACCENT  = RGBColor(0xE0, 0x7A, 0x5F)   # terracotta accent
ACC_DK  = RGBColor(0xC2, 0x5A, 0x40)   # darker accent
INK     = RGBColor(0x29, 0x2E, 0x3A)   # body text
MUTE    = RGBColor(0x66, 0x6E, 0x82)   # captions
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
PANEL   = RGBColor(0xEE, 0xF1, 0xF7)   # tinted content panel
PANEL2  = RGBColor(0xF6, 0xF7, 0xFB)   # lighter panel
LBLUE   = RGBColor(0xB7, 0xC2, 0xDE)   # light blue (on navy)
RULE    = RGBColor(0xD7, 0xDD, 0xEA)

FONT = "Calibri"
SW, SH = Inches(13.333), Inches(7.5)
TOTAL = 14


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
    """Full-width navy header band — strong anchor, no pale floating."""
    rect(slide, 0, 0, SW, SH, WHITE)
    band = rect(slide, 0, 0, SW, Inches(1.46), NAVY)
    rect(slide, 0, Inches(1.46), SW, Pt(3), ACCENT)
    tb(slide, Inches(0.7), Inches(0.26), Inches(10.2), Inches(0.72),
       [[(title, 27, WHITE, True, False)]], ls=1.0)
    if subtitle:
        tb(slide, Inches(0.72), Inches(0.98), Inches(10.0), Inches(0.4),
           [[(subtitle, 14, LBLUE, False, True)]])
    # page chip
    chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Inches(11.95), Inches(0.42), Inches(0.95),
                                  Inches(0.5))
    solid(chip, NAVY2)
    tb(slide, Inches(11.95), Inches(0.52), Inches(0.95), Inches(0.34),
       [[(f"{page:02d} / {TOTAL}", 13, WHITE, True, False)]],
       align=PP_ALIGN.CENTER)
    # footer
    rect(slide, Inches(0.7), Inches(7.04), Inches(11.93), Pt(0.75), RULE)
    tb(slide, Inches(0.7), Inches(7.1), Inches(9.0), Inches(0.3),
       [[("Прогнозирование временных рядов · ВКР · Лазарев А.К. · 2026",
          10, MUTE, False, False)]])


def panel(slide, x, y, w, h, title, items, fill=PANEL, size=14, gap=8):
    """Tinted content panel with accent left border + title + bullets.
    Sophisticated grouping block — fills space, not childish."""
    rect(slide, x, y, w, h, fill)
    rect(slide, x, y, Pt(4), h, ACCENT)
    tb(slide, x + Inches(0.28), y + Inches(0.18), w - Inches(0.4),
       Inches(0.4), [[(title, 15, NAVY, True, False)]])
    runs = []
    for it in items:
        if isinstance(it, tuple):
            runs.append([("▪  ", size, ACCENT, True, False),
                         (it[0], size, INK, True, False),
                         (it[1], size, INK, False, False)])
        else:
            runs.append([("▪  ", size, ACCENT, True, False),
                         (it, size, INK, False, False)])
    tb(slide, x + Inches(0.28), y + Inches(0.66), w - Inches(0.5),
       h - Inches(0.8), runs, ls=1.1, sa=gap)


def conclusion_band(slide, text, y=Inches(6.32)):
    """Full-width tinted conclusion strip with accent border — academic."""
    rect(slide, Inches(0.7), y, Inches(11.93), Inches(0.62), PANEL)
    rect(slide, Inches(0.7), y, Pt(4), Inches(0.62), ACCENT)
    tb(slide, Inches(1.0), y + Inches(0.08), Inches(1.7), Inches(0.46),
       [[("ВЫВОД", 13, ACC_DK, True, False)]], anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, Inches(2.15), y, Inches(10.3), Inches(0.62),
       [[(text, 14.5, INK, False, False)]], anchor=MSO_ANCHOR.MIDDLE, ls=1.05)


def keystat(slide, x, y, number, label, w=Inches(3.5)):
    tb(slide, x, y, w, Inches(0.9), [[(number, 38, ACCENT, True, False)]])
    tb(slide, x, y + Inches(0.78), w, Inches(0.85),
       [[(label, 13, MUTE, False, False)]], ls=1.06)


def img_fit(slide, path, x, y, mw, mh):
    w, h = png_size(path)
    ar = w / h
    if ar > mw / mh:
        nw, nh = mw, int(mw / ar)
    else:
        nh, nw = mh, int(mh * ar)
    slide.shapes.add_picture(str(path), x + (mw - nw) // 2,
                             y + (mh - nh) // 2, nw, nh)


prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def new():
    return prs.slides.add_slide(BLANK)

# ===== Slide 1 — Title =====================================================
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

# ===== Slide 2 — Актуальность и цель =======================================
s = new()
header(s, "Актуальность и цель исследования",
       "Раздел 1. Постановка задачи", 2)
panel(s, Inches(0.7), Inches(1.78), Inches(6.05), Inches(4.35),
      "Проблематика", [
    ("Прикладная значимость. ", "Краткосрочный прогноз применяется в планировании спроса, трафика и ресурсов."),
    ("Отсутствие универсальной модели. ", "Статистические методы устойчивы на коротких рядах, нейросетевые — на длинных и сложных."),
    ("Открытый вопрос. ", "Оправдывают ли обучаемая декомпозиция и механизмы долгосрочных зависимостей свою сложность?"),
], size=15, gap=12)
rect(s, Inches(7.0), Inches(1.78), Inches(5.63), Inches(4.35), NAVY)
rect(s, Inches(7.0), Inches(1.78), Pt(4), Inches(4.35), ACCENT)
tb(s, Inches(7.3), Inches(2.0), Inches(5.1), Inches(0.4),
   [[("ЦЕЛЬ РАБОТЫ", 14, ACCENT, True, False)]])
tb(s, Inches(7.3), Inches(2.55), Inches(5.05), Inches(3.4),
   [[("Установить условия, при которых обучаемая декомпозиция временных "
      "рядов и нейросетевые механизмы учёта долгосрочных зависимостей "
      "обеспечивают практическое преимущество над классическими "
      "статистическими моделями, и определить области, где более простые "
      "методы остаются предпочтительными.", 16, WHITE, False, False)]], ls=1.22)

# ===== Slide 3 — Классические модели =======================================
s = new()
header(s, "Классические статистические модели",
       "Раздел 1. Четыре эталонные модели сравнения", 3)
cards = [
    ("Seasonal Naive",
     "Прогноз воспроизводит значения предыдущего сезонного периода. "
     "Не имеет обучаемых параметров.",
     "Роль: эталон для метрики MASE и нижняя граница качества."),
    ("Auto-ARIMA",
     "Линейная модель стационаризованного ряда: авторегрессия, "
     "интегрирование (разности) и скользящее среднее. Порядки (p, d, q) "
     "подбираются автоматически по информационному критерию.",
     "Сильная сторона: устойчивость на коротких рядах."),
    ("ETS",
     "Экспоненциальное сглаживание уровня, тренда и сезонности в "
     "пространстве состояний. Веса α, β, γ оцениваются по правдоподобию.",
     "Сильная сторона: плавная динамика и сезонность."),
    ("Prophet",
     "Аддитивная декомпозиция: кусочно-линейный тренд, сезонные "
     "гармоники и календарные эффекты с байесовской оценкой "
     "параметров.",
     "Особенность: гибкая настройка сезонных компонент."),
]
x0 = Inches(0.7); cw = Inches(2.91); chh = Inches(4.35); gap = Inches(0.13)
for i, (nm, desc, role) in enumerate(cards):
    x = x0 + i * (cw + gap)
    rect(s, x, Inches(1.78), cw, chh, PANEL)
    rect(s, x, Inches(1.78), cw, Inches(0.6), NAVY)
    tb(s, x, Inches(1.89), cw, Inches(0.4),
       [[(nm, 15, WHITE, True, False)]], align=PP_ALIGN.CENTER)
    tb(s, x + Inches(0.22), Inches(2.55), cw - Inches(0.42), Inches(2.7),
       [[(desc, 13.5, INK, False, False)]], ls=1.18)
    rect(s, x + Inches(0.22), Inches(5.2), cw - Inches(0.44), Pt(1.2), RULE)
    tb(s, x + Inches(0.22), Inches(5.32), cw - Inches(0.42), Inches(0.78),
       [[(role, 12.5, ACC_DK, False, True)]], ls=1.12)
conclusion_band(s, "Классические модели интерпретируемы и устойчивы при "
                   "ограниченной истории наблюдений — это сильная база сравнения.")

# ===== Slide 4 — Нейросетевые модели =======================================
s = new()
header(s, "Нейросетевые модели и их архитектуры",
       "Раздел 1. Семь моделей: от линейных до гибридных", 4)
img_fit(s, FIG / "arch_neural_pipelines.png",
        Inches(0.7), Inches(1.66), Inches(8.55), Inches(4.55))
panel(s, Inches(9.42), Inches(1.66), Inches(3.21), Inches(4.55),
      "Семейства", [
    ("Линейные: ", "DLinear"),
    ("Базисные: ", "N-BEATS"),
    ("Свёрточные: ", "TimesNet"),
    ("Трансформеры: ", "Autoformer, FEDformer, PatchTST"),
    ("Гибридные: ", "Helformer"),
], size=13.5, gap=10)
conclusion_band(s, "Модели различаются способом выделения компонент ряда и "
                   "механизмом учёта зависимостей — это и есть предмет сравнения.")

# ===== Slide 5 — Эксперимент ===============================================
s = new()
header(s, "Экспериментальная база и протокол",
       "Раздел 2. Данные, режимы, тип декомпозиции", 5)
panel(s, Inches(0.7), Inches(1.78), Inches(3.8), Inches(4.35),
      "Данные", [
    ("M3 Competition. ", "Полный набор — 2829 рядов: 645 годовых, 756 квартальных, 1428 месячных."),
    ("M4 Competition. ", "3000 рядов: 1000 квартальных, 1000 месячных, 1000 ежедневных."),
    ("Итого ", "6 частотных категорий, 5829 рядов."),
    ("Метрики. ", "sMAPE, MASE, RMSE — для устойчивости выводов к выбору меры."),
], size=14, gap=13)
panel(s, Inches(4.66), Inches(1.78), Inches(3.8), Inches(4.35),
      "Три режима прогнозирования", [
    ("Прямой. ", "Весь горизонт прогнозируется за один проход модели."),
    ("Итеративный. ", "Пошаговое продвижение с подстановкой собственных прогнозов; ошибка накапливается."),
    ("С предобучением. ", "Обучение на корпусе рядов с последующим дообучением на целевом ряде."),
], size=14, gap=13)
panel(s, Inches(8.62), Inches(1.78), Inches(4.0), Inches(4.35),
      "Тип декомпозиции", [
    ("Фиксированная. ", "Правило задано заранее — ETS, DLinear, Helformer."),
    ("Обучаемая. ", "Компоненты выделяет сеть — Autoformer, FEDformer, TimesNet."),
    ("Без декомпозиции. ", "Прямое моделирование — Auto-ARIMA, PatchTST."),
], size=14, gap=13)
conclusion_band(s, "Сопоставление по трём осям — длина ряда, режим обучения "
                   "и тип декомпозиции — формирует структуру анализа.")

# ===== Slide 6 — Прямое прогнозирование ====================================
s = new()
header(s, "Прямое прогнозирование: общая картина",
       "Раздел 3. Сравнительная позиция моделей по частотам", 6)
img_fit(s, FIG / "rank_evolution.png",
        Inches(0.7), Inches(1.7), Inches(8.05), Inches(4.45))
panel(s, Inches(8.9), Inches(1.7), Inches(3.73), Inches(4.45),
      "Наблюдения", [
    ("Лидерство классики. ", "Auto-ARIMA, ETS, Seasonal Naive — верхние ранги на всех частотах."),
    ("Лучшая нейросеть. ", "TimesNet, средний ранг ≈ 4."),
    ("Отставание трансформеров ", "в режиме обучения на одном ряде."),
], size=13.5, gap=10)
conclusion_band(s, "В режиме обучения на одном ряде классические модели "
                   "сохраняют лидерство, однако разрыв убывает с ростом длины ряда.")

# ===== Slide 7 — Длина обучающей выборки ===================================
s = new()
header(s, "Длина обучающей выборки как фактор",
       "Раздел 3. Пороговая зависимость качества от длины ряда", 7)
img_fit(s, FIG / "bar_m4_daily_full.png",
        Inches(0.7), Inches(1.7), Inches(7.7), Inches(4.45))
panel(s, Inches(8.55), Inches(1.7), Inches(4.08), Inches(4.45),
      "Пороговый эффект (≈ 50–100 набл.)", [
    ("Короткие ряды. ", "Уверенное преимущество классических моделей."),
    ("Длинные ряды M4 daily. ", "Нейросети конкурентоспособны."),
    ("TimesNet (3.62) ", "сопоставим с Auto-ARIMA (3.22) и ETS (3.46)."),
], size=13.5, gap=11)
conclusion_band(s, "Существует пороговая длина обучающей выборки: ниже неё "
                   "лидирует классика, выше — нейросетевые модели выходят на её уровень.")

# ===== Slide 8 — Предобучение ==============================================
s = new()
header(s, "Предобучение: эффект переноса знаний",
       "Раздел 3. Относительное улучшение по частотам и моделям", 8)
img_fit(s, FIG / "pretrain_effect_all.png",
        Inches(0.7), Inches(1.7), Inches(7.9), Inches(4.35))
rect(s, Inches(8.75), Inches(1.7), Inches(3.88), Inches(4.35), NAVY)
rect(s, Inches(8.75), Inches(1.7), Pt(4), Inches(4.35), ACCENT)
tb(s, Inches(9.0), Inches(1.92), Inches(3.5), Inches(0.4),
   [[("СРЕДНИЙ ВЫИГРЫШ sMAPE", 12.5, ACCENT, True, False)]])
for i, (v, nm) in enumerate([("+31.5 %", "DLinear"), ("+25.5 %", "Autoformer"),
                              ("+18.2 %", "Helformer")]):
    yy = Inches(2.42) + i * Inches(1.18)
    tb(s, Inches(9.0), yy, Inches(3.5), Inches(0.6),
       [[(v, 30, WHITE, True, False)]])
    tb(s, Inches(9.0), yy + Inches(0.56), Inches(3.5), Inches(0.34),
       [[(nm, 13, LBLUE, False, False)]])
conclusion_band(s, "Предобучение на корпусе рядов выводит ранее слабые "
                   "нейросетевые модели на конкурентоспособный уровень.")

# ===== Slide 9 — Предобучение: эффект на прогнозе ==========================
s = new()
header(s, "Предобучение: эффект на прогнозе ряда",
       "Раздел 3. Одна модель: обучение с нуля против предобучения", 9)
img_fit(s, FIG / "pretrain_forecast_autoformer_D2111.png",
        Inches(0.7), Inches(1.7), Inches(7.9), Inches(4.4))
panel(s, Inches(8.75), Inches(1.7), Inches(3.88), Inches(4.4),
      "Что показывает график", [
    ("Серая линия — ", "обучающая история, чёрная — фактические значения."),
    ("Пунктир (с нуля). ", "Прогноз Autoformer без предобучения, sMAPE 3.8 %."),
    ("Сплошная (предобучен). ", "После предобучения sMAPE 1.0 % — прогноз плотно следует факту."),
], size=13.5, gap=12)
conclusion_band(s, "Предобучение на корпусе рядов существенно повышает "
                   "точность прогноза одной и той же модели.")

# ===== Slide 10 — Сравнительная таблица ====================================
s = new()
header(s, "Сводное сравнение: sMAPE по категориям",
       "Раздел 3. Предобученные нейросети и классические модели", 10)
rows = [
    ("Модель", ["M3-Y", "M3-Q", "M3-M", "M4-Q", "M4-M", "M4-D", "Ср."], "head"),
    ("Классические модели (per-series)", [""]*7, "group"),
    ("Auto-ARIMA",     ["17.9","10.9","18.1","10.2","14.7","3.22","12.5"], "classic"),
    ("ETS",            ["19.1","10.8","17.4","9.7","15.1","3.46","12.6"], "classic"),
    ("Seasonal Naive", ["17.9","11.1","17.2","12.2","15.9","4.05","13.0"], "classic"),
    ("Нейросетевые модели (с предобучением)", [""]*7, "group"),
    ("DLinear",   ["23.7","12.7","14.0","15.4","17.2","2.44","14.2"], "neural"),
    ("TimesNet",  ["25.1","12.4","13.7","22.5","17.2","2.46","15.6"], "neural"),
    ("PatchTST",  ["23.1","15.1","15.2","19.6","17.5","2.36","15.5"], "neural"),
    ("Autoformer",["23.4","13.8","15.1","20.1","16.3","2.81","15.3"], "neural"),
    ("FEDformer", ["22.8","15.8","16.3","19.6","15.6","4.16","15.7"], "neural"),
    ("N-BEATS",   ["23.8","14.4","17.3","21.5","18.3","3.20","16.4"], "neural"),
    ("Helformer", ["29.8","21.4","23.0","20.5","46.3","2.87","24.0"], "neural"),
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
            if kind == "classic":
                r.font.color.rgb = MUTE
            else:
                r.font.color.rgb = INK
            if j == 0:
                r.font.bold = True
            # highlight where pretrained neural beats best classical
            if kind == "neural" and j in (3, 6):
                c.fill.fore_color.rgb = RGBColor(0xFC, 0xE9, 0xE2)
                r.font.color.rgb = ACC_DK; r.font.bold = True
conclusion_band(s, "На M3 monthly и M4 daily предобученные нейросети "
                   "превосходят лучшую классику; на коротких рядах классика "
                   "сохраняет преимущество.", y=Inches(6.46))

# ===== Slide 11 — Анализ отдельных рядов ===================================
s = new()
header(s, "Анализ отдельных временных рядов",
       "Раздел 3. Массовый характер преимущества нейросетей", 11)
img_fit(s, FIG / "cherry_D2052.png",
        Inches(0.7), Inches(1.7), Inches(8.1), Inches(4.45))
rect(s, Inches(8.95), Inches(1.7), Inches(3.68), Inches(4.45), NAVY)
rect(s, Inches(8.95), Inches(1.7), Pt(4), Inches(4.45), ACCENT)
tb(s, Inches(9.2), Inches(2.0), Inches(3.2), Inches(0.9),
   [[("517", 44, WHITE, True, False)]])
tb(s, Inches(9.2), Inches(2.86), Inches(3.2), Inches(0.7),
   [[("из 1000 ежедневных рядов M4 — нейросеть точнее классики",
      13, LBLUE, False, False)]], ls=1.08)
tb(s, Inches(9.2), Inches(3.92), Inches(3.2), Inches(0.9),
   [[("295", 44, WHITE, True, False)]])
tb(s, Inches(9.2), Inches(4.78), Inches(3.2), Inches(0.7),
   [[("рядов, где побеждают 3 и более нейросетевых модели",
      13, LBLUE, False, False)]], ls=1.08)
conclusion_band(s, "Преимущество нейросетевых моделей на длинных рядах носит "
                   "массовый, а не единичный характер.")

# ===== Slide 12 — Итеративное прогнозирование ==============================
s = new()
header(s, "Итеративное прогнозирование",
       "Раздел 3. Устойчивость моделей к накоплению ошибки", 12)
img_fit(s, FIG / "rollout_m3_quarterly_T478.png",
        Inches(0.7), Inches(1.7), Inches(7.85), Inches(4.4))
panel(s, Inches(8.7), Inches(1.7), Inches(3.93), Inches(4.4),
      "Прямой против итеративного", [
    ("На графике ", "пунктир — прямой прогноз, сплошная — итеративный."),
    ("Обучаемая декомпозиция уязвима. ", "FEDformer и TimesNet теряют около 5 п.п. sMAPE."),
    ("PatchTST устойчив. ", "Наименьшая деградация среди трансформеров."),
    ("Классика стабильна. ", "ETS и Auto-ARIMA нечувствительны к режиму."),
], size=13, gap=11)
conclusion_band(s, "Сложная обучаемая декомпозиция повышает риск накопления "
                   "ошибки; тип декомпозиции должен соответствовать режиму "
                   "прогнозирования.")

# ===== Slide 13 — Выводы (structured matrix) ===============================
s = new()
header(s, "Выводы и практические рекомендации",
       "Раздел 4. Основные результаты исследования", 13)
findings = [
    ("Длина обучающей выборки",
     "Определяющий фактор. Ниже ≈ 50–100 наблюдений предпочтительны ETS и Auto-ARIMA; выше — нейросети конкурентоспособны."),
    ("TimesNet — лучшая нейросеть",
     "В режиме обучения на одном ряде; обучаемая 2D-декомпозиция эффективна при достаточном объёме данных."),
    ("Предобучение",
     "Повышает качество нейросетей на 18–32 % и выводит их на уровень и выше классики на M3 monthly и M4 daily."),
    ("Тип декомпозиции и режим",
     "Фиксированная декомпозиция устойчивее, обучаемая адаптивнее, но хрупка при итеративном прогнозировании."),
    ("Общий вывод",
     "Эффективность нейросетевых механизмов не абсолютна — она определяется длиной ряда, режимом обучения и объёмом данных."),
]
y = Inches(1.74); rh = Inches(1.0)
for i, (t, d) in enumerate(findings):
    fill = PANEL if i % 2 == 0 else PANEL2
    rect(s, Inches(0.7), y, Inches(11.93), rh - Inches(0.06), fill)
    rect(s, Inches(0.7), y, Pt(4), rh - Inches(0.06), ACCENT)
    num = slide_num = slide if False else s
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

# ===== Slide 14 — Заключение ===============================================
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

prs.save(str(OUT))
print(f"Saved: {OUT}  ({len(prs.slides._sldIdLst)} slides)")
