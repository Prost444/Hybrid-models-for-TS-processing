#!/usr/bin/env python3
"""Build the ВКР defense presentation (.pptx) — mature, clean, portable.

Design: deep navy + warm terracotta accent, light background, Calibri (safe
for PowerPoint export). Thin top accent bar, title + italic subtitle, footer
with slide number. No childish card-blocks. Real figures from thesis/figures.
"""
import struct
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "thesis" / "figures"
OUT = ROOT / "docs" / "Презентация_Лазарев_ВКР.pptx"

# ---- Palette --------------------------------------------------------------
NAVY      = RGBColor(0x1F, 0x2A, 0x4E)   # titles, top bar
NAVY_SOFT = RGBColor(0x37, 0x46, 0x76)   # subtitles
ACCENT    = RGBColor(0xE0, 0x7A, 0x5F)   # warm terracotta — key highlights
INK       = RGBColor(0x2B, 0x2B, 0x2B)   # body text
MUTE      = RGBColor(0x5A, 0x64, 0x78)   # captions
LIGHTBG   = RGBColor(0xFF, 0xFF, 0xFF)
PANEL     = RGBColor(0xF3, 0xF5, 0xFA)   # very light panel (used sparingly)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
RULE      = RGBColor(0xD8, 0xDD, 0xE8)

FONT = "Calibri"
SW, SH = Inches(13.333), Inches(7.5)


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return w, h
    return 1600, 900


def solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def textbox(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, line_spacing=1.0, space_after=0):
    """runs: list of paragraphs; each paragraph = list of (text, size, color, bold, italic)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        if space_after:
            p.space_after = Pt(space_after)
        for (txt, size, color, bold, italic) in para:
            r = p.add_run()
            r.text = txt
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.italic = italic
    return tb


def base(slide, title, subtitle, page, total):
    """Common chrome: bg, top accent bar, title, italic subtitle, footer."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    solid(bg, LIGHTBG)
    bg.shadow.inherit = False
    # top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.16))
    solid(bar, NAVY)
    bar.shadow.inherit = False
    # title
    textbox(slide, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.95),
            [[(title, 30, NAVY, True, False)]], line_spacing=1.0)
    if subtitle:
        textbox(slide, Inches(0.72), Inches(1.22), Inches(11.9), Inches(0.45),
                [[(subtitle, 16, NAVY_SOFT, False, True)]])
    # footer rule + page
    fr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(7.02),
                                Inches(11.93), Pt(0.75))
    solid(fr, RULE)
    fr.shadow.inherit = False
    textbox(slide, Inches(11.4), Inches(7.08), Inches(1.2), Inches(0.3),
            [[(f"{page} / {total}", 11, MUTE, False, False)]],
            align=PP_ALIGN.RIGHT)
    textbox(slide, Inches(0.7), Inches(7.08), Inches(7.0), Inches(0.3),
            [[("Прогнозирование временных рядов · ВКР · Лазарев А.К.",
               10, MUTE, False, False)]])


def add_image_fit(slide, path, x, y, max_w, max_h):
    w, h = png_size(path)
    ar = w / h
    box_ar = max_w / max_h
    if ar > box_ar:
        nw = max_w
        nh = int(max_w / ar)
    else:
        nh = max_h
        nw = int(max_h * ar)
    px = x + (max_w - nw) // 2
    py = y + (max_h - nh) // 2
    slide.shapes.add_picture(str(path), px, py, nw, nh)


def bullets(slide, x, y, w, h, items, size=17, gap=10, accent_lead=True):
    """Clean bullets — small accent square marker, no card boxes."""
    runs = []
    for it in items:
        if isinstance(it, tuple):
            head, tail = it
            runs.append([("▪  ", size, ACCENT, True, False),
                         (head, size, INK, True, False),
                         (tail, size, INK, False, False)])
        else:
            runs.append([("▪  ", size, ACCENT, True, False),
                         (it, size, INK, False, False)])
    textbox(slide, x, y, w, h, runs, line_spacing=1.12, space_after=gap)


def keystat(slide, x, y, number, label, w=Inches(3.0)):
    """A large accent number with a label underneath — for key takeaways."""
    textbox(slide, x, y, w, Inches(0.95),
            [[(number, 40, ACCENT, True, False)]])
    textbox(slide, x, y + Inches(0.85), w, Inches(0.7),
            [[(label, 13, MUTE, False, False)]], line_spacing=1.05)


prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]
TOTAL = 12


def new():
    return prs.slides.add_slide(BLANK)

# ---- Slide 1 — Title ------------------------------------------------------
s = new()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
solid(bg, NAVY); bg.shadow.inherit = False
strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.05), SW, Pt(2))
solid(strip, ACCENT); strip.shadow.inherit = False
textbox(s, Inches(0.9), Inches(0.7), Inches(11.5), Inches(0.4),
        [[("МИРЭА — Российский технологический университет", 15, WHITE, False, False)]])
textbox(s, Inches(0.9), Inches(1.06), Inches(11.5), Inches(0.4),
        [[("Институт искусственного интеллекта · Кафедра высшей математики",
           13, RGBColor(0xB9, 0xC3, 0xDE), False, True)]])
textbox(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(0.4),
        [[("ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА", 14, ACCENT, True, False)]])
textbox(s, Inches(0.9), Inches(2.55), Inches(11.6), Inches(2.0),
        [[("Прогнозирование временных рядов за счёт обучаемой", 31, WHITE, True, False)],
         [("декомпозиции и нейросетевых механизмов учёта", 31, WHITE, True, False)],
         [("долгосрочных зависимостей", 31, WHITE, True, False)]],
        line_spacing=1.05)
textbox(s, Inches(0.9), Inches(5.35), Inches(11.5), Inches(0.45),
        [[("Выполнил: ", 15, RGBColor(0xB9,0xC3,0xDE), False, False),
          ("студент группы КМБО-03-22 — Лазарев А.К.", 15, WHITE, True, False)]])
textbox(s, Inches(0.9), Inches(5.8), Inches(11.5), Inches(0.45),
        [[("Научный руководитель: ", 15, RGBColor(0xB9,0xC3,0xDE), False, False),
          ("к.ф.-м.н., доцент кафедры ВМ — Петрусевич Д.А.", 15, WHITE, True, False)]])
textbox(s, Inches(0.9), Inches(6.6), Inches(4.0), Inches(0.4),
        [[("Москва, 2026", 13, RGBColor(0x9A,0xA6,0xC8), False, False)]])

# ---- Slide 2 — Актуальность и цель ---------------------------------------
s = new()
base(s, "Актуальность и цель работы", "Раздел 1. Постановка исследования", 2, TOTAL)
bullets(s, Inches(0.7), Inches(1.95), Inches(6.4), Inches(4.4), [
    ("Краткосрочный прогноз ", "временных рядов нужен в планировании: продажи, трафик, спрос, ресурсы."),
    ("Универсальной модели нет: ", "классика устойчива на коротких рядах, нейросети сильнее на длинных и сложных."),
    ("Современные архитектуры ", "включают обучаемую декомпозицию и механизмы долгосрочных зависимостей — но дают ли они реальный выигрыш?"),
], size=17, gap=14)
# Goal panel (subtle, not a childish box: thin left accent rule)
rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.4), Inches(2.0), Pt(3), Inches(3.6))
solid(rule, ACCENT); rule.shadow.inherit = False
textbox(s, Inches(7.7), Inches(2.0), Inches(5.0), Inches(0.4),
        [[("ЦЕЛЬ РАБОТЫ", 14, ACCENT, True, False)]])
textbox(s, Inches(7.7), Inches(2.5), Inches(5.05), Inches(3.2),
        [[("Определить условия, при которых обучаемая декомпозиция и "
           "нейросетевые механизмы учёта долгосрочных зависимостей дают "
           "практическое преимущество над классическими статистическими "
           "моделями — и когда более простые методы предпочтительнее.",
           17, INK, False, False)]], line_spacing=1.18)

# ---- Slide 3 — Модели и эксперимент --------------------------------------
s = new()
base(s, "Модели и масштаб эксперимента", "Раздел 1. Что и на чём сравнивалось", 3, TOTAL)
textbox(s, Inches(0.7), Inches(1.95), Inches(6.0), Inches(0.4),
        [[("Классические модели (4)", 18, NAVY, True, False)]])
bullets(s, Inches(0.7), Inches(2.45), Inches(5.6), Inches(1.8), [
    "Seasonal Naive · Auto-ARIMA", "ETS · Prophet"], size=16, gap=6)
textbox(s, Inches(0.7), Inches(3.7), Inches(6.0), Inches(0.4),
        [[("Нейросетевые модели (7)", 18, NAVY, True, False)]])
bullets(s, Inches(0.7), Inches(4.2), Inches(5.7), Inches(2.2), [
    "DLinear · N-BEATS · TimesNet",
    "Autoformer · FEDformer · PatchTST",
    "Helformer (гибрид: HW-декомпозиция + нейросеть)"], size=16, gap=6)
# right: data + regimes
rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.4), Inches(2.0), Pt(3), Inches(4.5))
solid(rule, ACCENT); rule.shadow.inherit = False
textbox(s, Inches(7.7), Inches(2.0), Inches(5.0), Inches(0.4),
        [[("ДАННЫЕ", 14, ACCENT, True, False)]])
bullets(s, Inches(7.7), Inches(2.5), Inches(5.0), Inches(1.8), [
    ("M3 Competition — ", "2829 рядов (полный набор)"),
    ("M4 Competition — ", "3000 рядов (Q + M + D)"),
    "6 частотных категорий, 5829 рядов"], size=15, gap=7)
textbox(s, Inches(7.7), Inches(4.5), Inches(5.0), Inches(0.4),
        [[("ТРИ РЕЖИМА", 14, ACCENT, True, False)]])
bullets(s, Inches(7.7), Inches(5.0), Inches(5.0), Inches(1.5), [
    "Прямое прогнозирование",
    "Итеративное прогнозирование",
    "Предобучение на корпусе рядов"], size=15, gap=6)

# ---- Slide 4 — Методология -----------------------------------------------
s = new()
base(s, "Методология: декомпозиция и зависимости",
     "Раздел 2. Два ключевых механизма", 4, TOTAL)
bullets(s, Inches(0.7), Inches(1.95), Inches(5.0), Inches(3.5), [
    ("Фиксированная декомпозиция ", "— ETS, DLinear, Helformer: тренд и сезонность выделяются заранее заданным правилом."),
    ("Обучаемая декомпозиция ", "— Autoformer, FEDformer, TimesNet: компоненты выделяет сама сеть в ходе обучения."),
    ("Долгосрочные зависимости ", "— внимание, свёртки, линейные проекции."),
], size=16, gap=12)
add_image_fit(s, FIG / "decomposition_type_bars.png",
              Inches(5.95), Inches(1.95), Inches(6.7), Inches(4.4))
textbox(s, Inches(5.95), Inches(6.35), Inches(6.7), Inches(0.4),
        [[("Средний sMAPE и MASE по типу декомпозиции — обучаемая "
           "выигрывает только при достатке данных.", 12, MUTE, False, True)]])

# ---- Slide 5 — Прямое прогнозирование ------------------------------------
s = new()
base(s, "Прямое прогнозирование: общая картина",
     "Раздел 3. Кто лидирует по частотам", 5, TOTAL)
add_image_fit(s, FIG / "rank_evolution.png",
              Inches(0.7), Inches(1.85), Inches(8.0), Inches(4.7))
bullets(s, Inches(8.95), Inches(2.1), Inches(3.7), Inches(4.2), [
    "Классика (ARIMA, ETS, S. Naive) стабильно лидирует",
    "TimesNet — лучшая нейросеть, средний ранг ≈ 4",
    "Трансформеры в per-series режиме отстают",
    "С ростом частоты разрыв сокращается"], size=15, gap=12)

# ---- Slide 6 — Длина ряда -------------------------------------------------
s = new()
base(s, "Длина ряда — ключевой фактор",
     "Раздел 3. Где нейросети догоняют классику", 6, TOTAL)
add_image_fit(s, FIG / "bar_m4_daily_full.png",
              Inches(0.7), Inches(1.85), Inches(7.8), Inches(4.7))
textbox(s, Inches(8.7), Inches(2.0), Inches(4.0), Inches(0.4),
        [[("Порог ≈ 50–100 наблюдений", 17, NAVY, True, False)]])
bullets(s, Inches(8.7), Inches(2.6), Inches(3.95), Inches(3.0), [
    "Короткие ряды — преимущество классики",
    "Длинные ежедневные ряды M4 — нейросети конкурентоспособны",
    "TimesNet вплотную к Auto-ARIMA"], size=15, gap=11)

# ---- Slide 7 — Предобучение ----------------------------------------------
s = new()
base(s, "Предобучение — главный результат",
     "Раздел 3. Эффект переноса знаний", 7, TOTAL)
add_image_fit(s, FIG / "pretrain_effect_all.png",
              Inches(0.7), Inches(1.85), Inches(8.1), Inches(4.6))
keystat(s, Inches(9.1), Inches(2.0), "+31.5%", "DLinear — наибольший выигрыш")
keystat(s, Inches(9.1), Inches(3.6), "+25.5%", "Autoformer")
keystat(s, Inches(9.1), Inches(5.1), "+18.2%", "Helformer")

# ---- Slide 8 — Предобученные нейросети vs классика ------------------------
s = new()
base(s, "Предобученные нейросети обходят классику",
     "Раздел 3. M3 monthly и M4 daily", 8, TOTAL)
add_image_fit(s, FIG / "cherry_pretrained_D2111.png",
              Inches(0.7), Inches(1.9), Inches(7.9), Inches(4.5))
bullets(s, Inches(8.8), Inches(2.1), Inches(3.85), Inches(4.0), [
    "M3 monthly: предобученный TimesNet точнее ETS",
    "M4 daily: предобученный PatchTST обходит Auto-ARIMA и ETS",
    "Предобучение сдвигает порог конкурентоспособности вниз"], size=15, gap=13)

# ---- Slide 9 — Анализ отдельных рядов -------------------------------------
s = new()
base(s, "Анализ отдельных рядов",
     "Раздел 3. Где нейросети явно точнее", 9, TOTAL)
add_image_fit(s, FIG / "cherry_D2052.png",
              Inches(0.7), Inches(1.9), Inches(8.3), Inches(4.5))
keystat(s, Inches(9.3), Inches(2.0), "517", "из 1000 ежедневных рядов\nнейросеть лучше классики")
keystat(s, Inches(9.3), Inches(4.0), "295", "рядов, где побеждают\n3+ нейросетевых модели")

# ---- Slide 10 — Итеративное прогнозирование ------------------------------
s = new()
base(s, "Итеративное прогнозирование",
     "Раздел 3. Хрупкость обучаемой декомпозиции", 10, TOTAL)
bullets(s, Inches(0.7), Inches(2.1), Inches(7.4), Inches(4.0), [
    ("При накоплении ошибки ", "сильнее всего деградируют модели с обучаемой декомпозицией."),
    ("FEDformer и TimesNet ", "теряют около +5 п.п. sMAPE при переходе к итеративному режиму."),
    ("PatchTST ", "(без явной декомпозиции) — наиболее устойчивый трансформер."),
    ("Классические модели ", "(ETS, Auto-ARIMA) практически нечувствительны к режиму."),
], size=17, gap=15)
rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.5), Inches(2.3), Pt(3), Inches(3.4))
solid(rule, ACCENT); rule.shadow.inherit = False
textbox(s, Inches(8.8), Inches(2.3), Inches(3.9), Inches(3.6),
        [[("Вывод: ", 17, ACCENT, True, False),
          ("сложная обучаемая декомпозиция — это риск при многошаговом "
           "прогнозе. Простые и классические модели предпочтительнее, "
           "если нужен итеративный режим.", 17, INK, False, False)]],
        line_spacing=1.18)

# ---- Slide 11 — Выводы ----------------------------------------------------
s = new()
base(s, "Выводы и практические рекомендации",
     "Раздел 4. Основные результаты", 11, TOTAL)
bullets(s, Inches(0.7), Inches(1.9), Inches(11.9), Inches(4.8), [
    ("Длина ряда решает. ", "Ниже ≈ 50–100 наблюдений лидирует классика (ETS, Auto-ARIMA); выше — нейросети конкурентоспособны."),
    ("TimesNet — лучшая нейросеть ", "в per-series режиме; обучаемая 2D-декомпозиция эффективна при достатке данных."),
    ("Предобучение кардинально усиливает нейросети ", "(+18…32 %) и выводит их на уровень и выше классики (M3 monthly, M4 daily)."),
    ("Тип декомпозиции должен соответствовать режиму: ", "фиксированная — устойчивее, обучаемая — адаптивнее, но хрупка при итеративном прогнозе."),
    ("Ответ не бинарный: ", "эффективность нейросетевых механизмов зависит от длины ряда, режима обучения и объёма данных."),
], size=16.5, gap=13)

# ---- Slide 12 — Спасибо ---------------------------------------------------
s = new()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
solid(bg, NAVY); bg.shadow.inherit = False
strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(4.3), SW, Pt(2))
solid(strip, ACCENT); strip.shadow.inherit = False
textbox(s, Inches(0.9), Inches(2.7), Inches(11.5), Inches(1.1),
        [[("Спасибо за внимание!", 40, WHITE, True, False)]])
textbox(s, Inches(0.9), Inches(4.55), Inches(11.5), Inches(0.5),
        [[("Прогнозирование временных рядов за счёт обучаемой декомпозиции "
           "и нейросетевых механизмов учёта долгосрочных зависимостей",
           15, RGBColor(0xB9,0xC3,0xDE), False, True)]])
textbox(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(0.4),
        [[("Лазарев А.К. · КМБО-03-22 · Научный руководитель Петрусевич Д.А.",
           14, WHITE, False, False)]])

prs.save(str(OUT))
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides._sldIdLst)}")
