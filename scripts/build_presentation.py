#!/usr/bin/env python3
"""Build full VIK presentation with graphs, tables, and detailed content."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from copy import deepcopy
from lxml import etree
import os

INPUT = "docs/Lazarev-Presentation-VIK-25-26.pptx"
OUTPUT = "docs/Lazarev-Presentation-VIK-25-26-updated.pptx"
FIGS = "thesis/figures"

NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS = {'a': NS_A}


def clear_and_set(shape, new_text, font_size=None):
    """Clear shape text and set new multi-line text preserving first run format."""
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    txBody = tf._txBody

    first_rPr = None
    first_pPr = None
    for p in txBody.findall('.//a:p', NS):
        pPr = p.find('a:pPr', NS)
        if pPr is not None and first_pPr is None:
            first_pPr = deepcopy(pPr)
        for r in p.findall('a:r', NS):
            rPr = r.find('a:rPr', NS)
            if rPr is not None and first_rPr is None:
                first_rPr = deepcopy(rPr)
            break
        if first_rPr is not None:
            break

    for p in txBody.findall('a:p', NS):
        txBody.remove(p)

    lines = new_text.split('\n')
    for line in lines:
        p = etree.SubElement(txBody, f'{{{NS_A}}}p')
        if first_pPr is not None:
            p.append(deepcopy(first_pPr))
        if line.strip():
            r = etree.SubElement(p, f'{{{NS_A}}}r')
            rPr_new = deepcopy(first_rPr) if first_rPr is not None else etree.SubElement(r, f'{{{NS_A}}}rPr')
            if first_rPr is None:
                rPr_new.set('lang', 'ru-RU')
                rPr_new.set('dirty', '0')
            if font_size is not None:
                rPr_new.set('sz', str(int(font_size * 100)))
            r.append(rPr_new)
            t = etree.SubElement(r, f'{{{NS_A}}}t')
            t.text = line


def add_slide_from_layout(prs, layout_idx):
    """Add a new slide from a layout."""
    layout = prs.slide_layouts[layout_idx]
    return prs.slides.add_slide(layout)


def add_image_to_slide(slide, img_path, left, top, width, height=None):
    """Add an image to a slide."""
    if height:
        slide.shapes.add_picture(img_path, left, top, width, height)
    else:
        slide.shapes.add_picture(img_path, left, top, width)


def main():
    prs = Presentation(INPUT)
    slides = list(prs.slides)

    # ===== SLIDE 1: TITLE — keep unchanged =====

    # ===== SLIDE 2: Актуальность и результаты =====
    slide2 = slides[1]
    for s in slide2.shapes:
        if not s.has_text_frame:
            continue
        text = s.text_frame.text
        if "Актуальность тематики" in text:
            clear_and_set(s, "Актуальность исследования")
        elif "Ожидаемые результаты" in text:
            clear_and_set(s, "Полученные результаты")
        elif "Временные ряды лежат" in text:
            clear_and_set(s,
                "Временные ряды являются основным инструментом моделирования "
                "динамических процессов в экономике, энергетике, транспорте "
                "и финансах. Точность прогноза напрямую определяет качество "
                "планирования ресурсов и оценки рисков.\n"
                "\n"
                "Классические модели (ARIMA, ETS) хорошо изучены, но требуют "
                "ручной настройки и плохо обрабатывают нестационарность.\n"
                "\n"
                "Нейросетевые трансформеры (Autoformer, FEDformer, PatchTST) "
                "внедряют обучаемую декомпозицию ряда на тренд и сезонность "
                "непосредственно внутри архитектуры, однако их эффективность "
                "зависит от длины ряда и режима обучения.\n"
                "\n"
                "Центральный вопрос: когда обучаемая декомпозиция эффективнее "
                "фиксированной статистической, и при каких условиях "
                "гибридный подход (Helformer) оказывается лучшим?", font_size=11)
        elif "Реализовать и исследовать" in text:
            clear_and_set(s,
                "Реализован контур из 11 моделей:\n"
                "\u2022 Классические: ARIMA, ETS, Prophet, Seasonal Naive\n"
                "\u2022 Нейросетевые с обучаемой декомпозицией:\n"
                "   DLinear, N-BEATS, Autoformer, FEDformer\n"
                "\u2022 Нейросетевые без декомпозиции:\n"
                "   PatchTST, TimesNet\n"
                "\u2022 Гибридная: Helformer (HW + трансформер)\n"
                "\n"
                "Сравнение на 180 рядах (M3 + M4 Competition)\n"
                "по 6 метрикам: sMAPE, MAPE, RMSE, MSE, MAE, MASE\n"
                "\n"
                "Три режима оценки:\n"
                "\u2022 Прямой прогноз (direct)\n"
                "\u2022 Итеративный прогноз (rollout)\n"
                "\u2022 Предобучение трансформеров (pretrain)\n"
                "\n"
                "Главный результат: Helformer \u2014 лучшая модель\n"
                "на M3 (sMAPE 10,61%, MASE 0,94)", font_size=11)

    # ===== SLIDE 3: Новизна =====
    slide3 = slides[2]
    for s in slide3.shapes:
        if not s.has_text_frame:
            continue
        text = s.text_frame.text
        if "Новизна планируемых" in text:
            clear_and_set(s, "Научная новизна")
        elif "Практическая значимость планируемых" in text:
            clear_and_set(s, "Практическая значимость")
        elif "Выполняется углублённое" in text:
            clear_and_set(s,
                "1. Проведено систематическое сравнение фиксированной "
                "и обучаемой декомпозиции на 180 рядах M3/M4 "
                "с участием 11 моделей \u2014 крупнейшее в данной "
                "постановке.\n"
                "\n"
                "2. Helformer (HW-декомпозиция + трансформер) "
                "превосходит все чисто нейросетевые модели:\n"
                "   \u2022 sMAPE 10,61% vs 13,89% у ETS\n"
                "   \u2022 sMAPE 10,61% vs 17,87% у TimesNet\n"
                "   \u2022 MASE 0,94 vs 1,47 у ETS\n"
                "\n"
                "3. Выявлена граница ~50\u2013100 наблюдений: ниже "
                "неё нейросети уступают статистике.\n"
                "\n"
                "4. Предобучение трансформеров на корпусе рядов "
                "даёт +7\u201311% sMAPE.", font_size=11)
        elif "Улучшение качества" in text:
            clear_and_set(s,
                "Воспроизводимый программный стенд на Python/PyTorch "
                "с модульной архитектурой: фабрики моделей, JSON-конфиги, "
                "чекпоинтирование, GPU-поддержка.\n"
                "\n"
                "Практические рекомендации по выбору модели:\n"
                "\u2022 Короткие ряды с сезонностью \u2192 Helformer\n"
                "\u2022 Длинные ряды (>100 точек) \u2192 DLinear или N-BEATS\n"
                "\u2022 Средние ряды \u2192 ETS или TimesNet\n"
                "\u2022 При наличии корпуса \u2192 предобучение трансформеров\n"
                "\n"
                "Результаты применимы к прогнозированию в финансах, "
                "энергетике, логистике, городской инфраструктуре.", font_size=11)

    # ===== SLIDE 4: Описание (Методология) =====
    slide4 = slides[3]
    for s in slide4.shapes:
        if not s.has_text_frame:
            continue
        if "Описание проекта" in s.text_frame.text:
            clear_and_set(s, "Методология исследования")
        elif "Проект посвящён" in s.text_frame.text:
            clear_and_set(s,
                "Отличие фиксированной и обучаемой декомпозиции:\n"
                "\n"
                "Фиксированная декомпозиция (ETS, Helformer): ряд разлагается "
                "методом Хольта\u2013Уинтерса на тренд, сезонность и остаток ДО "
                "подачи в нейросеть. Параметры декомпозиции определяются "
                "статистически, нейросеть обрабатывает только остатки.\n"
                "\n"
                "Обучаемая декомпозиция (Autoformer, DLinear): декомпозиция "
                "встроена в саму нейросеть. Autoformer выполняет progressive "
                "decomposition на каждом слое энкодера через скользящее среднее, "
                "а DLinear использует обучаемое ядро для разделения тренда "
                "и остатка с последующей линейной проекцией.\n"
                "\n"
                "Протокол эксперимента:\n"
                "\u2022 Per-series обучение: каждая нейросеть обучается отдельно "
                "на каждом ряде (80 эпох, AdamW, z-score нормализация)\n"
                "\u2022 M3: 90 рядов (30 yearly H=6, 30 quarterly H=8, "
                "30 monthly H=18)\n"
                "\u2022 M4: 90 рядов (30 quarterly H=8, 30 monthly H=18, "
                "30 daily H=14)\n"
                "\u2022 Seed=42, GPU RTX A4000, воспроизводимые конфигурации", font_size=10)

    # ===== NEW SLIDE: Results heatmap (layout 2 = column + picture) =====
    slide_results = add_slide_from_layout(prs, 2)  # Белый слайд_колонка и рисунок
    for ph in slide_results.placeholders:
        if ph.placeholder_format.idx == 0:  # title
            clear_and_set(ph, "Результаты: кросс-датасетное сравнение")
        elif ph.placeholder_format.idx == 10:  # left column header
            clear_and_set(ph, "Основные наблюдения")
        elif ph.placeholder_format.idx == 11:  # left column body
            clear_and_set(ph,
                "Helformer \u2014 лучшая модель на M3:\n"
                "sMAPE 10,61% (vs 13,89% ETS)\n"
                "\n"
                "ETS \u2014 лучшая статистическая:\n"
                "стабильно top-2 на всех частотах\n"
                "\n"
                "DLinear \u2014 лучший на M4 daily:\n"
                "sMAPE 2,43% (длинные ряды)\n"
                "\n"
                "Autoformer, FEDformer \u2014\n"
                "слабейшие (sMAPE 22\u201334%),\n"
                "обучаемая decomp не окупается\n"
                "при per-series обучении",
                font_size=11)
        elif ph.placeholder_format.idx == 12:  # picture
            ph.insert_picture(os.path.join(FIGS, "heatmap_smape.png"))

    # ===== NEW SLIDE: Rollout (layout 2) =====
    slide_rollout = add_slide_from_layout(prs, 2)
    for ph in slide_rollout.placeholders:
        if ph.placeholder_format.idx == 0:
            clear_and_set(ph, "Итеративное прогнозирование (rollout)")
        elif ph.placeholder_format.idx == 10:
            clear_and_set(ph, "Деградация при rollout")
        elif ph.placeholder_format.idx == 11:
            clear_and_set(ph,
                "При итеративном прогнозе модель\n"
                "предсказывает по одному шагу,\n"
                "используя собственные предсказания\n"
                "как входные данные.\n"
                "\n"
                "Деградация sMAPE (M3):\n"
                "\u2022 FEDformer: +6,3 п.п.\n"
                "\u2022 Autoformer: +4,9 п.п.\n"
                "\u2022 TimesNet: +4,3 п.п.\n"
                "\u2022 N-BEATS: +3,9 п.п.\n"
                "\u2022 PatchTST: +1,8 п.п. (лучший)\n"
                "\u2022 DLinear: +0,4 п.п.\n"
                "\u2022 ETS, ARIMA: 0,0 п.п.\n"
                "\n"
                "Вывод: сложная обучаемая\n"
                "декомпозиция наименее устойчива",
                font_size=11)
        elif ph.placeholder_format.idx == 12:
            ph.insert_picture(os.path.join(FIGS, "rollout_m3_quarterly_T478.png"))

    # ===== NEW SLIDE: Pretrain (layout 2) =====
    slide_pretrain = add_slide_from_layout(prs, 2)
    for ph in slide_pretrain.placeholders:
        if ph.placeholder_format.idx == 0:
            clear_and_set(ph, "Предобучение трансформеров")
        elif ph.placeholder_format.idx == 10:
            clear_and_set(ph, "Эффект предобучения")
        elif ph.placeholder_format.idx == 11:
            clear_and_set(ph,
                "Вместо обучения каждой модели\n"
                "с нуля на одном ряде \u2014\n"
                "предобучение на всех 1428\n"
                "месячных рядах M3 (70 тыс. окон),\n"
                "затем дообучение 20 эпох.\n"
                "\n"
                "Улучшение sMAPE:\n"
                "\u2022 PatchTST: 19,0 \u2192 17,0 (\u221210,8%)\n"
                "\u2022 Autoformer: 16,7 \u2192 15,1 (\u22129,6%)\n"
                "\u2022 FEDformer: 18,6 \u2192 17,2 (\u22127,3%)\n"
                "\n"
                "Вывод: трансформерам критически\n"
                "не хватает данных при per-series\n"
                "обучении. Предобучение частично\n"
                "компенсирует этот дефицит.",
                font_size=11)
        elif ph.placeholder_format.idx == 12:
            ph.insert_picture(os.path.join(FIGS, "pretrain_m3_monthly_patchtst_T779.png"))

    # ===== NEW SLIDE: Forecast examples (layout 6 = синий + picture) =====
    slide_forecast = add_slide_from_layout(prs, 6)
    for ph in slide_forecast.placeholders:
        if ph.placeholder_format.idx == 0:
            clear_and_set(ph, "Примеры прогнозов")
        elif ph.placeholder_format.idx == 10:
            clear_and_set(ph, "Визуальный анализ")
        elif ph.placeholder_format.idx == 11:
            clear_and_set(ph,
                "На графике \u2014 прогноз всех 11\n"
                "моделей на квартальном ряде M3.\n"
                "\n"
                "Синий \u2014 обучающая выборка,\n"
                "красный \u2014 тестовая (8 шагов).\n"
                "Пунктир \u2014 прогнозы моделей.\n"
                "\n"
                "Helformer (фиолетовый) и ETS\n"
                "(тёмно-синий) ближе всего\n"
                "к реальным значениям.\n"
                "\n"
                "Autoformer и FEDformer дают\n"
                "наибольшие отклонения.",
                font_size=11)
        elif ph.placeholder_format.idx == 12:
            ph.insert_picture(os.path.join(FIGS, "m3_quarterly_T478.png"))

    # ===== SLIDE 5: План-график — keep as is =====

    # ===== SLIDE 6: Заключение =====
    slide6 = slides[5]
    for s in slide6.shapes:
        if not s.has_text_frame:
            continue
        if "В заключении предполагается" in s.text_frame.text:
            clear_and_set(s,
                "1. Гибридная модель Helformer (фиксированная декомпозиция "
                "Хольта\u2013Уинтерса + трансформерная коррекция остатков) \u2014 "
                "лучшая модель среди всех 11 протестированных на M3 Competition "
                "(sMAPE 10,61%, MASE 0,94, средний ранг 2,00).\n"
                "\n"
                "2. Длина обучающей выборки \u2014 критический фактор: при менее "
                "50 наблюдениях все нейросети уступают ETS и ARIMA; при более "
                "100 \u2014 DLinear и N-BEATS конкурируют и побеждают.\n"
                "\n"
                "3. Обучаемая декомпозиция трансформерного типа (Autoformer, "
                "FEDformer) в per-series режиме показала наихудшие результаты "
                "(sMAPE 22\u201334%). Простая обучаемая декомпозиция (DLinear) "
                "эффективна только на длинных рядах.\n"
                "\n"
                "4. Предобучение трансформеров на корпусе рядов улучшает sMAPE "
                "на 7\u201311%, частично компенсируя дефицит данных при per-series обучении.\n"
                "\n"
                "5. Итеративное прогнозирование выявляет хрупкость обучаемой "
                "декомпозиции: FEDformer деградирует на +6,3 п.п., тогда как "
                "PatchTST (без декомпозиции) \u2014 наиболее устойчивый (+1,8 п.п.).",
                font_size=10)

    # ===== SLIDE 7: Thank you — keep unchanged =====

    prs.save(OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    os.chdir("/Applications/forvscode/BachelorFinal/Hybrid-models-for-TS-processing")
    main()
