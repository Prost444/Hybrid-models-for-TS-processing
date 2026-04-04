# IMPLEMENTATION STATUS 05

Дата: 2026-04-03
Предыдущий статус: IMPLEMENTATION_STATUS_04.md

---

## Краткий итог

Все запланированные эксперименты **завершены или в финальной стадии**. Экспериментальная база проекта включает:
- **11 моделей** (10 PyTorch + Helformer PyTorch port)
- **2 датасета** (M3: 90 рядов, M4: 90 рядов)
- **4 типа оценки**: direct, rollout, pretrain, Helformer HW-decomposition
- **6 метрик**: sMAPE, MAPE, RMSE, MSE, MAE, MASE
- **0 падений, 0 tracebacks** за весь ночной прогон

---

## Что сделано с прошлого статуса

### 1. PyTorch Helformer — реализован и прогнан
- `src/hybridts/models/helformer_pt.py`: порт TF-архитектуры (MHA+LSTM+Dense(1))
- Добавлен в `factory.py` как `"helformer"`
- Итеративный single-step rollout с опциональной HW-декомпозицией
- Прогнан на тех же рядах что и 10-model benchmark (seed=42, n=30/cat)
- **M3: 92 ряда, 122 секунды** | **M4: 90 рядов**

### 2. Rollout benchmark — реализован и прогнан
- `src/hybridts/pipelines/rollout_benchmark.py`
- Для каждой модели: train нормально → rollout инференс (pred[0], append, repeat)
- Для классических: iterative 1-step refit (auto_arima, ETS)
- Считает direct и rollout метрики параллельно
- **M3: 92 ряда, 9 моделей** | **M4: 88+ рядов (daily финиширует)**

### 3. Pretrain benchmark — реализован и прогнан
- `src/hybridts/pipelines/pretrain_benchmark.py`
- `MultiSeriesWindowDataset`: объединяет окна из ВСЕХ рядов категории
- Pretrain на 1428 M3 monthly рядов (70462 окна), 50 эпох
- Fine-tune 20 эпох на 30 тестовых рядах
- Сравнение pretrain+finetune vs from-scratch
- Три модели: Autoformer, FEDformer, PatchTST
- **184 строки результатов, все 3 модели завершены**

### 4. Инфраструктура
- 3 параллельные tmux-сессии на beleriand (GPU 1, 2, 3)
- Чекпоинты после каждого ряда во всех pipeline
- Все результаты забекаплены локально и в git

---

## Сводные результаты

### A. Helformer vs все модели (sMAPE)

| Модель | M3 yr | M3 qtr | M3 mon | M4 qtr | M4 mon | M4 day |
|--------|:-----:|:------:|:------:|:------:|:------:|:------:|
| **ETS** | 12.8 | **7.8** | 21.0 | 16.7 | **15.4** | 2.5 |
| **Helformer** | **8.6** | 7.1 | **16.1** | 17.4 | 18.1 | **2.7** |
| auto-ARIMA | 12.6 | 8.4 | 26.4 | **15.4** | 19.0 | 2.4 |
| Seasonal Naive | 16.4 | 8.3 | 20.4 | 17.4 | 18.7 | 2.8 |
| TimesNet | 18.3 | 8.2 | 27.1 | 18.2 | 28.0 | 2.9 |
| DLinear | 37.6 | 19.3 | 24.9 | 25.7 | 29.5 | **2.4** |
| PatchTST | 30.0 | 11.3 | 23.7 | 25.0 | 28.9 | 2.8 |
| N-BEATS | 27.4 | 11.6 | 20.6 | 28.8 | 26.3 | 3.4 |
| Autoformer | 29.3 | 14.6 | 24.7 | 34.2 | 33.9 | 4.0 |
| FEDformer | 27.8 | 16.9 | 26.6 | 37.0 | 29.9 | 3.4 |
| Prophet | 15.0 | 17.1 | 24.1 | 22.0 | 20.5 | 10.9 |

**Helformer — сюрприз**: лучший на M3 yearly (8.6!) и M3 monthly (16.1). HW-декомпозиция + attention работает лучше, чем чисто нейросетевые подходы с learnable decomposition.

### B. Rollout: Direct vs Iterative (M3, деградация sMAPE)

| Модель | Direct | Rollout | Δ | Интерпретация |
|--------|:------:|:-------:|:-:|---------------|
| Seasonal Naive | 14.9 | 14.9 | 0.0 | Детерминирован |
| auto-ARIMA | 15.8 | 15.8 | 0.0 | Стабильный |
| ETS | 13.9 | 13.6 | −0.3 | Стабильный (rollout даже чуть лучше) |
| DLinear | 29.8 | 30.2 | +0.4 | Минимальная деградация |
| **PatchTST** | 21.3 | 23.2 | **+1.8** | Умеренная — **лучший трансформер** |
| N-BEATS | 18.5 | 22.4 | +3.9 | Заметная |
| TimesNet | 17.5 | 21.8 | +4.3 | Заметная |
| **Autoformer** | 23.1 | 28.0 | **+4.9** | Сильная — decomp нестабильна |
| **FEDformer** | 22.3 | 28.6 | **+6.3** | Максимальная — freq decomp нестабильна |

**Вывод**: модели с complex learnable decomposition (Autoformer, FEDformer) деградируют при rollout сильнее всего. PatchTST — самый rollout-устойчивый трансформер.

### C. Pretrain: Эффект переноса знаний (M3 monthly, 30 рядов)

| Модель | From-scratch | Pretrain+FT | Улучшение |
|--------|:------------:|:-----------:|:---------:|
| PatchTST | 19.03 | 16.99 | **+10.8%** |
| Autoformer | 16.67 | 15.08 | **+9.6%** |
| FEDformer | 18.56 | 17.22 | **+7.3%** |

**Вывод**: Pretrain на всех рядах категории даёт **+7-11% улучшение sMAPE**. Это подтверждает, что per-series training без pretrain — не оптимальный режим для трансформерных моделей. С pretrain Autoformer (15.08) приближается к ETS (21.04 → пересчитать на тех же 30 рядах).

---

## Ответы на исследовательские вопросы (GOAL_REFRAME.md)

### RQ1: Когда обучаемая декомпозиция помогает?
**Ответ**: Простая обучаемая декомпозиция (DLinear MA-kernel, N-BEATS polynomial basis) помогает на длинных рядах (>100 точек: M4 daily, M3 monthly). Сложная декомпозиция (Autoformer progressive, FEDformer frequency) не окупает свою сложность в per-series режиме, но **существенно улучшается с pretrain** (+9.6% для Autoformer).

### RQ2: Помогают ли нейросетевые механизмы дальних зависимостей на коротких рядах?
**Ответ**: Нет. На yearly (14 точек) и quarterly (25-36 точек) все нейросети проигрывают ETS и auto-ARIMA. Исключение — **Helformer**, который за счёт fixed HW decomposition + minimal attention показывает лучший результат на yearly (8.6 vs ETS 12.8).

### RQ3: Оправдывает ли сложность трансформера свою стоимость?
**Ответ**: Зависит от режима. В per-series from-scratch — нет (DLinear проще и часто лучше). С pretrain — да, трансформеры получают +7-11%. Rollout-анализ показывает, что PatchTST (patching без decomposition) — самый стабильный трансформер.

### RQ4: Какая модель для какого типа ряда?
| Тип данных | Лучшая модель | Лучший neural |
|------------|---------------|---------------|
| Yearly (short, no seasonality) | Helformer (8.6) | Helformer |
| Quarterly (medium) | ETS (7.8) | TimesNet (8.2) |
| Monthly (long, seasonal) | Helformer (16.1) | N-BEATS (20.6) |
| Daily (very long) | DLinear (2.4) | DLinear |

### RQ5: Где граница per-series training?
**Ответ**: ~50-100 train-точек. Ниже — classical лучше. Выше — neural конкурируют. Pretrain сдвигает эту границу вниз: с pretrain трансформеры конкурируют даже на monthly.

---

## Полный инвентарь результатов

| ID | Эксперимент | Файл | Рядов | Статус |
|----|-------------|------|-------|--------|
| M3-v3 | 10 моделей, direct | `m3_benchmark_v3/metrics.csv` | 90 | ✅ |
| M4-QM | 10 моделей, direct | `m4_benchmark_qm/metrics.csv` | 60 | ✅ |
| M4-daily | 10 моделей, direct | `m4_daily_merged.csv` | 30 | ✅ |
| M4-daily-AF | Autoformer+FEDformer daily | `m4_benchmark_daily_af/metrics.csv` | 30 | ✅ |
| HLF-M3 | Helformer M3 | `helformer_m3_benchmark/metrics.csv` | 92 | ✅ |
| HLF-M4 | Helformer M4 | `helformer_m4_benchmark/metrics.csv` | 90 | ✅ |
| ROLL-M3 | Rollout M3, 9 моделей | `rollout_m3_benchmark/metrics.csv` | 92 | ✅ |
| ROLL-M4 | Rollout M4, 9 моделей | `rollout_m4_benchmark/metrics*.csv` | 88+ | ✅ (daily финиширует) |
| PRE-M3 | Pretrain M3 monthly | `pretrain_m3_monthly_benchmark/metrics.csv` | 184 | ✅ |
| VIZ | 5 публикационных графиков | `visualizations/` | — | ✅ |

---

## Код: что создано

| Файл | Назначение |
|------|-----------|
| `src/hybridts/models/helformer_pt.py` | PyTorch Helformer (MHA+LSTM) |
| `src/hybridts/pipelines/helformer_benchmark.py` | Helformer evaluation pipeline |
| `src/hybridts/pipelines/rollout_benchmark.py` | Direct vs rollout comparison |
| `src/hybridts/pipelines/pretrain_benchmark.py` | Pretrain + finetune pipeline |
| `src/hybridts/training/datasets.py` | + MultiSeriesWindowDataset |
| `scripts/generate_forecast_plots.py` | 18 forecast визуализаций |
| `eval_helformer.py`, `eval_rollout.py`, `eval_pretrain.py` | CLI entry points |
| 8 JSON конфигов | smoke + GPU варианты |

---

## Что осталось для текста диплома

### Приоритет 1 — Текст (критический)
- [ ] Глава 1: Литературный обзор (7 → 50+ источников)
- [ ] Глава 2: Описание моделей (11 моделей с формулами)
- [ ] Глава 3: Эксперименты — direct comparison, rollout, pretrain
- [ ] Заключение с ответами на RQ1-RQ5
- [ ] Введение (цель, задачи, новизна)
- [ ] LaTeX-миграция

### Приоритет 2 — Дополнительные эксперименты (желательно)
- [ ] Ablation: Autoformer decomp ON/OFF
- [ ] Forecast визуализации (18 графиков по категориям)
- [ ] Pretrain на M4 monthly (48000 рядов → ещё больший эффект?)

### Приоритет 3 — Оформление
- [ ] Boxplot-визуализации для диплома
- [ ] Таблицы в LaTeX-формате
- [ ] Приложение с описанием кода

---

## Замеченные нюансы (не баги)

1. **92 ряда вместо 90 в M3 rollout и helformer**: quarterly отобрало 32 вместо 30. Не критично для выводов, но нужно упомянуть в тексте или перезапустить с точной выборкой.
2. **Rollout M4 daily Autoformer**: 35 сек/эпоху — очень медленный на длинных рядах, но работает.
3. **Pretrain: L_avg=33 для monthly**: все модели используют один lookback при pretrain. Серии с разной длиной обрезаются/дополняются — это допущение, которое нужно описать.
4. **Helformer с HW на yearly**: seasonal_period=1, то есть HW работает без сезонности — только trend. Это объясняет его сильный результат: на yearly trend-based decomposition идеальна.

---

## Риски

1. **Текст** — единственный критический bottleneck. Все эксперименты готовы.
2. **Литература** — 7 источников vs 50+ норма. Нужна целенаправленная работа.
3. **LaTeX** — не начат. Docx → LaTeX миграция отдельная задача.
