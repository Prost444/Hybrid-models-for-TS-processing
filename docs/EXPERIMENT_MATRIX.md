# EXPERIMENT MATRIX — Матрица экспериментов

Дата создания: 2026-04-01
Последнее обновление: 2026-04-02 (после M3 v3 + M4 full)

---

## Статус выполненных экспериментов

| ID | Эксперимент | Статус | Результат |
|----|-------------|--------|-----------|
| M3-smoke | 2 ряда/кат × 2 кат × 8 моделей | ✅ done | Все модели работают, pipeline OK |
| M3-v2 | 30 рядов/кат × 3 кат × 8 моделей × 80 эпох | ✅ done (промежуточный) | ETS лучший, Autoformer 12 failures, DLinear плох на yearly |
| M3-smoke-v3 | 2 ряда/кат × 2 кат × 10 моделей × 30 эпох | ✅ done | Все 10 моделей OK, ETS лучший |
| M3-v3 | 30 рядов/кат × 3 кат × 10 моделей × 80 эпох | ✅ done (GPU, beleriand) | ETS лучший (13.89), TimesNet лучший neural (17.87), N-BEATS лучший на monthly (20.58), 0 crashes |
| M4-smoke | 2 ряда/кат × 2 кат × 10 моделей × 30 эпох | ✅ done | Все 10 моделей OK, pipeline валидирован |
| M4-QM | 30 рядов/кат × 2 кат × 10 моделей × 80 эпох | ✅ done (GPU 1, tmux) | ETS лучший (16.06), auto_arima 2-й (17.19), TimesNet лучший neural (23.10) |
| M4-daily | 30 рядов × 1 кат × 8 моделей × 50 эпох | ✅ done (GPU 2, tmux) | DLinear лучший (2.43!), нейросети конкурируют с классикой |
| Ablation | — | ⬜ TODO | — |

---

## 1. Модели

### Группа A — Классические baseline
| ID | Модель | Тип | Decomposition | Уже реализована? |
|----|--------|-----|---------------|-----------------|
| A1 | Seasonal Naive | Статистический | — | Частично (в m3.py) |
| A2 | ARIMA (auto) | Статистический | — | Да |
| A3 | ETS (Holt-Winters) | Статистический | Фиксированная (HW) | Да |
| A4 | Prophet | Статистический | Фиксированная (Fourier) | Да |

### Группа B — Нейросетевые с learnable decomposition
| ID | Модель | Тип | Decomposition | Уже реализована? |
|----|--------|-----|---------------|-----------------|
| B1 | DLinear | Linear + decomp | Learnable (MA kernel) | Да |
| B2 | N-BEATS (interpretable) | MLP + basis | Learnable (poly + harmonic) | Да |
| B3 | Autoformer | Transformer + decomp | Learnable (progressive) | Да |
| B4 | FEDformer | Transformer + freq decomp | Learnable (frequency) | Да |

### Группа C — Нейросетевые без explicit decomposition
| ID | Модель | Тип | Mechanism | Уже реализована? |
|----|--------|-----|-----------|-----------------|
| C1 | PatchTST | Transformer (patched) | Patched attention | Да |
| C2 | TimesNet | CNN (2D periodic) | Multi-period 2D conv | Да |

### Группа D — Гибриды (decomposition + model per component)
| ID | Модель | Тип | Decomposition | Уже реализована? |
|----|--------|-----|---------------|-----------------|
| D1 | MODWT + TimesNet + ETS | Wavelet hybrid | MODWT (фиксированная) | Да (VWHybridMixed) |
| D2 | STL + ARIMA (residual) | Stat hybrid | STL (фиксированная) | Нет → реализовать |

### Группа E — Вспомогательные (не обязательные)
| ID | Модель | Тип | Уже реализована? |
|----|--------|-----|-----------------|
| E1 | Helformer | TF/Keras + HW | Да (legacy) |
| E2 | iTransformer | Inverted Transformer | Нет → TSLib |

---

## 2. Датасеты

| ID | Датасет | Частоты | Рядов на частоту | Горизонт | Данные в проекте? |
|----|---------|---------|-----------------|----------|-------------------|
| D-S | Синтетические | — | 6 профилей × 2 | H=24 | Генерируются кодом |
| D-M3 | M3 Competition | yearly, quarterly, monthly | 30 на частоту | 6/8/18 | Да |
| D-M4 | M4 Competition | quarterly, monthly, daily | 30 на частоту | 8/18/14 | Да (raw) |

### Примечания:
- M3: yearly (H=6, P=1), quarterly (H=8, P=4), monthly (H=18, P=12)
- M4: quarterly (H=8, P=4), monthly (H=18, P=12), daily (H=14, P=7)
- M5: **исключён** — hierarchical dataset, не соответствует point forecasting + decomposition

---

## 3. Метрики

| ID | Метрика | Тип | Стандарт | Реализована? |
|----|---------|-----|----------|-------------|
| sMAPE | Symmetric MAPE | Процентная | M3 Competition | Да |
| MAPE | Mean Abs. Percentage Error | Процентная | Общая | Да |
| RMSE | Root Mean Squared Error | Абсолютная | Общая | Да |
| MSE | Mean Squared Error | Абсолютная | Общая | Да |
| MAE | Mean Absolute Error | Абсолютная | Общая | Да |
| MASE | Mean Abs. Scaled Error | Масштабированная | M4 Competition | Да |

---

## 4. Матрица экспериментов

### Обязательный минимум (MUST)

| Эксперимент | Модели | Датасет | Метрики | Цель |
|------------|--------|---------|---------|------|
| E1. Baseline comparison | A1–A4 + B1 + B2 | M3 (all freq) | Все 6 | Установить baseline |
| E2. Decomposition Transformers | B3 + B4 + C1 | M3 (all freq) | Все 6 | Сравнить learnable decomp |
| E3. Full M3 benchmark | A2–A4 + B1–B4 + C1–C2 | M3 (all freq) | Все 6 | Полная сравнительная таблица |
| E4. M4 benchmark | A2–A4 + B1–B4 + C1–C2 | M4 (3 freq) | Все 6 | Второй датасет |
| E5. Synthetic control | A2–A4 + B1–B4 | Synth (6 prof) | Все 6 | Контрольные условия |

### Важные дополнения (SHOULD)

| Эксперимент | Модели | Датасет | Цель |
|------------|--------|---------|------|
| E6. Ablation: decomp ON/OFF | B3 (±decomp), B4 (±decomp) | M3 quarterly | Вклад декомпозиции |
| E7. Ablation: learnable vs fixed | B1 vs A3, B3 vs (STL+Transformer) | M3 monthly | Learnable vs fixed |
| E8. Hybrid comparison | D1 + D2 vs B3 + B4 | M3 (all freq) | Гибрид vs end-to-end |
| E9. Per-type analysis | Лучшие 4 модели | M3 + M4 | Какая модель для какого типа рядов |

### Опциональные (NICE)

| Эксперимент | Модели | Датасет | Цель |
|------------|--------|---------|------|
| E10. Helformer comparison | E1 vs B3 | M3 | Fixed HW + attention vs learnable + auto-corr |
| E11. iTransformer | E2 vs B3 + C1 | M3 | Modern Transformer baseline |
| E12. Sensitivity | B3 (разные lookback) | M3 monthly | Чувствительность к гиперпараметрам |

---

## 5. Порядок запуска

```
ФАЗА 0 — Инфраструктура (0.5–1 день)
├── Добавить MAE, MASE в src/hybridts/data/m3.py и m4.py
├── Реализовать DLinear (~30 строк)
├── Написать TSLib eval-wrapper для Autoformer/FEDformer/PatchTST
├── Создать новые конфиги для M3 и M4
└── Расчистить репозиторий (REPO_CLEANUP_PLAN)

ФАЗА 1 — Быстрые эксперименты (1–2 дня)
├── E1. Baseline comparison на M3 (уже частично есть)
├── E5. Синтетика с расширенным набором моделей
└── Валидация: DLinear работает, TSLib wrapper работает

ФАЗА 2 — Основные эксперименты (3–5 дней, на сервере)
├── E3. Full M3 benchmark (все модели × все частоты)
├── E4. M4 benchmark (3 частоты × 30 рядов)
├── E2. Decomposition Transformers отдельно
└── E6. Ablation: decomp ON/OFF

ФАЗА 3 — Дополнительные (1–2 дня)
├── E7. Learnable vs fixed decomposition
├── E8. Hybrid comparison
├── E9. Per-type analysis (аналитика по готовым результатам)
└── E10–E12 (если хватит времени)
```

---

## 6. Критерии годности модели для диплома

Модель считается годной, если:
1. **Запускается без ошибок** на всех тестовых датасетах
2. **Даёт непустые прогнозы** (без NaN, Inf, вырожденных случаев)
3. **Результаты воспроизводимы** при фиксированном seed
4. **Метрики вычислены корректно** по всем 6 показателям
5. **Есть графики** для репрезентативных рядов
6. **Можно описать в тексте**: архитектура, гиперпараметры, процедура обучения, результаты

Модель **не годна**, если:
- Результаты подозрительно хорошие (проверить на утечку)
- Воспроизводимость не обеспечена
- Нет объяснения архитектуры и параметров

---

## 7. Ожидаемый объём результатов

| Категория | Количество |
|----------|------------|
| Моделей в сравнении | 8–10 |
| Датасетов | 3 (synth + M3 + M4) |
| Частот M3 | 3 (yearly, quarterly, monthly) |
| Частот M4 | 3 (quarterly, monthly, daily) |
| Рядов всего | ~6 (synth) + ~90 (M3) + ~90 (M4) = ~186 |
| Метрик | 6 |
| Таблиц в работе | 8–12 (сводные по частотам + ablation + per-type) |
| Графиков в работе | 10–15 (representative cases + boxplots + decomposition) |
