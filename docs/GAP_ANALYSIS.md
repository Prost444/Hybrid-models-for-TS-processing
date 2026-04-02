# GAP ANALYSIS v4 — Требования vs фактическое состояние

Дата обновления: 2026-04-02 (после M3 v3 + M4 full)
Изменения v4: M3 v3 завершён (90 рядов, 10 моделей, GPU). M4 завершён (90 рядов, 3 частоты). Все модели реализованы. Чекпоинты добавлены.

---

## Сводная таблица

| # | Требование | Подтверждающий файл / артефакт | Статус | Комментарий | Что нужно для закрытия |
|---|-----------|-------------------------------|--------|-------------|----------------------|
| 1 | **Синтетические ряды** | `src/outputs/synth_eval/metrics.csv` | **partial** | 6 профилей есть, но только 5 старых моделей. Нет DLinear, Autoformer, FEDformer, PatchTST. | Расширить набор моделей. Перезапустить с новыми моделями и 6 метриками. |
| 2 | **ARIMA (auto)** | `src/hybridts/models/classic.py` | **done** | Реализован, результаты есть. | Закрыто. |
| 3 | **ETS** | `src/hybridts/models/classic.py` | **done** | Реализован, результаты есть. | Закрыто. |
| 4 | **Prophet** | `src/hybridts/models/classic.py` | **done** | Работает на M3 и M4 (все частоты). | Закрыто. |
| 5 | **DLinear** | `src/hybridts/models/dlinear.py` | **done** | Реализован. MA decomp + 2 linear. Лучший на M4 daily (sMAPE=2.43). | Закрыто. |
| 6 | **Autoformer** | `src/hybridts/models/autoformer.py` | **done** | Реализован. Topk bug исправлен. 0 crashes на M3 v3. Слабый на M4 (34+ sMAPE). | Закрыто. |
| 7 | **FEDformer** | `src/hybridts/models/fedformer.py` | **done** | Реализован. Frequency attention + series decomp. Слабый на M4. | Закрыто. |
| 8 | **PatchTST** | `src/hybridts/models/patchtst.py` | **done** | Реализован. Patch embedding + TransformerEncoder. Конкурентный на M3 quarterly. | Закрыто. |
| 9 | **N-BEATS** | `src/hybridts/models/nbeats.py` | **done** | Реализован (interpretable: poly + harmonic basis). Соответствует теме: learnable basis decomposition. | Закрыто. Нужно запустить в полном benchmark. |
| 10 | **TimesNet** | `src/hybridts/models/timesnet.py` | **done** | Реализован. Multi-periodicity via FFT + 2D conv. | Закрыто. Нужно запустить в полном benchmark. |
| 11 | **Helformer** | `src/hybridts/models/helformer.py` | **demoted** | Реализован. Но: утечка данных в статье, результаты ненадёжны. Остаётся как вспомогательный вариант (fixed HW + attention). | Не центральная модель. Использовать для контраста fixed vs learnable decomposition. |
| 12 | **Комбинированные модели** | Код: `modwt_hybrid.py`. Результатов нет. | **partial** | VWHybridMixed готов в коде, не запускался. Нужна хотя бы одна гибридная схема с результатами. | Запустить VWHybridMixed или реализовать STL + ARIMA hybrid. Описать в тексте. |
| 13 | **Learnable decomposition (тема)** | Результаты M3+M4 | **done** | 4 модели с learnable decomp: DLinear, Autoformer, FEDformer, N-BEATS. Сравнение с ETS/ARIMA/Prophet (fixed). | Закрыто. Ablation ON/OFF — отдельная задача. |
| 14 | **M3** | `src/outputs/m3_benchmark_v3/metrics.csv` — 90 рядов | **done** | 90 рядов × 10 моделей × 6 метрик × 80 эпох, GPU. | Закрыто. |
| 15 | **M4** | `src/outputs/m4_benchmark_qm/` + `m4_benchmark_daily/` — 90 рядов | **done** | 60 qm (10 моделей) + 30 daily (8 моделей), GPU. | Закрыто. |
| 16 | **M5** | — | **excluded** | Решено исключить. M5 — hierarchical dataset, не соответствует point forecasting + decomposition. Обосновать в тексте. | Написать обоснование исключения (1 абзац в тексте). |
| 17 | **Метрики** | `src/hybridts/data/m3.py` | **done** | 6 метрик: sMAPE, MAPE, RMSE, MSE, MAE, MASE. | Закрыто. |
| 18 | **Литература** | 7 источников в тексте | **broken** | Катастрофически мало. Нет статей по ключевым моделям. | **КРИТИЧНО.** Довести до 50+. Приоритет: decomposition models, Transformers for TS, M-competitions. |
| 19 | **Оформление (LaTeX)** | Нет .tex/.bib | **missing** | Работа в DOCX. | Мигрировать в LaTeX. |
| 20 | **Объём текста** | ~26 стр содержания + 54 стр кода | **broken** | Непропорционально. | Расширить содержание до 40+ стр, код в приложении ≤10 стр. |
| 21 | **Чистота кодовой базы** | Дубли, шимы, чужие файлы | **broken** | Дублирование моделей, шимы обратной совместимости, артефакты Лыкова, BTC-эксперименты. | Расчистить по REPO_CLEANUP_PLAN. |
| 22 | **Ablation study** | — | **missing** | Нет ablation: decomposition ON/OFF, learnable vs fixed. Необходимо для темы. | Запустить ablation на M3 (как минимум Autoformer ±decomp). |
| 23 | **Воспроизводимость** | eval_*.py + configs + requirements-core.txt | **done** | PyTorch-only pipeline, seed=42, JSON configs, checkpointing. TF только для legacy Helformer. | Закрыто. |

---

## Сводка по статусам

| Статус | Кол-во | Элементы |
|--------|--------|----------|
| **done** | 14 | ARIMA, ETS, Prophet, DLinear, Autoformer, FEDformer, PatchTST, N-BEATS, TimesNet, learnable decomp, M3, M4, метрики, воспроизводимость |
| **partial** | 2 | Синтетика, комбинированные модели |
| **missing** | 2 | LaTeX, ablation |
| **broken** | 3 | Литература, объём текста, чистота кода |
| **excluded** | 1 | M5 |
| **demoted** | 1 | Helformer |

---

## Обновление v3: что изменилось после M3 v2

| # | Было | Стало |
|---|------|-------|
| 14 | M3 — partial (старые модели) | M3 — **done (промежуточный)**. 90 рядов, 8 моделей, 6 метрик. Но Autoformer с багом. |
| 5 | DLinear — missing | DLinear — **done** (реализован, в pipeline, результаты есть). Но проблема kernel_size на yearly. |
| 6 | Autoformer — missing | Autoformer — **partial** (реализован, но баг topk, 12/90 failures). |
| 17 | Метрики — partial | Метрики — **done**. Все 6: sMAPE, MAPE, RMSE, MSE, MAE, MASE. |
| NEW | — | **Валидация кодовой базы** — **missing**. Нужна перед интерпретацией результатов. |

## Топ-7 самых опасных провалов (обновлённый v3)

1. **Литература: 7 источников вместо 50+** — дисквалифицирует работу
2. **Autoformer баг (topk crash)** — 12/90 failures, результаты неполные, нужен fix
3. **DLinear kernel_size на yearly** — MASE=8.28, вероятно некорректная конфигурация
4. **M4: 3 ряда одной частоты** — нет расширенного M4
5. **Объём: 26 стр содержания + 54 стр кода** — текст слишком тонкий
6. **Нет ablation study** — не показан вклад декомпозиции
7. **Нет FEDformer / PatchTST** — формальное требование задания (FEDformer) не закрыто
