[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project - Прогноз победителя карты в CS2

**Студент:** Везелев Михаил Вячеславович
**Задача:** бинарная классификация - по карте профессионального матча CS2 предсказать, победит ли `team1`.


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** бинарная классификация. Таргет — `team1_win ∈ {0,1}`, победит ли команда `team1` на конкретной карте матча CS2.

**Датасет:** дамп профессиональных матчей CS2 (`cs2_all_tiers_games.csv`, 19 031 × 99). После фильтрации покарточных строк и очистки — 9 731 наблюдение за 2023-10 … 2026-04.
**Источник данных:** [Kaggle: Counter-Strike Pro Matches](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches/data)

**Целевая метрика:** **ROC-AUC** (основная), LogLoss (tie-breaker), Accuracy и F1-macro (вторичные). AUC выбран за устойчивость к лёгкому дисбалансу классов (53/47).


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# Окружение
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt

# Линтер
make lint

# Docker
docker-compose up --build

# Пайплайн
В процессе
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты

| Модель | ROC-AUC (val) | ROC-AUC (test) | Accuracy (test) | LogLoss (test) | Примечание |
|---|---|---|---|---|---|
| Baseline (LogReg, raw) | 0.508 | 0.500 | 0.541 | 0.690 | без feature engineering |
| LogReg (FE) | 0.640 | — | — | — | C=0.01 |
| KNN (FE) | 0.731 | — | — | — | n=31, distance |
| Decision Tree (FE) | 0.736 | — | — | — | depth=8 |
| Random Forest (FE) | 0.761 | — | — | — | depth=8, n=200 |
| Gradient Boosting (FE)** | 0.766 | 0.706 | 0.652 | 0.615 | depth=4, lr=0.05, n=100 |
| Voting Ensemble (FE) | 0.753 | — | — | — | soft voting: LogReg + RF + GBDT |

Прирост лучшей модели над baseline: +0.21 ROC-AUC на test.


## Отчёт

Финальный отчёт будет в: [`report/report.md`](report/report.md)
