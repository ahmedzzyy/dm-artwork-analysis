# DM Project

> Discovering artistic movements through machine learning on the Museum of Modern Art collection.

**Ahmed Sahigara | MIT Manipal**

## Table of Contents

1. [Features](#features)
2. [Setup (uv)](#setup-uv)
3. [Notebooks](#notebooks)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)

## Features

- **Automated Artwork Classification**
  Predicts artwork category (based on attributes such as style, movement,
  period) using supervised learning.

- **Artwork Clustering**
  Groups similar artworks to reveal hidden relationships in the dataset.

- **Large-Scale Data Processing**
  Handles 160k+ artwork records efficiently.

- **Data Preprocessing Pipeline**
  Missing value handling, encoding, feature selection, and normalization.

- **Model Evaluation**
  Performance measured using standard ML metrics.

- **Insight Generation**
  Identifies important metadata features influencing artistic trends.

---

## Setup (uv)

1. To install uv, visit [Installation | uv - Astral Software](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 2. Create env and install deps
uv sync
```

3. Get the data

- Download from: https://github.com/MuseumofModernArt/collection
- Place Artworks.csv and Artists.csv in the data/ folder

```bash
# 4. Run the full pipeline
uv run python main.py

# Or just EDA (faster sanity check)
uv run python main.py --eda-only

# Skip the slow SVM if you're in a hurry
uv run python main.py --skip-svm
```

---

## Notebooks

```bash
uv run jupyter notebook notebooks/
```

- `midterm_progress.ipynb` — EDA + baseline model (Phase 1)
- `final_showcase.ipynb` — Full pipeline with all models (Phase 2)

---

## Project Structure

```txt
├── main.py                    ← run everything from here
├── pyproject.toml             ← uv dependency management
├── src /
│   ├── config.py              ← all paths and hyperparameters
│   ├── data_loader.py         ← load, merge, clean CSVs
│   ├── features.py            ← feature engineering
│   ├── eda.py                 ← exploratory plots
│   ├── classifier.py          ← LR + RF + SVM
│   ├── clusterer.py           ← KMeans + DBSCAN + t-SNE
│   ├── visualizer.py          ← shared plot utilities
│   └── utils.py               ← save/load, timer decorator
├── notebooks/
│   ├── midterm_progress.ipynb
│   └── final_showcase.ipynb
├── data/                      ← put Artworks.csv + Artists.csv here
└── outputs/
    ├── figures/               ← classification + clustering plots
    ├── eda/                   ← exploratory plots
    └── models/                ← saved .pkl models
```

---

## Dataset

Museum of Modern Art (MoMA) Collection:

- https://github.com/MuseumofModernArt/collection
- `Artworks.csv` (~160K rows) and `Artists.csv` (~15K rows)
- Publicly available for non-commercial research use
