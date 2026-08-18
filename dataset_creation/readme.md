# How to Generate the MoRe Dataset

This directory contains the code used to construct the **Monotonicity Reasoning (MoRe) dataset** as well as generate the experimental splits used for the experiments of this thesis.

> **Note:** The dataset and the experimental splits are not included in this repository due to their size. To generate MoRe and the experimental splits, run the designated cells in the `Dataset_Generation.ipynb` notebook. This will automatically generate all the data used in this work and save it in the `data/` folder.

---

## Directory Structure

```text
dataset_creation/
│
├── data/
│   └── .gitkeep
│
├── taxonomy/
│   ├── taxonomy.py
│   ├── taxonomy_helpers.py
│   └── taxonomy_stats.ipynb
│
└── experimental_splits/
    ├── Dataset_Generation.ipynb
    ├── dataset_generation.py
    ├── experimental_splits.py
    ├── pluralization_helper.py
    └── sentence_banks.py
```

---

# 1. Taxonomy Construction

The taxonomy defines the hypernym–hyponym relations used to construct the MoRe examples.

### `taxonomy.py`

Contains the collection of hypernym–hyponym pairs used to create MoRe, organized in a taxonomy.

### `taxonomy_helpers.py`

Contains helper functions for:

* generating hypernym–hyponym pairs;
* visualizing the taxonomy; and
* calculating taxonomy statistics.

### `taxonomy_stats.ipynb`

A notebook for inspecting and analysing the taxonomy used in the dataset.

---


# 2. Dataset Generation

### `sentence_banks.py`

Contains the sentence templates used to generate the dataset.

### `pluralization_helper.py`

Contains helper functions for handling the pluralization of hypernym pairs in the dataset generation phase.

### `dataset_generation.py`

Contains all the necessary functions to generate the dataset.

### `Dataset_Generation.ipynb`

This notebook can be used to reproduce all the data used in this thesis. Running the designated cells generates the complete MoRe dataset, as well as all experimental splits used in our experiments, including the Hyponym Generalization, Hyponym–Hypernym Generalization, Diagnostic, and Flipped Pattern splits. The generated data will be automatically  saved in the folder `data/`.

---

# 3. Experimental Splits

### `experimental_splits.py`

Contains the functions used to construct MoRe's splits, namely:
* **Hyponym Generalization**
* **Hyponym–Hypernym Generalization**
* **Hyponym Generalization – Diagnostic**
* **Flipped Pattern**

The resulting datasets are saved under:

```text
data/more_splits/
```



# 4. Reproducing the Dataset

To reproduce the datasets, run the cells of the `Dataset_Generation.ipynb`.

The generated files will be written to the `data/` directory. In particular:

```text
data/
├── more_dataset/
└── more_splits/
```
---

