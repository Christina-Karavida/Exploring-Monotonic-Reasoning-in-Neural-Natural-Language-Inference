# MoRe Dataset Creation Pipeline

This directory contains the code used to construct the **Monotonicity Reasoning (MoRe) dataset** and generate the experimental splits used in this thesis.

The pipeline constructs the taxonomy, generates the natural language inference examples, and creates the experimental splits used to evaluate models under different generalization conditions.

> **Note:** The generated datasets are not included in this repository due to their size. The data can be reproduced by running the provided generation code.

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

The `data/` directory is used as the output directory for the generated datasets and experimental splits.

---

# 1. Taxonomy Construction

The taxonomy defines the hypernym–hyponym relations used to construct the MoRe examples.

### `taxonomy.py`

Contains the collection of hypernym–hyponym pairs used for dataset construction, organized into semantic categories.

### `taxonomy_helpers.py`

Contains helper functions for:

* generating and processing hypernym–hyponym pairs;
* inspecting category distributions;
* visualizing the taxonomy; and
* calculating taxonomy statistics.

### `taxonomy_stats.ipynb`

A notebook for inspecting and analysing the taxonomy used in the dataset.

---

# 2. Dataset Generation

The dataset generation pipeline combines the taxonomy with sentence templates and logical rules to construct the MoRe examples.

### `sentence_banks.py`

Contains the sentence templates used to generate the dataset. The templates are organized according to the logical operators and patterns used in the MoRe dataset.

### `pluralization_helper.py`

Contains helper functions for handling the pluralization of lexical items during dataset generation.

### `dataset_generation.py`

Contains the main dataset generation functions.

The complete MoRe dataset, including the individual rule-specific datasets, is generated from this pipeline and saved under:

```text
data/more_dataset/
```

---

# 3. Experimental Splits

The generated MoRe dataset is further divided into experimental splits designed to test different forms of generalization.

### `experimental_splits.py`

Contains the functions used to construct the experimental splits.

The pipeline generates:

* **Hyponym Generalization**
* **Hyponym–Hypernym Generalization**
* **Hyponym Generalization – Diagnostic**
* **Flipped Pattern**

The resulting datasets are saved under:

```text
data/more_splits/
```

### `Dataset_Generation.ipynb`

Provides an executable walkthrough of the dataset generation process and the construction of the experimental splits.

---

# 4. Reproducing the Dataset

To reproduce the datasets, run the generation pipeline provided in `Dataset_Generation.ipynb`.

The generated files will be written to the `data/` directory. In particular:

```text
data/
├── more_dataset/
└── more_splits/
```

The notebook also performs checks on the generated splits to verify properties such as rule coverage and overlap between the relevant lexical items.

---

## Output

Running the pipeline produces:

1. The complete MoRe dataset and the individual rule-specific datasets.
2. The four experimental splits used in the thesis.
3. The diagnostic version of the Hyponym Generalization split.

The generated files are intentionally excluded from the repository and can be recreated using the code provided here.
