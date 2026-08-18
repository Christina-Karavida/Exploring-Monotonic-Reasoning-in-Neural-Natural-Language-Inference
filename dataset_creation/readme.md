# MoRe Dataset Creation Pipeline

This directory contains the code necessary to generate Monotonicity Reasoning (MoRe) dataset introduced in this thesis. 

---

# Directory Structure

```text
dataset_creation/
│
├── data/
│   ├── more_dataset/
│   │   ├── all_rules_of_more.csv
│   │   ├── all_rules_of_more.jsonl
│   │   ├── R1.csv ... R12.csv
│   │   
│   │
│   └── more_splits/
│       ├── split1/
│       ├── split2/
│       └── split3/
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
    ├── sentence_banks.py

```

---

# 1. Taxonomy Construction

## taxonomy.py

Contains the collection of hyperym/hyponym pairs used for the creation of the dataset organized taxonomically.

---

## taxonomy_helpers.py

Includes helper functions for:

- generating hypernym–hyponym pairs
- inspecting category distributions
- visualizing the taxonomy 
- calculating taxonomy statistics


## taxonomy_stats.ipynb

Notebook used to inspect the taxonomy


# 2. Dataset Generation

## sentence_banks.py

Contains the collection of sentence templates that we used to generate MoRe's examples.

This collection of templates is grouped by operator:

```python
ALL
SOME
NO
NO_OPERATOR
```

Example:

```python
"All {X} crossed the field."
```
---

## pluralization_helper.py

The script handles regular and irregular plurals.

---

## dataset_generation.py

This is the main dataset generation script, that contains all the functions necessary to generate the 12 rules of MoRe.

---

## Dataset_Generation.ipynb

This notebook generates the dataset.

---

