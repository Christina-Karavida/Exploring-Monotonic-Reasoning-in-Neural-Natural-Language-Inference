# ============================================================
# gpt_classification/dataset_config.py
# Dual-compatible configuration for MoRe and SNLI datasets
# ============================================================

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any

# External loader for SNLI
# from data.data_loader_unified import get_snli_data


# -------------------------------------------------------------
# Prompt Templates (3-way NLI)
# -------------------------------------------------------------

SNLI_PROMPT = '''
Perform NLI on the following sentences:
Sentence 1: {premise}
Sentence 2: {hypothesis}

OPTIONS: A: entailment  B: neutral  C: contradiction

LABEL: '''

MORE_PROMPT = '''
Perform NLI on the following sentences:
Sentence 1: {premise}
Sentence 2: {hypothesis}

OPTIONS: A: entailment  B: neutral  C: contradiction

LABEL: '''

def get_snli_data(folder_path="./data/"):
    """
    Loads SNLI data from data/snli_1.0/ without downloading.
    Returns train, test, val splits and label mapping.
    """
    snli_folder = os.path.join(folder_path, "snli_1.0")

    train_path = os.path.join(snli_folder, "snli_1.0_train.jsonl")
    test_path = os.path.join(snli_folder, "snli_1.0_test.jsonl")
    val_path = os.path.join(snli_folder, "snli_1.0_dev.jsonl")

    if not all(os.path.exists(p) for p in [train_path, test_path, val_path]):
        raise FileNotFoundError(f"SNLI files not found in {snli_folder}")

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    val = load_jsonl(val_path)

    label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    print(f"SNLI loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, test, val, label_mapping
    
# -------------------------------------------------------------
# JSONL Loader for MoRe (train/val/test)  
# -------------------------------------------------------------
def get_more_data(folder_path="./data/"):  #can be used to basically load the split1 only for 1. the raw gpt 2. the snli model 
    """Load MoRe dataset with train/val/test splits from JSONL files."""
    more_folder = os.path.join(folder_path, "my_data")

    train_path = os.path.join(more_folder, "split1_train.jsonl")
    val_path = os.path.join(more_folder, "val.jsonl")
    test_path = os.path.join(more_folder, "split1_test.jsonl")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    test = load_jsonl(test_path)

    label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

    print(f"MoRe dataset loaded successfully:")
    print(f"  Train = {len(train)} | Val = {len(val)} | Test = {len(test)}")
    return train, test, val, label_mapping



def load_rule_file(path):
    """
    Load a JSONL dataset and automatically assign example['rule']
    based on the filename (R1.jsonl → rule='R1').
    """
    filename = os.path.basename(path)
    rule_id = os.path.splitext(filename)[0]   # "R1"

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["rule"] = rule_id
            data.append(ex)

    return data


def load_all_rules(folder_path):
    """
    Load every R*.jsonl file in a folder and automatically tag example['rule'].
    Returns a dict: {"R1": [...], "R2": [...], ...}
    """
    rule_data = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".jsonl"):
            continue
        if not filename.startswith("R"):
            continue

        rule_id = os.path.splitext(filename)[0]  # "R1"
        full_path = os.path.join(folder_path, filename)
        rule_data[rule_id] = load_rule_file(full_path)

    return rule_data


def select_rules(rule_dict, selected_rules):
    """Flatten selected rules into a single list."""
    data = []
    for rule in selected_rules:
        if rule in rule_dict:
            data.extend(rule_dict[rule])
        else:
            print(f"Rule {rule} not found in folder.")
    return data




# def get_more_split(split_name: str, folder_path="./data/"):   # to be able to train on the three splits
#     """
#     Load MoRe split1, split2, or split3 using the same structure as get_more_data().
#     """
#     more_folder = os.path.join(folder_path, "my_data")

#     train_path = os.path.join(more_folder, f"{split_name}_train.jsonl")
#     test_path  = os.path.join(more_folder, f"{split_name}_test.jsonl")

#     for p in [train_path, test_path]:
#         if not os.path.exists(p):
#             raise FileNotFoundError(p)

#     def load_jsonl(path):
#         with open(path, "r", encoding="utf-8") as f:
#             return [json.loads(line.strip()) for line in f if line.strip()]

#     train = load_jsonl(train_path)
#     test  = load_jsonl(test_path)
#     label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

#     print(f"[MoRe-{split_name}] Train={len(train)} , Test={len(test)}")

#     return train, test, label_mapping



def get_more_split(split_name: str, folder_path="./data/", train_rules=None, test_rules=None):
    """
    Dual-mode loader:
    1) Default mode if no rule-selection is given:
         - Loads any file matching:
             split_name*_train.jsonl
             split_name*_test.jsonl
    2) Rule-selection mode (NEW):
         - Loads R*.jsonl inside my_data/split_name/ and selects subsets
    """

    base_folder = os.path.join(folder_path, "my_data")

    # ---------------------------------------
    # MODE 1 — OLD BEHAVIOR (default)
    # ---------------------------------------
    if train_rules is None and test_rules is None:

        # Flexible finder for train/test files
        def find_file(base, split, mode):
            for filename in os.listdir(base):
                if filename.startswith(split) and filename.endswith(f"_{mode}.jsonl"):
                    return os.path.join(base, filename)
            return None

        train_path = find_file(base_folder, split_name, "train")
        test_path  = find_file(base_folder, split_name, "test")

        if not train_path or not test_path:
            raise FileNotFoundError(
                f"Could not find train/test JSONL for split '{split_name}'.\n"
                f"Searched inside: {base_folder}"
            )

        # loader
        def load_jsonl(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line.strip()) for line in f if line.strip()]

        train = load_jsonl(train_path)
        test  = load_jsonl(test_path)

        label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

        print(f"[OLD MODE: {split_name}] → Loaded:")
        print(f"  Train file: {os.path.basename(train_path)} ({len(train)} examples)")
        print(f"  Test  file: {os.path.basename(test_path)} ({len(test)} examples)")

        return train, test, label_mapping

    # ---------------------------------------
    # MODE 2 — NEW RULE-BASED BEHAVIOR
    # ---------------------------------------
    rule_folder = os.path.join(base_folder, split_name)

    if not os.path.isdir(rule_folder):
        raise FileNotFoundError(
            f"Expected folder for rule files not found: {rule_folder}\n"
            f"Create this structure:\n  my_data/{split_name}/R1.jsonl, ..., R12.jsonl"
        )

    # Load all 12 rule files
    rule_dict = load_all_rules(rule_folder)
    all_rules = sorted(rule_dict.keys())

    # Defaults if user does not specify test/train lists
    if train_rules is None:
        train_rules = all_rules
    if test_rules is None:
        test_rules = all_rules

    # Select subsets
    train = select_rules(rule_dict, train_rules)
    test  = select_rules(rule_dict, test_rules)

    label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

    print(f"[NEW RULE MODE: MoRe-{split_name}]")
    print(f"  Train rules = {train_rules} → {len(train)} examples")
    print(f"  Test rules  = {test_rules} → {len(test)} examples")

    return train, test, label_mapping



# -------------------------------------------------------------
# Auto-normalization helper for SNLI
# -------------------------------------------------------------
def normalize_snli_fields(example: Dict[str, Any]) -> Dict[str, Any]:
    """Unify SNLI examples to always use 'premise' and 'hypothesis' fields."""
    premise = example.get("premise") or example.get("sentence1") or example.get("sentence1_text")
    hypothesis = example.get("hypothesis") or example.get("sentence2") or example.get("sentence2_text")
    label = example.get("gold_label") or example.get("label")

    if premise and hypothesis and label:
        return {"premise": premise, "hypothesis": hypothesis, "gold_label": label}
    return {}


def normalize_snli_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [normalize_snli_fields(ex) for ex in dataset]
    normalized = [ex for ex in normalized if ex]  # remove empty dicts
    print(f"Normalized {len(normalized)} SNLI examples (kept {len(normalized)}/{len(dataset)})")
    return normalized


# -------------------------------------------------------------
# Dataset Configuration Class
# -------------------------------------------------------------
@dataclass
class DatasetConfig:
    name: str
    labels: List[str]
    prompt_template: str
    data_loader: Callable
    required_fields: List[str]
    label_token_mapping: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.label_token_mapping is None:
            self.label_token_mapping = {label: chr(65 + i) for i, label in enumerate(self.labels)}
        self.label_mapping = {label: i for i, label in enumerate(self.labels)}
        
        # FIXED VERSION (3 or 4 outputs)
    # -----------------------------
    def load_data(self, data_dir: str = "./data/"):
        loaded = self.data_loader(folder_path=data_dir)

        # Case: MoRe full + SNLI → 4 outputs
        if len(loaded) == 4:
            train_data, test_data, val_data, label_mapping = loaded

        # Case: MoRe_split1/2/3 → 3 outputs (no val)
        elif len(loaded) == 3:
            train_data, test_data, label_mapping = loaded
            val_data = []

        else:
            raise ValueError(f"Unexpected number of outputs from loader: {len(loaded)}")

        # SNLI normalization
        if self.name.lower() == "snli":
            train_data = normalize_snli_dataset(train_data)
            test_data = normalize_snli_dataset(test_data)
            val_data = normalize_snli_dataset(val_data)

        self.label_mapping = label_mapping
        return train_data, test_data, val_data, label_mapping



#works for previou experiments  - changed for more finetuning reuse if

    # def load_data(self, data_dir: str = "./data/"):
    #     train_data, test_data, val_data, _ = self.data_loader(folder_path=data_dir)

    #     # Auto-normalize SNLI datasets
    #     if self.name.lower() == "snli":
    #         train_data = normalize_snli_dataset(train_data)
    #         test_data = normalize_snli_dataset(test_data)
    #         val_data = normalize_snli_dataset(val_data)

    #     return train_data, test_data, val_data, self.label_mapping
    
    
    

    def validate_example(self, example: Dict[str, Any]) -> bool:
        for field in self.required_fields:
            if field not in example:
                return False
        return 'gold_label' in example and example['gold_label'] in self.label_mapping


# -------------------------------------------------------------
# Dataset Registry
# -------------------------------------------------------------
DATASET_CONFIGS = {
    # --- SNLI (3-way) ---
    'snli': DatasetConfig(
        name='snli',
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=SNLI_PROMPT,
        data_loader=get_snli_data,
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),

    'MoRe': DatasetConfig(
        name='MoRe',
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=get_more_data,
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),

    "more_split1": DatasetConfig(
        name="more_split1",
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split1", folder_path),
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),
    "split1_diagnostic": DatasetConfig(
        name="split1_diagnostic",
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split1_diagnostic", folder_path),
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),
    
    "more_split2": DatasetConfig(
        name="more_split2",
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split2", folder_path),
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),
    
    "more_split3": DatasetConfig(
        name="more_split3",
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split3", folder_path),
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),
    "more_rules": DatasetConfig(
        name="more_rules",
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split(
            "rules",
            folder_path
        ),
        required_fields=['premise', 'hypothesis', 'gold_label']
    ),
    "split1_dif_templates": DatasetConfig(
        name="split1_dif_templates",
        labels=['entailment','neutral','contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split1_dif_templates", folder_path),
        required_fields=['premise','hypothesis','gold_label']
    ),

    "split2_dif_templates": DatasetConfig(
        name="split2_dif_templates",
        labels=['entailment','neutral','contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split2_dif_templates", folder_path),
        required_fields=['premise','hypothesis','gold_label']
    ),

    "split3_dif_templates": DatasetConfig(
        name="split3_dif_templates",
        labels=['entailment','neutral','contradiction'],
        prompt_template=MORE_PROMPT,
        data_loader=lambda folder_path: get_more_split("split3_dif_templates", folder_path),
        required_fields=['premise','hypothesis','gold_label']
    )
    

}



# -------------------------------------------------------------
# Accessors
# -------------------------------------------------------------
def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]


def get_dataset(dataset_name: str):
    return get_dataset_config(dataset_name).load_data()


def get_prompt_template(dataset_name: str):
    return get_dataset_config(dataset_name).prompt_template


