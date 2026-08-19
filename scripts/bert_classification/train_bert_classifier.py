import os
import yaml
import torch
import random
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data.dataset_config import get_dataset_config, get_more_split


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



class BERTNLIClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"]["name"],
            num_labels=len(cfg["model"]["labels"]),
        ).to(self.device)


    def load_data(self):
        dataset = self.cfg["dataset"]
        data_dir = self.cfg["data_dir"]

        if dataset == "more":
            train_data, test_data, label_mapping = get_more_split(
                split_name=self.cfg["split_name"],
                folder_path=data_dir,
            )

        elif dataset == "more_rules":
            train_data, test_data, label_mapping = get_more_split(
                split_name="rules",
                folder_path=data_dir,
                train_rules=self.cfg.get("train_rules"),
                test_rules=self.cfg.get("test_rules"),
            )

        else:
            dataset_config = get_dataset_config(dataset)
            train_data, test_data, _, label_mapping = dataset_config.load_data(data_dir)

        debug_cfg = self.cfg.get("debug", {})
        if debug_cfg.get("enabled", False):
            random.seed(42)
            train_n = debug_cfg.get("train_size", 10)
            test_n = debug_cfg.get("test_size", 10)

            train_data = random.sample(train_data, min(train_n, len(train_data)))
            test_data = random.sample(test_data, min(test_n, len(test_data)))

            print("DEBUG MODE ENABLED")
            print(f"   Train examples: {len(train_data)}")
            print(f"   Test  examples: {len(test_data)}")

        self.label_mapping = label_mapping
        self.train_ds = self._to_dataset(train_data)
        self.test_ds = self._to_dataset(test_data)
        self.test_raw = test_data   




    def _to_dataset(self, data):
        rows = []
        for ex in data:
            if ex["gold_label"] not in self.label_mapping:
                continue
            rows.append({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": self.label_mapping[ex["gold_label"]],
            })
        return Dataset.from_pandas(pd.DataFrame(rows))



    def tokenize(self):
        def tok(batch):
            return self.tokenizer(
                batch["premise"],
                batch["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=self.cfg["model"]["max_length"],
            )

        self.train_ds = self.train_ds.map(tok, batched=True)
        self.test_ds = self.test_ds.map(tok, batched=True)

        cols = ["input_ids", "attention_mask", "label"]
        self.train_ds.set_format("torch", columns=cols)
        self.test_ds.set_format("torch", columns=cols)


    def train(self):
        out_dir = os.path.join(
            self.cfg["paths"]["output_dir"],
            self.cfg["paths"]["model_dir"],
        )

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=self.cfg["training"]["epochs"],
            per_device_train_batch_size=self.cfg["training"]["batch_size"],
            per_device_eval_batch_size=self.cfg["training"]["eval_batch_size"],
            learning_rate=float(self.cfg["training"]["learning_rate"]),
            warmup_ratio=self.cfg["training"]["warmup_ratio"],
            weight_decay=self.cfg["training"]["weight_decay"],
            gradient_accumulation_steps=self.cfg["training"]["gradient_accumulation_steps"],
            fp16=self.cfg["training"]["fp16"],
            seed=self.cfg["training"]["seed"],
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            log_level="error",
            report_to="none",
            disable_tqdm=self.cfg["training"].get("disable_tqdm", False),
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.test_ds,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        self.trainer.train()




    def save_model(self):
        save_path = os.path.join(
            self.cfg["paths"]["output_dir"],
            self.cfg["paths"]["model_dir"],
            "final_model",
        )
        os.makedirs(save_path, exist_ok=True)
        self.trainer.save_model(save_path)


    def save_predictions(self):
        preds_output = self.trainer.predict(self.test_ds)
        preds = preds_output.predictions.argmax(axis=-1)
        labels = preds_output.label_ids
    
        inv_label_map = {v: k for k, v in self.label_mapping.items()}
        results = []
    
        for i, (pred, gold) in enumerate(zip(preds, labels)):
            ex = self.test_raw[i]   
    
            results.append({
                "id": ex.get("id", ""),
                "premise": ex.get("premise", ""),
                "hypothesis": ex.get("hypothesis", ""),
                "gold_label": inv_label_map[int(gold)],
                "predicted_label": inv_label_map[int(pred)],
                "rule": ex.get("rule", ""),
                "category": ex.get("category", ""),
                "subcategory": ex.get("subcategory", ""),
                "hypernym": ex.get("hypernym", ""),
                "hyponym": ex.get("hyponym", ""),
            })
    
        pred_path = os.path.join(
            self.cfg["paths"]["output_dir"],
            self.cfg["paths"]["model_dir"],
            self.cfg["paths"]["predictions_path"],
        )
    
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w") as f:
            yaml.safe_dump(results, f)
    
        print(f"predictions saved to: {pred_path}")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    CONFIG_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "bert_config.yml")
    )

    print(f"Loading config from: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    clf = BERTNLIClassifier(cfg)
    clf.load_data()
    clf.tokenize()
    clf.train()
    clf.save_model()
    clf.save_predictions()
