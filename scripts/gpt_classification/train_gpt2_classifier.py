# Code taken and adapted from Eshuijs et al. (2025): https://github.com/watermeleon/shortcut_mechanisms/blob/main/robin_nlp/gpt_classification/train_gpt_text_classifier.py



from collections import defaultdict
from typing import Any, Dict, List, Optional
import os
import random
import re
import json
import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformer_lens import HookedTransformer
from logging import Logger
from tqdm import tqdm

from data.dataset_config import get_dataset_config, DatasetConfig
from data.dataset_config import get_more_split



class GPTClassifier:
    def __init__(self, args: Any, logger: Logger, dataset_config: DatasetConfig) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.val_data_full: Optional[List[Dict[str, str]]] = None
        self.manual_prepend_bos = getattr(self.args, 'manual_prepend_bos', False)
        self.dataset_config = dataset_config
        self.label_mapping: Dict[str, int] = dataset_config.label_mapping
        self.label_token_mapping: Dict[str, str] = dataset_config.label_token_mapping
        self.prompt_template: str = dataset_config.prompt_template
        self.template_params: List[str] = self._get_template_params_regex(self.prompt_template)
        self.setup_model_and_tokenizer()


    def setup_model_and_tokenizer(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        is_gpt2 = "gpt2" in self.args.model_name
        self.model = HookedTransformer.from_pretrained(
            model_name=self.args.model_name,
            refactor_factored_attn_matrices=is_gpt2,
            default_padding_side="left",
        )
        self.tokenizer = self.model.tokenizer
        
        
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
            
        
        
    def set_custom_data(self, train_data, test_data, val_data, label_mapping):
        self.label_mapping = label_mapping
        self.prepare_datasets(train_data, val_data, test_data)


    def _get_template_params_regex(self, template_string: str) -> List[str]:
        pattern = r'\{(\w+)\}'
        return re.findall(pattern, template_string)
        

    def _process_dataset(self, examples, data_type):
        features = []
        template_params = set(re.findall(r'\{(\w+)\}', self.prompt_template))
        for example in examples:
            label = example["gold_label"]
            if label not in self.label_mapping:
                continue
            processed_example = {
                key: self.tokenizer.decode(
                    self.tokenizer.encode(value, add_special_tokens=False)[:self.args.max_tokens]
                ) for key, value in example.items() if key in template_params
            }
            if any(param not in processed_example for param in template_params):
                continue
            prompt = self.prompt_template.format(**processed_example)
            if data_type == "train":
                prompt += self.label_token_mapping[label]
            features.append({"text": prompt, "label": self.label_mapping[label], "raw": example})
        return features



    def prepare_datasets(self, train_data, val_data, test_data):
        self.label_mapping = {label: idx for idx, label in enumerate(self.dataset_config.labels)}
        datasets = {"train": (train_data, "train"), "val": (val_data, "test"), "test": (test_data, "test")}
        self.dataloaders = {}
        for name, (data, data_type) in datasets.items():
            features = self._process_dataset(data, data_type)
            sampler = RandomSampler(features) if name == "train" else SequentialSampler(features)
            batch_size = self.args.batch_size if name == "train" else self.args.eval_batch_size
            self.dataloaders[name] = DataLoader(
                features,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

    def encode_batch(self, texts):
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.args.max_tokens + 50,
            padding='longest',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
    
        if self.manual_prepend_bos:
            texts = [self.tokenizer.bos_token + text for text in texts]
    
        encoded = self.encode_batch(texts)
    
        # return raw batch so we can save predictions later
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels),
            batch,     # <--- the 4th element
        )




    def train(self):
        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=float(self.args.learning_rate))
        total_steps = len(self.dataloaders["train"]) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

        for epoch in range(self.args.epochs):
            train_loss = self._train_epoch(optimizer, scheduler)
            print(f"\nEpoch {epoch+1}/{self.args.epochs} — Train loss: {train_loss:.4f}")

        print("Training finished.\n")

    def _train_epoch(self, optimizer, scheduler):
        self.model.train()
        total_loss = 0

        for step, (input_ids, attention_mask, _, _) in enumerate(tqdm(self.dataloaders["train"])):
    


            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask, loss_per_token=True, return_type="loss")
            loss = outputs[:, -1].mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(self.dataloaders["train"])
        
    
    @torch.no_grad()
    def evaluate(self, dataloader, return_predictions=False):
        self.model.eval()
        correct, total = 0, 0
        results = []

        label_token_ids = torch.tensor([
            self.tokenizer.encode(" " + tok, add_special_tokens=False)[0]
            for tok in self.label_token_mapping.values()
        ]).to(self.device)

        for input_ids, attention_mask, labels, batch_raw in tqdm(dataloader):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[:, -2, :]

            label_probs = logits[:, label_token_ids]
            preds = torch.argmax(label_probs, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if return_predictions:
                for p, g in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                    results.append((p, g))

        acc = (correct + 1e-8) / total
        return acc, results





# to save predictions 
    def evaluate_and_save_predictions(self, save_path):
        self.model.eval()

        label_token_ids = torch.tensor([
            self.tokenizer.encode(" " + tok, add_special_tokens=False)[0]
            for tok in self.label_token_mapping.values()
        ]).to(self.device)

        all_results = []

        for input_ids, attention_mask, labels, batch_raw in tqdm(self.dataloaders["test"]):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[:, -2, :]

            probs = logits[:, label_token_ids]
            preds = torch.argmax(probs, dim=-1)

            for pred_idx, true_idx, raw in zip(preds.cpu().tolist(), labels.tolist(), batch_raw):
                gold_label = list(self.label_mapping.keys())[list(self.label_mapping.values()).index(true_idx)]
                pred_label = list(self.label_mapping.keys())[pred_idx]
            
                ex = raw["raw"]
            
                out = {
                    "id": ex.get("id",""),
                    "premise": ex.get("premise", ""),
                    "hypothesis": ex.get("hypothesis", ""),
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "rule": ex.get("rule", ""),
                    "category": ex.get("category", ""),
                    "subcategory": ex.get("subcategory", ""),
                    "hypernym": ex.get("hypernym", ""),
                    "hyponym": ex.get("hyponym", "")
                    
                }

                all_results.append(out)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
        print(f"\nSaved predictions to: {save_path}")
        return all_results




def parse_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample(data, n):
    if n is None or n <= 0:
        return data
    return data[:min(n, len(data))]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gpt_classifier")
    #args = parse_config("./config.yml")
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    args = parse_config(config_path)
    dataset_config = get_dataset_config(args["dataset"])

    class X: pass
    args_obj = X()
    args_obj.__dict__.update(args)

    data_dir = "/content/drive/MyDrive/thesis_code/data" 
    
    if args["dataset"] == "more_rules":
        # RULE MODE → pass train_rules + test_rules
        train_data, test_data, label_mapping = get_more_split(
            split_name="rules",
            folder_path=data_dir,
            train_rules=args.get("train_rules", None),
            test_rules=args.get("test_rules", None)
        )
        val_data = []   # no val in rule mode
    else:
        train_data, test_data, val_data, label_mapping = dataset_config.load_data(data_dir)

    if "sample_size" in args and args["sample_size"] > 0:   #optional --> for debugging
        print(f"\n Using sample_size={args['sample_size']}\n")
        train_data = sample(train_data, args["sample_size"])
        test_data  = sample(test_data,  args["sample_size"])
        val_data   = sample(val_data,   args["sample_size"])


    # init model
    classifier = GPTClassifier(args_obj, logger, dataset_config)
    classifier.set_custom_data(train_data, test_data, val_data, label_mapping)

    # training classifier
    classifier.train()
    
    test_acc, _ = classifier.evaluate(
        classifier.dataloaders["test"], return_predictions=False
    )
    print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%\n")
    
    #saving model
    model_path = os.path.join(
        args["paths"]["output_dir"], 
        args["paths"]["model_save_path"]
    )
    classifier.save_model(model_path)
    print(f"Saved model to: {model_path}")
    
    # preds
    pred_path = os.path.join(
        args["paths"]["output_dir"],
        args["paths"]["predictions_path"]
    )
    classifier.evaluate_and_save_predictions(pred_path)
    print(f"Saved predictions to: {pred_path}")

