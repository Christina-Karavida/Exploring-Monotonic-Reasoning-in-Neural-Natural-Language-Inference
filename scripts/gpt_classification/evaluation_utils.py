
import os
import os, glob, csv, re, torch
from gpt_classification.gpt_utils import get_logger
from gpt_classification.train_gpt2_classifier import GPTClassifier
from data.dataset_config import get_dataset_config
from gpt_classification.gpt_args import GPTArgs



def load_model(model_path: str, dataset_name: str) -> GPTClassifier:
    logger = get_logger()
    dataset_config = get_dataset_config(dataset_name)
    args = GPTArgs()
    args.model_name = "gpt2"
    args.dataset = dataset_name
    args.exp_name = os.path.basename(os.path.dirname(model_path))

    classifier = GPTClassifier(args, logger, dataset_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    classifier.model.load_state_dict(state_dict)
    classifier.model.eval().to(device)

    print(f"Model loaded from: {model_path}")
    return classifier



def load_dataset(dataset_name: str, data_dir: str = "./data/"):
    dataset_config = get_dataset_config(dataset_name)
    train_data, test_data, val_data, label_mapping = dataset_config.load_data(data_dir)
    print(f"Dataset '{dataset_name}' loaded: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")
    return train_data, test_data, val_data, label_mapping



def unpack_prediction(res):
    if isinstance(res, dict):
        return res["true_label"], res["predicted_label"]
    else:
        # assume tuple: (true_label, predicted_label)
        return res


def evaluate_model_on_dataset(
    classifier: GPTClassifier,
    dataset_name: str,
    data_dir: str,
    split: str = "test",  # new argument
    save_csv: bool = True,
    n_preview: int = 5
):
    print(f"Evaluating on '{dataset_name}' ({split} split)")
    train_data, test_data, val_data, label_mapping = load_dataset(dataset_name, data_dir)

    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        data = test_data

    template_vars = set(re.findall(r"\{(\w+)\}", classifier.prompt_template))
    formatted_data = []
    for x in data:
        entry = {"gold_label": x["gold_label"], "rule": x.get("rule", "unknown")}
        if "premise" in template_vars and "hypothesis" in template_vars:
            entry["premise"] = x.get("premise", x.get("sentence1", ""))
            entry["hypothesis"] = x.get("hypothesis", x.get("sentence2", ""))
        elif "sentence1" in template_vars and "sentence2" in template_vars:
            entry["sentence1"] = x.get("sentence1", x.get("premise", ""))
            entry["sentence2"] = x.get("sentence2", x.get("hypothesis", ""))
        else:
            raise ValueError(f"Unexpected template variables: {template_vars}")
        formatted_data.append(entry)

    classifier.set_custom_data(
        train_data=formatted_data[:1],
        val_data=formatted_data[:1],
        test_data=formatted_data,
        label_mapping=label_mapping,
    )

    acc, results = classifier.evaluate(classifier.dataloaders["test"], return_predictions=True)
    print(f"Accuracy on {dataset_name} ({split} split): {acc:.2%}")

    per_rule_acc = {}
    rule_counts = {}
    for res, sample in zip(results, formatted_data):
        rule = sample.get("rule", "unknown")
        true_label, predicted_label = unpack_prediction(res)
        correct = true_label == predicted_label
        rule_counts.setdefault(rule, {"correct": 0, "total": 0})
        rule_counts[rule]["correct"] += int(correct)
        rule_counts[rule]["total"] += 1
    for rule, stats in rule_counts.items():
        per_rule_acc[rule] = stats["correct"] / stats["total"]

    print("Per-rule accuracy:")
    for r, a in sorted(per_rule_acc.items()):
        print(f"  {r}: {a:.2%}")

    if save_csv:
        os.makedirs("./results/eval_results/", exist_ok=True)
        out_path = f"./results/eval_results/predictions_{classifier.args.exp_name}_on_{dataset_name}_{split}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence1/premise", "sentence2/hypothesis", "gold_label", "predicted_label", "rule"])
            for i, item in enumerate(results):
                s1 = formatted_data[i].get("premise", formatted_data[i].get("sentence1", ""))
                s2 = formatted_data[i].get("hypothesis", formatted_data[i].get("sentence2", ""))
                rule = formatted_data[i]["rule"]
                true_label, predicted_label = unpack_prediction(item)
                writer.writerow([s1, s2, true_label, predicted_label, rule])
        print(f"Predictions saved to {out_path}")

    if n_preview > 0:
        print("Sample predictions:")
        for i, item in enumerate(results[:n_preview]):
            s1 = formatted_data[i].get("premise", formatted_data[i].get("sentence1", ""))
            s2 = formatted_data[i].get("hypothesis", formatted_data[i].get("sentence2", ""))
            rule = formatted_data[i]["rule"]
            true_label, predicted_label = unpack_prediction(item)
            print(
            f"{i+1}. Premise: {s1}\n   Hypothesis: {s2}\n   Rule: {rule}\n"
            f"   Gold: {true_label} | Predicted: {predicted_label}\n"
            )

    return {"overall_accuracy": acc, "per_rule_accuracy": per_rule_acc}