
import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_predictions(path: str) -> pd.DataFrame:
    """
    Loads a json file into a df.
    Expected columns:
      - gold_label
      - predicted_label
      - rule
    """
    with open(path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    required_columns = {"gold_label", "predicted_label", "rule"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f" A missing column found: {missing}")

    return df



def split_accuracy(df):
    correct = (df.gold_label == df.predicted_label).sum()
    total = len(df)
    accuracy = correct / total

    print(f"\nAccuracy for this split:")
    print(f"{accuracy:.4f} ({correct}/{total})")

    return accuracy



def per_rule_report(df,show_error_types=True,max_error_types=5):
    """
    The funcntion displays the statistics of each rule
    Required columns:
        - gold_label
        - predicted_label
        - rule
    results are sorted from lowest to highest accuracy
    """
    scores = []

    print("\n Per-Rule Report: \n")

    for rule, group in df.groupby("rule"):
        total = len(group)
        correct = (group.gold_label == group.predicted_label).sum()
        accuracy = correct / total
        errors = total - correct

        scores.append({
            "rule": rule,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
        })

    scores = sorted(scores, key=lambda x: x["accuracy"])

    for score in scores:    # this way we can see which was the misclassified label
        print(
            f"{score['rule']:>3} | "
            f"acc: {score['accuracy']:.6f} | "
            f"{score['correct']:>6}/{score['total']:<6} | "
            f"errors: {score['errors']}"
        )

        if show_error_types and score["errors"] > 0:  # displays the misclassified label e.g. neutral instead of entailment
            show = df[
                (df.rule == score["rule"]) &
                (df.gold_label != df.predicted_label)
            ]

            counts = Counter(zip(show.gold_label, show.predicted_label))
            
            for (g, p), c in counts.most_common(max_error_types):
                print(f"      └─ {g} → {p}: {c}")

    return pd.DataFrame(scores)



def inspect_errors(df):
    """
    Prints all misclassified samples
    """

    errors = df[df["gold_label"] != df["predicted_label"]]

    print(f"\nFound {len(errors)} errors.\n")

    for idx, row in errors.iterrows():
        print("=" * 80)
        print(f"Row: {idx}")
        print(f"Gold label:      {row['gold_label']}")
        print(f"Predicted label: {row['predicted_label']}")

        for col in df.columns:
            if col not in ["gold_label", "predicted_label"]:
                print(f"{col}: {row[col]}")

        print()

        
        
        

def snli_predictions(df: pd.DataFrame):
    """
    Analyze SNLI predictions.

    Returns:
        - per-label accuracy
        - confusion matrix
        - label confusion breakdown
    """

    # Per-label accuracy
    accuracy = (
        df.assign(correct=df["gold_label"] == df["predicted_label"])
        .groupby("gold_label")
        .agg(
            accuracy=("correct", "mean"),
            correct=("correct", "sum"),
            total=("correct", "size")
        )
        .assign(errors=lambda x: x["total"] - x["correct"])
        .reset_index()
    )

    # Confusion matrix
    matrix = pd.DataFrame(
        confusion_matrix(
            df["gold_label"],
            df["predicted_label"],
            labels=["entailment", "neutral", "contradiction"]
        ),
        index=["entailment", "neutral", "contradiction"],
        columns=["entailment", "neutral", "contradiction"]
    )

    # Error breakdown
    errors = (
        df[df["gold_label"] != df["predicted_label"]]
        .groupby(["gold_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return accuracy, matrix, errors
     


    

    
    
def pair_errors_per_rule(df, top_n=20):
    """
    Function used to inspect the specific errors made for each rule.
    """

    errors = df[df["gold_label"] != df["predicted_label"]]

    for rule in sorted(df["rule"].unique()):

        rule_df = df[df["rule"] == rule]
        rule_errors = errors[errors["rule"] == rule]

        total = len(rule_df)
        n_errors = len(rule_errors)
        correct = total - n_errors

        print("\n" + "-" * 90)
        print(f"{rule}")
        print("-" * 90)

        print(
            f"Accuracy: {correct/total:.4f} "
            f"({correct}/{total}) | Errors: {n_errors}"
        )

        if len(rule_errors) == 0:
            continue

        stats = []

        grouped = (
            rule_errors
            .groupby(
                [
                    "hypernym",
                    "hyponym",
                    "gold_label",
                    "predicted_label"
                ]
            )
            .size()
            .reset_index(name="errors")
            .sort_values("errors", ascending=False)
        )

        for _, row in grouped.head(top_n).iterrows():

            hyper = row["hypernym"]
            hypo = row["hyponym"]

            pair_total = len(
                rule_df[
                    (rule_df["hypernym"] == hyper)
                    &
                    (rule_df["hyponym"] == hypo)
                ]
            )

            pair_errors = row["errors"]
            pair_correct = pair_total - pair_errors

            stats.append(
                [
                    hyper,
                    hypo,
                    pair_correct,
                    pair_total,
                    pair_errors,
                    row["gold_label"],
                    row["predicted_label"]
                ]
            )

        stats = pd.DataFrame(
            stats,
            columns=[
                "hypernym",
                "hyponym",
                "correct",
                "total",
                "errors",
                "gold",
                "predicted"
            ]
        )

        print(stats.to_string(index=False))
        

        
        from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

        
    
    
    
    
    
        
def metrics_report(df, detailed_rules=False):
    """
    Prints:
        - Overall classification report
        - Per-rule metrics
        - (Optional) detailed report for each rule

    Returns:
        overall_metrics : dict
        overall_df      : DataFrame
        per_rule_df     : DataFrame
    """

    labels = ["entailment", "neutral", "contradiction"]

    print("\n" + "-" * 90)
    print("Report")
    print("=" * 90)

    print(
        classification_report(
            df["gold_label"],
            df["predicted_label"],
            labels=labels,
            digits=4,
            zero_division=0,
        )
    )

    overall_df = pd.DataFrame(
        classification_report(
            df["gold_label"],
            df["predicted_label"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T

    overall_metrics = {
        "accuracy": accuracy_score(
            df["gold_label"],
            df["predicted_label"],
        ),
        "precision_macro": precision_score(
            df["gold_label"],
            df["predicted_label"],
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            df["gold_label"],
            df["predicted_label"],
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            df["gold_label"],
            df["predicted_label"],
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": precision_score(
            df["gold_label"],
            df["predicted_label"],
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            df["gold_label"],
            df["predicted_label"],
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            df["gold_label"],
            df["predicted_label"],
            average="weighted",
            zero_division=0,
        ),
    }

    print("\nOverall metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric:20}: {value:.4f}")

    print("\n" + "=" * 90)
    print("Per-rule metrics")
    print("-" * 90)

    rows = []

    for rule, group in sorted(df.groupby("rule"), key=lambda x: x[0]):

        gold = group["gold_label"].iloc[0]

        precision, recall, f1, support = precision_recall_fscore_support(
            group["gold_label"],
            group["predicted_label"],
            labels=[gold],
            average=None,
            zero_division=0,
        )

        rows.append({
            "rule": rule,
            "label": gold,
            "accuracy": accuracy_score(
                group["gold_label"],
                group["predicted_label"]
            ),
            "precision": precision[0],
            "recall": recall[0],
            "f1": f1[0],
            "support": support[0],
        })

    per_rule_df = (
        pd.DataFrame(rows)
        .sort_values("accuracy")
        .reset_index(drop=True)
    )

    print(per_rule_df.to_string(index=False))

    if detailed_rules:

        print("\n" + "-" * 90)
        print("Detailed per-rule reports")
        print("-" * 90)

        for rule, group in sorted(df.groupby("rule"), key=lambda x: x[0]):

            print("\n" + "-" * 90)
            print(rule)
            print("-" * 90)

            print(
                classification_report(
                    group["gold_label"],
                    group["predicted_label"],
                    labels=labels,
                    digits=4,
                    zero_division=0,
                )
            )

    return overall_metrics, overall_df, per_rule_df
        
        
        
        
from sklearn.metrics import confusion_matrix
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt


def matrix_per_rule(df, title="GPT-2 Predictions"):

    labels = ["entailment", "neutral", "contradiction"]
    short = {
        "entailment": "ent",
        "neutral": "neu",
        "contradiction": "con",
    }

    rows = []
    index = []

    for rule, group in sorted(df.groupby("rule"), key=lambda x: x[0]):

        pred_dist = (
            group["predicted_label"]
            .value_counts(normalize=True)
            .reindex(labels, fill_value=0)
        )

        rows.append(pred_dist.values)

        gold = group["gold_label"].iloc[0]
        index.append(f"{rule} ({short[gold]})")

    matrix = pd.DataFrame(rows, columns=labels, index=index)

    fig, ax = plt.subplots(figsize=(5, 6))

    im = ax.imshow(matrix.values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)

    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index)

    ax.set_title(title)

    # write values inside cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    return fig, matrix

# # to fix jsonls
# import json

# input_file = "predictions/bert_base_on_more_splits/bert_split3.json"
# output_file = "predictions/bert_base_on_more_splits/bert_on_split3_fixed.json"

# records = []
# current = {}

# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.rstrip()

#         if line.startswith("- "):
#             if current:
#                 records.append(current)
#             current = {}

#             line = line[2:]
#             if ": " in line:
#                 k, v = line.split(": ", 1)
#                 current[k] = v

#         elif ": " in line:
#             k, v = line.strip().split(": ", 1)
#             current[k] = v

# if current:
#     records.append(current)

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(records, f, ensure_ascii=False)

# print(f"Saved {len(records)} records")