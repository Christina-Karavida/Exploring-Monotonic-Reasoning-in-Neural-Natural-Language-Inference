import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import os
from tqdm import tqdm
from data.dataset_config import get_dataset_config
from data.dataset_config import get_snli_data
import json



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


#this function loads snli + more's main test split
def load_data():
    data_dir = "/content/drive/MyDrive/thesis_code/data"

    # SNLI
    snli_train, snli_test, snli_val, _ = get_dataset_config("snli").load_data(data_dir)
    snli_test = pd.DataFrame(snli_test)[["premise", "hypothesis", "gold_label"]]

    # MoRe split1
    more_train, more_test, more_val, _ = get_dataset_config("more_split1").load_data(data_dir)
    mydata_test = pd.DataFrame(more_test)

    print(f"Loaded SNLI test = {len(snli_test)}, MoRe test = {len(mydata_test)}")
    return snli_test, mydata_test






# prompts
def zero_shot_prompt(premise, hypothesis):
    return (
        "Determine the logical relationship between the following premise and hypothesis for the task of NLI.\n"
        "Choose one label: entailment, neutral, contradiction.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )


def few_shot_prompt(premise, hypothesis):
    return (
        "Determine the logical relationship between the following premise and hypothesis for the task of NLI."
        "Choose one label: entailment, neutral, or contradiction.\n\n"

        "Examples:\n"
        "Premise: All insects crawled under the leaves.\n"
        "Hypothesis: All ants crawled under the leaves.\n"
        "Answer: entailment\n\n"

        "Premise: All parrots rested on the branches.\n"
        "Hypothesis: All birds rested on the branches.\n"
        "Answer: neutral\n\n"

        "Premise: Some tunas swam upstream.\n"
        "Hypothesis: Some fish swam upstream.\n"
        "Answer: entailment\n\n"

        "Premise: Some mammals ran in the forest.\n"
        "Hypothesis: Some horses ran in the forest.\n"
        "Answer: neutral\n\n"

        "Premise: No vehicles entered the tunnel.\n"
        "Hypothesis: No cars entered the tunnel.\n"
        "Answer: entailment\n\n"

        "Premise: No shirts were sold yesterday.\n"
        "Hypothesis: No clothes were sold yesterday.\n"
        "Answer: neutral\n\n"

        "Premise: The toys were placed in the box.\n"
        "Hypothesis: The dolls were placed in the box.\n"
        "Answer: neutral\n\n"

        "Premise: The cakes smelled nicely.\n"
        "Hypothesis: The sweets smelled nicely.\n"
        "Answer: neutral\n\n"

        "Premise: All fruit were cut in pieces.\n"
        "Hypothesis: No bananas were cut in pieces.\n"
        "Answer: contradiction\n\n"

        "Premise: No buildings collapsed after the earthquake.\n"
        "Hypothesis: All houses collapsed after the earthquake.\n"
        "Answer: contradiction\n\n"

        "Premise: No trees were planted along the avenue.\n"
        "Hypothesis: Some oaks were planted along the avenue.\n"
        "Answer: contradiction\n\n"
        
        "Premise: Some tulips bloomed in the garden.\n"
        "Hypothesis: No flowers bloomed in the garden.\n"
        "Answer: contradiction\n\n"

        "Now classify the following pair.\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )





def build_prompt(premise, hypothesis, mode):
    return zero_shot_prompt(premise, hypothesis) if mode == "zero" else few_shot_prompt(premise, hypothesis)


def predict_label(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    labels = ["entailment", "neutral", "contradiction"]
    scores = {lbl: logits[tokenizer.encode(" " + lbl)[0]].item() for lbl in labels}
    return max(scores, key=scores.get)



def evaluate(df, model, tokenizer, mode):
    preds = []
    golds = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {mode}"):
        prompt = build_prompt(row["premise"], row["hypothesis"], mode)
        pred = predict_label(prompt, model, tokenizer)
        preds.append(pred)
        golds.append(row["gold_label"])

    accuracy = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    return accuracy




def run_zero(save_path="/content/drive/MyDrive/thesis_code/results/zero_shot_predictions.json"):
    """
    Load model + data, run zero-shot evaluation on MoRe,
    save predictions as JSON, return accuracy.
    """
    model, tokenizer = load_model()
    _, more_test = load_data()

    acc = evaluate_and_save(
        df=more_test,
        model=model,
        tokenizer=tokenizer,
        mode="zero",
        save_path=save_path
    )

    print(f"\nZero-shot accuracy: {acc:.4f}")
    print(f"Predictions saved to: {save_path}")
    return acc



def run_few(save_path="/content/drive/MyDrive/thesis_code/results/few_shot_predictions.json"):
    """
    Load model + data, run few-shot evaluation on MoRe,
    save predictions as JSON, return accuracy.
    """
    model, tokenizer = load_model()
    _, more_test = load_data()

    acc = evaluate_and_save(
        df=more_test,
        model=model,
        tokenizer=tokenizer,
        mode="few",
        save_path=save_path
    )

    print(f"\nFew-shot accuracy: {acc:.4f}")
    print(f"Predictions saved to: {save_path}")
    return acc
    
    

def run_snli_zero(
    save_path="/content/drive/MyDrive/thesis_code/results/snli_zero_shot_predictions.json"
):
    model, tokenizer = load_model()

    snli_test, _ = load_data()

    acc = evaluate_and_save(
        df=snli_test,
        model=model,
        tokenizer=tokenizer,
        mode="zero",
        save_path=save_path
    )

    print(f"\nSNLI zero-shot accuracy: {acc:.4f}")
    print(f"Predictions saved to: {save_path}")

    return acc
    


def evaluate_and_save(df, model, tokenizer, mode, save_path):
    results = []
    preds = []
    golds = df["gold_label"].astype(str).tolist()  # ensure string labels

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{mode} evaluation"):
        prompt = build_prompt(row["premise"], row["hypothesis"], mode)
        pred = predict_label(prompt, model, tokenizer)

        preds.append(pred)

        entry = {
            "id": row.get("id", ""),
            "premise": row.get("premise", ""),
            "hypothesis": row.get("hypothesis", ""),
            "gold_label": row.get("gold_label", ""),
            "predicted_label": pred,
            "rule": row.get("rule", ""),
            "category": row.get("category", ""),
            "subcategory": row.get("subcategory", ""),
            "hypernym": row.get("hypernym", ""),
            "hyponym": row.get("hyponym", "")
        }

        results.append(entry)
        

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    accuracy = sum(p == g for p, g in zip(preds, golds)) / len(golds)

    return accuracy



def run_zero_shot_and_save(model, tokenizer, test_df, save_path):
    return evaluate_and_save(test_df, model, tokenizer, mode="zero", save_path=save_path)


def run_few_shot_and_save(model, tokenizer, test_df, save_path):
    return evaluate_and_save(test_df, model, tokenizer, mode="few", save_path=save_path)