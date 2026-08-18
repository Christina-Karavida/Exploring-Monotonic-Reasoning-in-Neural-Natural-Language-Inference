from __future__ import annotations
import json
import csv
import os
import random
from collections import defaultdict
import sys
from pathlib import Path


from pluralization_helper import pluralize
from sentence_banks import ALL, SOME, NO, NO_OPERATOR

sys.path.append(str(Path.cwd().parent / "taxonomy"))
from taxonomy_helpers import load_pairs


   


"""
MoRe Dataset Generation

This script generates the final MoRe dataset:

1. It loads hypernym–hyponym pairs from the taxonomy (via the load_pairs function present in the taxonomy_helpers.py)
2. It retrieves the sentence templates loaded from sentence_banks.py.
3. It combines (1) and (2) into 12 monotonicity rules (R1–R12).

"""


# Helpers

"""
Optional limit on the number of templates used per category.

If None, all templates are used.
If set to an integer, only that many templates are selected 
--> this setting can be used for the generation of Split 1 with unseen templates
"""

TEMPLATE_LIMIT = 15      # set to 15 to keep first 15 (or last 15 if tail=True), else set to None 
TEMPLATE_USE_TAIL = False  # False = for the first 15 templates, True = for the last 15 templates


def get_templates(bank: dict, category: str, limit=None, tail=False):
    """retrieves templates when an optional limit is set (first N or last N)."""
    templates = (
        bank.get(category, [])
        or bank.get(category.lower(), [])
        or bank.get(category.capitalize(), [])
    )

    if limit is None:
        return templates
    if tail:
        return templates[-limit:]
    else:
        return templates[:limit]

    
def pluralize_hypernym(word: str, number: str = "pl") -> str:
    """
    converts a hypernym to its singular or plural form 
    """
    if number == "sing":
        return word.lower()
    return pluralize(word)



# Functions to generate Rules 1 - 12


def r1 (H, h, category, subcategory):
    """
    Rule 1 = All + hypernym → hyponym = entailment
    Example: P: All animals moved | H: All dogs moved = entailment
    Returns a list of dicts containing all the generated examples for R1
    """
    rows = []  #empty list to store the examples that will be generated for this rule
    for tmpl in get_templates(ALL, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL): # retrieves all sentence templates from sentence_bank.py genrated for the operator 'all'
        prem = tmpl.format(X=pluralize_hypernym(H, "pl")) # this line creates the premise : it appends the hypernym term pluralized (e.g. All dogs)
        hyp  = tmpl.format(X=pluralize(h))  # creates the hypothesis : it appends the hyponym term pluralized (e.g. All beagles)
        rows.append({
            "gold_label": "entailment",
            "rule": "R1",
            "operator": "all", "monotonicity": "downward",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows



def r2 (H, h, category, subcategory):
    """Rule 2: All + hyponym → hypernym = neutral """
    rows = []
    for tmpl in get_templates(ALL, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize(h))
        hyp  = tmpl.format(X=pluralize_hypernym(H, "pl"))
        rows.append({
            "gold_label": "neutral",
            "rule": "R2",
            "operator": "all", "monotonicity": "non-monotone",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows

def r3 (H, h, category, subcategory):
    """Rule 3: Some + hyponym → hypernym = entailment """
    rows = []
    for tmpl in get_templates(SOME, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize(h))
        hyp  = tmpl.format(X=pluralize_hypernym(H, "pl"))
        rows.append({
            "gold_label": "entailment",
            "rule": "R3",
            "operator": "some", "monotonicity": "upward",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows


def r4 (H, h, category, subcategory):
    """Rule 4: Some + hypernym → hyponym = neutral """
    rows = []
    for tmpl in get_templates(SOME, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize_hypernym(H, "pl"))
        hyp  = tmpl.format(X=pluralize(h))
        rows.append({
            "gold_label": "neutral",
            "rule": "R4",
            "operator": "some", "monotonicity": "non-monotone",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows

def r5 (H, h, category, subcategory):
    """Rule 5 : No + hyper → hypo = entailment """
    rows = []
    for tmpl in get_templates(NO, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize_hypernym(H, "pl"))   
        hyp  = tmpl.format(X=pluralize(h))                
        rows.append({
            "gold_label": "entailment",
            "rule": "R5",
            "operator": "no", "monotonicity": "downward",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows

def r6 (H, h, category, subcategory):
    """Rule 6 : No + hypo → hyper = neutral"""
    rows = []
    for tmpl in get_templates(NO, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize(h))                # plural under 'No'
        hyp  = tmpl.format(X=pluralize_hypernym(H, "pl"))   # plural under 'No'
        rows.append({
            "gold_label": "neutral",
            "rule": "R6",
            "operator": "no", "monotonicity": "downward",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows


#helper for rules 7,8
def r7_and_8_helper(tmpl: str, noun: str) -> str:
    """
    Helper function for the rules that don't start with an operator (i.e. R7 _R8)
    The templates of R7,8 start with a placeholder '{X}' in the sentence_banks.py
    The purpose of this helper is to therefore put in the place of the placeholder
    the following structure : 'The ' +  pluralized, lowercased noun 
    e.g. "{X} crossed the meadow at sunrise." --> "The mammals crossed the meadow at sunrise."
    """
    s = tmpl.strip() # removes whitespace
    s = s.replace("{x}", "{X}") #lowercases the placeholder to ensure full coverage (as in some cases gpt generated as both lowercase and upper case)
    out = "The " + s.replace("{X}", noun.lower(), 1) # removes {X} and replaces it with 'The'
    return out  # returns the sentence having replaces {x} with 'The'


def r7 (H, h, category, subcategory):
    """Rule 7 :hyper → hypo = entailment"""
    rows = []
    for tmpl in get_templates(NO_OPERATOR, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = r7_and_8_helper(tmpl, pluralize(H)) # appneds the pluralized hypernym after 'The' e.g. The dogs sat on the mat
        hyp  = r7_and_8_helper(tmpl, pluralize(h)) # appneds the pluralized hyponym after 'The' e.g. The beagles sat on the mat
        rows.append({
            "gold_label": "entailment",
            "rule": "R7",
            "operator": "none", "monotonicity": "upward",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows


def r8 (H, h, category, subcategory):
    """Rule 7 :hypo → hyper = neutral"""
    rows = []
    for tmpl in get_templates(NO_OPERATOR, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = r7_and_8_helper(tmpl, pluralize(h))  #same logic as rule 7
        hyp  = r7_and_8_helper(tmpl, pluralize(H))
        rows.append({
            "gold_label": "neutral",
            "rule": "R8",
            "operator": "none", "monotonicity": "downward",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows


# contradiction rules  = R9-12
# for rules 9 -12 we recombine the templates for 'all' 'some' and 'no' from sentence_bank.py

def r9 (H, h, category, subcategory):
    """Rule 9 = (All + hypo) + (No + hyper) = contradiction """
    rows = []
    for tmpl in get_templates(ALL, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize_hypernym(H, "pl"))
        hyp  = tmpl.replace("All", "No").format(X=pluralize(h))  # the premise uses the 'all' template +  hypernym   and the hypothesis uses the 'no' template +  hyponym 
        rows.append({
            "gold_label": "contradiction",
            "rule": "R9",
            "operator": "all-no", "monotonicity": "non-monotone",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows




def r10 (H, h, category, subcategory):
    """Rule 10 = (No + hyper) + (All + hypo) = contradiction """
    rows = []
    for tmpl in get_templates(NO, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize_hypernym(H, "pl"))              
        hyp  = tmpl.replace("No", "All").format(X=pluralize(h))  #same logic as R9 but reversed
        rows.append({
            "gold_label": "contradiction",
            "rule": "R10",
            "operator": "no-all", "monotonicity": "non-monotone",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows




def r11 (H, h, category, subcategory):
    """Rule 11 = (No + hyper) + (Some + hypo) = contradiction """
    rows = []
    for tmpl in get_templates(NO, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize_hypernym(H, "pl"))                
        hyp  = tmpl.replace("No", "Some").format(X=pluralize(h))        
        rows.append({
            "gold_label": "contradiction",
            "rule": "R11",
            "operator": "no-some", "monotonicity": "non-monotone",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows


def r12 (H, h, category, subcategory):
    """Rule 12 = (Some + hypo) + (No + hyper) = contradiction """
    rows = []
    for tmpl in get_templates(SOME, category, limit=TEMPLATE_LIMIT, tail=TEMPLATE_USE_TAIL):
        prem = tmpl.format(X=pluralize(h))                              
        hyp  = tmpl.replace("Some", "No").format(X=pluralize_hypernym(H, "pl"))  
        rows.append({
            "gold_label": "contradiction",
            "rule": "R12",
            "operator": "some-no", "monotonicity": "non-monotone",
            "premise": prem, "hypothesis": hyp,
            "category": category.lower(), "subcategory": subcategory.lower(),
            "hypernym": H.lower(), "hyponym": h.lower()
        })
    return rows



rule_functions = {
    "R1": r1,
    "R2": r2,
    "R3": r3,
    "R4": r4,
    "R5": r5,
    "R6": r6,
    "R7": r7,
    "R8": r8,
    "R9": r9,
    "R10": r10,
    "R11": r11,
    "R12": r12
}




def generate_examples():
    """
    This function generates the complete MoRe dataset:
    - it loads all hypernym–hyponym pairs from the taxonomy
    - it applies each of the 12 monotonicity rules to every pair and generates the corresponding  examples
    
    Each example is assigned a unique id and stored:
    (1) in a list containing the full dataset 
    (2) in a dictionary that groups examples by rule

    Returns
    all_rows --> list[dict] i.e. the complete generated dataset.
    by_rule --> dict[str, list[dict]] i.e. examples grouped by rule
    
    """
    all_examples = []  # empty list to append generates examples
    examples_by_rule = {rule_name: [] for rule_name in rule_functions} # a dict where R1-12 stores its own examples
    example_id = 1  #sets an unique identifier / example
    pairs = load_pairs()  #loads all pairs

    for pair in pairs:
        hyponym = pair["item"]  # gets the hyponym 
        hypernym = pair["hypernym"]  # gets the hypernym
        category = pair["category"]   # gets the general category (e.g animals mammals etc)
        subcategory = pair["subcategory"] # gets the more specific category (e.g. dogs, cats)
        for rule_name, rule in rule_functions.items():  # iterates over all registered rules
            generated_rows = rule(hypernym, hyponym, category, subcategory) # and applies them to all retrieved pairs
            for row in generated_rows:  
                row_with_id = {"id": str(example_id), **row}  #adds the unique id
                all_examples.append(row_with_id)  # every example is added in the dataset
                examples_by_rule[rule_name].append(row_with_id) # organizes per rule
                example_id += 1
    return all_examples, examples_by_rule  # returns 1) the full dataset 2) the dataset grouped by rule




# writers in jsnl and csv format
def write_jsonl(rows, path):
    "This functions writes the examples to a jsonl file"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            
def write_csv(rows, path):
    "Writes the examples in a csv"
    headers = [
        "id","premise","hypothesis","gold_label","category",
            "subcategory","hypernym","hyponym","rule","operator","monotonicity"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, "") for key in headers})
    
    
    
def main(out_dir="data_final"):
    """Generates the full dataset"""
    os.makedirs(out_dir, exist_ok=True)
    all_examples, examples_by_rule = generate_examples()
    write_jsonl(all_examples, os.path.join(out_dir, "all_rules_of_more.jsonl")) # writes all rules together 
    write_csv(all_examples, os.path.join(out_dir, "all_rules_of_more.csv"))

    for rule_name, rows in examples_by_rule.items():  # writes them per rule
        write_jsonl(rows, os.path.join(out_dir, f"{rule_name}.jsonl"))
        write_csv(rows, os.path.join(out_dir, f"{rule_name}.csv"))

    print(f" Files generated in: {out_dir}")
    for rule_name in sorted(examples_by_rule.keys()):
        print(f"  {rule_name}: {len(examples_by_rule[rule_name])} examples")
    print(f"  All: {len(all_examples)} examples total")

    




def print_rule_stats(jsonl_path="data_final/all_rules.jsonl", topn=10):
    import pandas as pd
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    df = pd.DataFrame(data)

    print("\n=== GLOBAL STATS ===")
    print("Rule Distribution:"); print(df["rule"].value_counts(), "\n")
    print("Label Distribution:"); print(df["gold_label"].value_counts(), "\n")
    print("Operator Distribution:"); print(df["operator"].value_counts(), "\n")
    print(f"Total examples: {len(df)}\n")

    df["prem_len"] = df["premise"].apply(lambda x: len(x.split()))
    df["hyp_len"] = df["hypothesis"].apply(lambda x: len(x.split()))

    print("Average Lengths (global):")
    print(df[["prem_len","hyp_len"]].mean(), "\n")

    print("Average Length per Rule:")
    print(df.groupby("rule")[["prem_len","hyp_len"]].mean(), "\n")

    print("Top Categories (global):")
    print(df["category"].value_counts().head(topn), "\n")

    print("Top Subcategories (global):")
    print(df["subcategory"].value_counts().head(topn), "\n")

    return df



def split_statistics(df):

    print("\nHYPERNYM FREQUENCIES")
    print(df["hypernym"].value_counts())

    print("\nHYPONYM FREQUENCIES")
    print(df["hyponym"].value_counts())

    print("\nHYPERNYM-HYPONYM PAIR FREQUENCIES")
    print(
        df.groupby(["hypernym", "hyponym"])
          .size()
          .sort_values(ascending=False)
    )

    print("\nUNIQUE COUNTS")
    print(f"Hypernyms: {df['hypernym'].nunique()}")
    print(f"Hyponyms: {df['hyponym'].nunique()}")
    print(
        f"Pairs: {df[['hypernym','hyponym']].drop_duplicates().shape[0]}"
    )