import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split






# helper
def save_dataframe_as_jsonl(df, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


            
         
               
# HYponym Generalization Split    
def split_1(
    csv_path="../data/more_dataset/all_rules_of_more.csv",
    output_dir="../data/more_splits/split1",
    seed=42
):
    """
    This function creates the Split 1 of MoRe.
    Split 1 evaluates whether models can generalize monotonicity
    patterns to previously unseen hyponyms.
    The split is created by splitting the hyponyms from the complete MoRe dataset 
    (i.e. from all_rules_of_more.csv) into disjoint training and test sets.
    Consequently:
    - train and test contain different hyponyms
    - train and test contain all 12 monotonicity rules
    - train and test contain all hypernyms
    - train and test share the same sentence templates

    Parameters
    csv_path : str -->  Path to the full MoRe dataset.

    output_dir : str ---> irectory where the generated train/test files will be saved.

    Returns
    - Split 1 train data

    - Split 1 test data 

    Output: the function automatically creates:
    split1_train.csv
    split1_test.csv
    split1_train.jsonl
    split1_test.jsonl
-
    The function also performs two sanity checks:

    1. Hyponym overlap check: verifies that no hyponym appears in both train and test

    2. Rule coverage check: verifies that all monotonicity rules are represented in both partitions.
    """
    
    os.makedirs(output_dir, exist_ok=True)  # creates an output directory if not defined
    dataset = pd.read_csv(csv_path)    # loading the entire more dataset
    all_hyponyms = dataset["hyponym"].unique() # retrievs all unique hyponyms
    train_hyponyms, test_hyponyms = train_test_split(  # divides them in half randomly
        all_hyponyms,
        test_size=0.5,
        random_state=seed
    )
    train_split = dataset[
        dataset["hyponym"].isin(train_hyponyms)
    ].reset_index(drop=True)

    test_split = dataset[
        dataset["hyponym"].isin(test_hyponyms)
    ].reset_index(drop=True)
   
    # sanity check for potential hyponym overlap
    train_hyponym_set = set(train_split["hyponym"])
    test_hyponym_set = set(test_split["hyponym"])
    overlap = train_hyponym_set.intersection(test_hyponym_set)
    if len(overlap) == 0:
        print("Hyponym overlap check passed.")
    else:
        print("Hyponym overlap detected.")
        print("Examples:", list(overlap)[:20])
        
        
    # sanity check that all rules are in both sets
    all_rules = set(dataset["rule"].unique())

    if (
        all_rules.issubset(set(train_split["rule"].unique()))
        and
        all_rules.issubset(set(test_split["rule"].unique()))
    ):
        print("Rule coverage check passed.")
    else:
        print("Missing rules detected.")
    
    # saving in csv
    train_csv_path = os.path.join(output_dir, "split1_train.csv")
    test_csv_path = os.path.join(output_dir, "split1_test.csv")
    train_split.to_csv(train_csv_path, index=False)
    test_split.to_csv(test_csv_path, index=False)
    
    # saving in jsonl
    train_jsonl_path = os.path.join(output_dir, "split1_train.jsonl")
    test_jsonl_path = os.path.join(output_dir, "split1_test.jsonl")
    save_dataframe_as_jsonl(train_split, train_jsonl_path)
    save_dataframe_as_jsonl(test_split, test_jsonl_path)

    # statistics
    print("\nSplit 1 ok!")
    print(f"Train size: {len(train_split)}")
    print(f"Test size: {len(test_split)}")
    print(f"Output directory: {output_dir}")

    # Pair overlap check
    train_pairs = set(zip(train_split["hyponym"], train_split["hypernym"]))
    test_pairs = set(zip(test_split["hyponym"], test_split["hypernym"]))
    pair_overlap = train_pairs.intersection(test_pairs)

    if len(pair_overlap) == 0:
        print("\n Pair overlap check passed.")
    else:
        print("\n Pair overlap detected.")
        print("Examples:", list(pair_overlap)[:10])
        

    # train statistics
    print(f" Train Stats")
    print(f"Unique hyponyms: {train_split['hyponym'].nunique()}")
    print(f"Unique hypernyms: {train_split['hypernym'].nunique()}")
    print(
        f"Unique hyponym-hypernym pairs: "
        f"{train_split[['hyponym','hypernym']].drop_duplicates().shape[0]}"
    )

    print("\nLabel distribution:")
    print(train_split["gold_label"].value_counts())

    # test statistics
    print("\n Test Stats")
    print(f"Unique hyponyms: {test_split['hyponym'].nunique()}")
    print(f"Unique hypernyms: {test_split['hypernym'].nunique()}")
    print(
        f"Unique hyponym-hypernym pairs: "
        f"{test_split[['hyponym','hypernym']].drop_duplicates().shape[0]}"
    )

    print("\nLabel distribution:")
    print(test_split["gold_label"].value_counts())

    return train_split, test_split










# Hyponym Generalization Split
def split_2(
    csv_path="../data/more_dataset/all_rules_of_more.csv",
    output_dir="../data/more_splits/split2"
):
    """
    This function generates Split 2 of MoRe, that evaluates whether models can generalize
    to unseen hypernym-hyponyms and templates
    In this case, train and test data are separated by dividing the 44 semantic categories in two.
    This way we make sure that train and test data contain:
    - different hypernyms
    - different hyponyms
    - different templates (since each category uses its own templates)
    """

    os.makedirs(output_dir, exist_ok=True)
    dataset = pd.read_csv(csv_path)
    dataset = dataset[dataset["hypernym"] == dataset["category"]].copy() # we keep only the  broader categories e.g. beagle -> dogs, not beagle -> mammals / animals
    
    train_categories = [
        "mammals", "rodents", "felines", "ungulates", "fish", "insects",   # 22 
        "amphibians", "crustaceans", "dinosaurs", "bacteria", "protozoa",
        "trees", "fungi", "fruit", "furniture", "clothing", "jewellery",
        "toys", "buildings", "electronics", "instruments", "shapes"
    ]

    test_categories = [
        "dogs", "primates", "marsupials", "cetaceans", "birds", "reptiles",  # 22
        "arachnids", "mollusks", "microbes", "viruses", "plants", "shrubs",
        "vegetables", "sweets", "shoes", "accessories", "tops", "headwear",
        "shops", "monuments", "appliances", "vehicles"
    ]

    train_split = dataset[
        dataset["category"].isin(train_categories)
    ].reset_index(drop=True)

    test_split = dataset[
        dataset["category"].isin(test_categories)
    ].reset_index(drop=True)

    
    # sanity checks
    category_overlap = set(train_split["category"]) & set(test_split["category"])
    hyponym_overlap = set(train_split["hyponym"]) & set(test_split["hyponym"])
    hypernym_overlap = set(train_split["hypernym"]) & set(test_split["hypernym"])
    
    train_pairs = set(zip(train_split["hyponym"], train_split["hypernym"]))
    test_pairs = set(zip(test_split["hyponym"], test_split["hypernym"]))
    pair_overlap = train_pairs & test_pairs

    print("Category overlap check passed." if len(category_overlap) == 0 else " Category overlap detected.")
    print("Hyponym overlap check passed." if len(hyponym_overlap) == 0 else "Hyponym overlap detected.")
    print("Hypernym overlap check passed." if len(hypernym_overlap) == 0 else "Hypernym overlap detected.")
    print("Pair overlap check passed." if len(pair_overlap) == 0 else "Pair overlap detected.")

    if category_overlap:
        print("Category overlap examples:", list(category_overlap)[:10])
    if hyponym_overlap:
        print("Hyponym overlap examples:", list(hyponym_overlap)[:10])
    if hypernym_overlap:
        print("Hypernym overlap examples:", list(hypernym_overlap)[:10])
    if pair_overlap:
        print("Pair overlap examples:", list(pair_overlap)[:10])

    train_csv_path = os.path.join(output_dir, "split2_train.csv")
    test_csv_path = os.path.join(output_dir, "split2_test.csv")

    train_jsonl_path = os.path.join(output_dir, "split2_train.jsonl")
    test_jsonl_path = os.path.join(output_dir, "split2_test.jsonl")

    train_split.to_csv(train_csv_path, index=False)
    test_split.to_csv(test_csv_path, index=False)

    save_dataframe_as_jsonl(train_split, train_jsonl_path)
    save_dataframe_as_jsonl(test_split, test_jsonl_path)

    # stats
    
    print("\nSplit 2 ok!")
    print(f"Train size: {len(train_split)}")
    print(f"Test size: {len(test_split)}")
    print(f"Output directory: {output_dir}")

    print("\n Train Stats")
    print(f"Unique categories: {train_split['category'].nunique()}")
    print(f"Unique hyponyms: {train_split['hyponym'].nunique()}")
    print(f"Unique hypernyms: {train_split['hypernym'].nunique()}")
    print(
        f"Unique hyponym-hypernym pairs: "
        f"{train_split[['hyponym', 'hypernym']].drop_duplicates().shape[0]}"
    )
    print("\nLabel distribution:")
    print(train_split["gold_label"].value_counts())

    print("\n Test Stats")
    print(f"Unique categories: {test_split['category'].nunique()}")
    print(f"Unique hyponyms: {test_split['hyponym'].nunique()}")
    print(f"Unique hypernyms: {test_split['hypernym'].nunique()}")
    print(
        f"Unique hyponym-hypernym pairs: "
        f"{test_split[['hyponym', 'hypernym']].drop_duplicates().shape[0]}"
    )
    print("\nLabel distribution:")
    print(test_split["gold_label"].value_counts())

    return train_split, test_split





# SPLIT 3 : Flipped Pattern 

def split_3(
    csv_path="../data/more_dataset/all_rules_of_more.csv",
    output_dir="../data/more_splits/split3"
):
    """
    This function creates the flipped pattern split
    Train and test sets are separated by:
    - hyponyms
    - hypernyms
    - rules
    
    Train rules: only the ones that follow the pattern: hypernym [p] hyponym [h]
    R1, R4, R5, R7

    Test rules: only the ones that follow the pattern: hyponym [p] hypernym [h]
    R2, R3, R6, R8
    """

    os.makedirs(output_dir, exist_ok=True)
    dataset = pd.read_csv(csv_path)

    # we employ the same logic as in split 2 : but we keep only the 44 categories as hypernyms and dividing them in half
    dataset = dataset[dataset["hypernym"] == dataset["category"]].copy()

    train_categories = [
        "mammals", "rodents", "felines", "ungulates", "fish", "insects",
        "amphibians", "crustaceans", "dinosaurs", "bacteria", "protozoa",
        "trees", "fungi", "fruit", "furniture", "clothing", "jewellery",
        "toys", "buildings", "electronics", "instruments", "shapes"
    ]

    test_categories = [
        "dogs", "primates", "marsupials", "cetaceans", "birds", "reptiles",
        "arachnids", "mollusks", "microbes", "viruses", "plants", "shrubs",
        "vegetables", "sweets", "shoes", "accessories", "tops", "headwear",
        "shops", "monuments", "appliances", "vehicles"
    ]

    train_rules = ["R1", "R4", "R5", "R7"]
    test_rules = ["R2", "R3", "R6", "R8"]

    train_split = dataset[
        dataset["category"].isin(train_categories)
        & dataset["rule"].isin(train_rules)
    ].reset_index(drop=True)

    test_split = dataset[
        dataset["category"].isin(test_categories)
        & dataset["rule"].isin(test_rules)
    ].reset_index(drop=True)

    
    # sanity checks
    category_overlap = set(train_split["category"]) & set(test_split["category"])
    hyponym_overlap = set(train_split["hyponym"]) & set(test_split["hyponym"])
    hypernym_overlap = set(train_split["hypernym"]) & set(test_split["hypernym"])
    rule_overlap = set(train_split["rule"]) & set(test_split["rule"])

    train_pairs = set(zip(train_split["hyponym"], train_split["hypernym"]))
    test_pairs = set(zip(test_split["hyponym"], test_split["hypernym"]))
    pair_overlap = train_pairs & test_pairs

    print("Category overlap check passed." if not category_overlap else "Category overlap detected.")
    print("Hyponym overlap check passed." if not hyponym_overlap else "Hyponym overlap detected.")
    print("Hypernym overlap check passed." if not hypernym_overlap else "Hypernym overlap detected.")
    print("Pair overlap check passed." if not pair_overlap else "Pair overlap detected.")
    print("Rule overlap check passed." if not rule_overlap else "Rule overlap detected.")


    train_split.to_csv(os.path.join(output_dir, "split3_train.csv"), index=False)
    test_split.to_csv(os.path.join(output_dir, "split3_test.csv"), index=False)

    save_dataframe_as_jsonl(
        train_split,
        os.path.join(output_dir, "split3_train.jsonl")
    )
    save_dataframe_as_jsonl(
        test_split,
        os.path.join(output_dir, "split3_test.jsonl")
    )

    # split 3 stats
    print("\nSplit 3 ok!")
    print(f"Train size: {len(train_split)}")
    print(f"Test size: {len(test_split)}")
    print(f"Output directory: {output_dir}")

    print("\n Train stats")
    print(f"Unique categories: {train_split['category'].nunique()}")
    print(f"Unique hyponyms: {train_split['hyponym'].nunique()}")
    print(f"Unique hypernyms: {train_split['hypernym'].nunique()}")
    print(f"Unique rules: {train_split['rule'].nunique()}")
    print("\nLabel distribution:")
    print(train_split["gold_label"].value_counts())

    print("\n Test stats")
    print(f"Unique categories: {test_split['category'].nunique()}")
    print(f"Unique hyponyms: {test_split['hyponym'].nunique()}")
    print(f"Unique hypernyms: {test_split['hypernym'].nunique()}")
    print(f"Unique rules: {test_split['rule'].nunique()}")
    print("\nLabel distribution:")
    print(test_split["gold_label"].value_counts())

    return train_split, test_split






### diagnostic for split 1 



def split1_diagnostic(
    split1_test_path="../data/more_splits/split1/split1_test.csv",
    output_dir="../data/more_splits/split1_diagnostic"
):
    """
    This function generates a diagnostic for Split 1 by replacing the
    hypernyms in its test distribution with unseen hypernyms.
    """

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(split1_test_path)

    print(f"Original examples: {len(df)}")

    # we take these categories out as they have vague hypernyms
    REMOVE = {
        "plants",
        "trees",
        "shrubs",
        "bushes",
        "fungi",
        "buildings",
        "monuments",
        "landmarks",
        "shops",
        "stores",
        "vehicles",
        "shapes",
        "forms",
    }

    before = len(df)
    df = df[~df["hypernym"].isin(REMOVE)].copy()

    print(f"Removed examples: {before - len(df)}")
    print(f"Remaining examples: {len(df)}")

    new_hypernyms = {

        # organisms
        "animals": "organisms",
        "vertebrates": "organisms",
        "verterbrates": "organisms",
        "veterbrates": "organisms",
        "invertebrates": "organisms",
        "inverterbrates": "organisms",
        "arthropods": "organisms",

        "mammals": "organisms",
        "dogs": "organisms",
        "canines": "organisms",
        "primates": "organisms",
        "rodents": "organisms",
        "felines": "organisms",
        "ungulates": "organisms",
        "marsupials": "organisms",
        "cetaceans": "organisms",
        "birds": "organisms",
        "fish": "organisms",
        "reptiles": "organisms",
        "amphibians": "organisms",
        "insects": "organisms",
        "arachnids": "organisms",
        "crustaceans": "organisms",
        "mollusks": "organisms",
        "dinosaurs": "organisms",
        "protozoa": "organisms",

        # microorganisms
        "bacteria": "microorganisms",
        "microbes": "microorganisms",
        "microorganisms": "microorganisms",
        "viruses": "microorganisms",
        "pathogens": "microorganisms",

        # produce
        "fruit": "produce",
        "vegetables": "produce",

        # food
        "sweets": "food",
        "desserts": "food",

        # objects
        "accessories": "objects",
        "appliances": "objects",
        "clothing": "objects",
        "devices": "objects",
        "electronics": "objects",
        "footwear": "objects",
        "furniture": "objects",
        "garments": "objects",
        "headwear": "objects",
        "instruments": "objects",
        "jewellery": "objects",
        "shoes": "objects",
        "tops": "objects",
        "toys": "objects",
    }
    

    df["original_hypernym"] = df["hypernym"]
    df["hypernym"] = df["hypernym"].replace(new_hypernyms)
    for idx, row in df.iterrows():

        old = row["original_hypernym"]
        new = row["hypernym"]

        premise = row["premise"].replace(old, new)
        hypothesis = row["hypothesis"].replace(old, new)

        premise = premise.replace("objectss", "objects") 
        hypothesis = hypothesis.replace("objectss", "objects")

        premise = premise.replace("organismss", "organisms")
        hypothesis = hypothesis.replace("organismss", "organisms")

        premise = premise.replace("foods", "food")
        hypothesis = hypothesis.replace("foods", "food")

        premise = premise.replace("produces", "produce")
        hypothesis = hypothesis.replace("produces", "produce")

        df.at[idx, "premise"] = premise
        df.at[idx, "hypothesis"] = hypothesis


    df.drop(columns="original_hypernym", inplace=True)

    print("\n-------------")
    print("Sanity check")
    print("---------------")

    print("\nUnique hypernyms:")
    print(sorted(df["hypernym"].unique()))

    expected = {
        "organisms",
        "microorganisms",
        "objects",
        "produce",
        "food"
    }

    remaining = set(df["hypernym"].unique()) - expected

    if len(remaining) == 0:
        print("\n All hypernyms replaced.")
    else:
        print("\n old hypernyms found:")
        print(sorted(remaining))

    remaining_text = []

    for old in new_hypernyms.keys():
        pattern = rf"\b{re.escape(old)}\b"

    if (
        df["premise"].str.contains(pattern, regex=True).any()
        or
        df["hypothesis"].str.contains(pattern, regex=True).any()
    ):
        remaining_text.append(old)
        
    if len(remaining_text) == 0:
        print("\n All old hypernyms out.")
    else:
        print("\n Old hypernyms still in the samples:")
        print(sorted(remaining_text))

    print("\nRule distribution:")
    print(df["rule"].value_counts().sort_index())

    print("\nLabel distribution:")
    print(df["gold_label"].value_counts())

    print("\nHypernym distribution:")
    print(df["hypernym"].value_counts())

    print("\nUnique hypernyms:", df["hypernym"].nunique())
    print("Unique hyponyms:", df["hyponym"].nunique())

    print(
        "Unique hypernym-hyponym pairs:",
        df[["hypernym", "hyponym"]].drop_duplicates().shape[0]
    )

    out_csv = os.path.join(
        output_dir,
        "split1_diagnostic.csv"
    )

    out_jsonl = os.path.join(
        output_dir,
        "split1_diagnostic.jsonl"
    )

    df.to_csv(out_csv, index=False)
    save_dataframe_as_jsonl(df, out_jsonl)

    print("\nDiagnostic for Split 1 test saved.")
    print(out_csv)

    return df

