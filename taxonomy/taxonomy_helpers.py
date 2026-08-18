import random 
from taxonomy import CATEGORY_CONFIG


def load_pairs():
    """
    This function generates all hyponym–hypernym pairs from CATEGORY_CONFIG.
    Returns :
        A list of dictionaries. Each dictionary contains:
        - item: hyponym
        - hypernym: corresponding hypernym
        - category: semantic category
        - subcategory: semantic subcategory
    """
    pairs = []

    for category, cfg in CATEGORY_CONFIG.items():
        for item in cfg["examples"]:
            for hyper in cfg["hypernyms"]:
                pairs.append({
                    "item": item,
                    "hypernym": hyper,
                    "category": category.lower(),
                    "subcategory": category.lower()
                })

    return pairs



def category_counts(CATEGORY_CONFIG):
    """
    counts all the examples per category present in the taxonomy
    
    """
    print("Category counts:\n")
    for cat, cfg in CATEGORY_CONFIG.items():
        n = len(cfg["examples"])
        print(f"{cat:<15} -> {n} examples")

        

def category_examples(CATEGORY_CONFIG, category, n=5):
    """
    illustrateS some examples per category
    """
    if category not in CATEGORY_CONFIG:
        print(f"category '{category}' missing.")
        return
    ex = random.sample(CATEGORY_CONFIG[category]["examples"], min(n, len(CATEGORY_CONFIG[category]["examples"])))
    print(f"\n{category} random examples ({len(ex)} examples):")
    print(", ".join(ex))

    

def hypo_hyper_pairs(CATEGORY_CONFIG, category, n=2):
    """
    It shows the hypernym-hyponym pairs of a given category : 
    e.g. mastiff < dogs , mastiff < canines , mastiff < mammals etc.
    """
    if category not in CATEGORY_CONFIG:
        print(f"Category '{category}' not found.")
        return
    examples = random.sample(CATEGORY_CONFIG[category]["examples"], min(n, len(CATEGORY_CONFIG[category]["examples"])))
    hypers = CATEGORY_CONFIG[category]["hypernyms"]
    
    print(f"\nHyponym → Hypernym pairs for {category}:")
    for e in examples:
        for h in hypers:
            print(f"  {e} -> {h}")


            

def total_pairs(CATEGORY_CONFIG):
    """
    Calculates the total number of hyponym–hypernym pairs per category
    """
    total = 0
    details = {}
    for cat, cfg in CATEGORY_CONFIG.items():
        num_hypos = len(cfg["examples"])
        num_hypers = len(cfg["hypernyms"])
        count = num_hypos * num_hypers
        details[cat] = count
        total += count
    print(f"\nTotal hyponym-hypernym pairs across all 44 categories: {total}")
    return details



def hierarchy_breakdown(CATEGORY_CONFIG, n_examples = 5):
    """
    A summry of the hierarchy, showing the number of examples/category, the number of hypernyms as well as some examples.
    """
    print("Taxonomy hierarchy with counts:\n")
    for cat, cfg in CATEGORY_CONFIG.items():
        n = len(cfg["examples"])
        hypers = cfg["hypernyms"]
        count = n * len(hypers)
        preview = ", ".join(cfg["examples"][:n_examples])
        print(f"- {cat} ({n} examples, {len(hypers)} hypernyms → {count} pairs)")
        print(f"   Examples: {preview} ...")
        print(f"   Hypernyms: {', '.join(hypers)}\n")



        
