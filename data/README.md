## File Structure

```text
data/
├── my_data/
│   ├── rules/
│   ├── split1_train.jsonl
│   ├── split1_test.jsonl
│   ├── split1_diagnostic_train.jsonl
│   ├── split1_diagnostic_test.jsonl
│   ├── split2_train.jsonl
│   ├── split2_test.jsonl
│   ├── split3_train.jsonl
│   ├── split3_test.jsonl
│   └── val.jsonl
│
└── snli_1.0/
    ├── snli_1.0_dev.jsonl
    ├── snli_1.0_dev.txt
    ├── snli_1.0_test.jsonl
    ├── snli_1.0_test.txt
    ├── snli_1.0_train.jsonl
    └── snli_1.0_train.txt


The `snli_1.0/` directory is **not included in this repository**. To reproduce the experiments involving SNLI, download the **SNLI 1.0** dataset from the official [SNLI website](https://nlp.stanford.edu/projects/snli/), and place its contents in the`snli_1.0` folder that is inside this directory.
