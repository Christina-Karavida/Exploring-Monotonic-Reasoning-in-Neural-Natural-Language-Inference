# gpt_args.py

class GPTArgs:
    def __init__(
        self,
        model_name="gpt2",
        learning_rate=5e-5,
        epochs=3,
        batch_size=16,
        eval_batch_size=32,
        max_tokens=100,
        manual_prepend_bos=True,
        seed=42,
        exp_name="gpt_classifier",
        wandb_name="gpt_classifier",
        use_wandbid_name=True,
        update_label_only=True,
        gradient_accumulation_steps=4,
        num_workers=2,
        eval_every=1,
        use_hooked_transform=True,
        save_model=False,
        dataset=None,
        paths=None
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_tokens = max_tokens
        self.manual_prepend_bos = manual_prepend_bos
        self.seed = seed
        self.exp_name = exp_name
        self.wandb_name = wandb_name
        self.use_wandbid_name = use_wandbid_name
        self.update_label_only = update_label_only
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.eval_every = eval_every
        self.use_hooked_transform = use_hooked_transform
        self.save_model = save_model
        self.dataset = dataset
        self.paths = paths or {
            'output_dir': "./results/",
            'model_save_path': "gpt2_classifier.pth",
            'results_save_path': "more_split1_classification_results.json"
        }
