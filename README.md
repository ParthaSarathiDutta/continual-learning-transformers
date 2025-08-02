# Continual Learning with Transformers (EWC + Replay)

This project demonstrates how to perform continual learning on a sequence of NLP classification tasks using RoBERTa. It implements and compares three strategies:

- Baseline (no mitigation for forgetting)
- EWC (Elastic Weight Consolidation)
- Replay Buffer


---

## What is Continual Learning?

Continual learning is the ability to train a model on new tasks without forgetting previous ones. Standard neural networks suffer from **catastrophic forgetting**, where performance on older tasks drops as new tasks are learned.

This repo tackles that with two popular strategies:

| Technique | Description |
|----------|-------------|
| EWC      | Penalizes changes to important weights for previous tasks |
| Replay   | Reuses a buffer of old examples when training on new tasks |

---

## Task Setup

We train the model on a sequence of text classification tasks:

1. IMDb → Sentiment classification (positive/negative)
2. AG News → Topic classification (e.g., World, Sports, Business)
3. SNIPS → Intent classification (e.g., play music, get weather)

These tasks represent diverse real-world NLP challenges.

---

## Directory Structure

```
continual-learning-transformers/
├── data/                  # CSVs for IMDb, AG News, SNIPS (replaceable with full datasets)
├── model/                 # (Reserved for future use)
├── plots/                 # Output evaluation plots (accuracy over time)
├── utils/
│   ├── data_loader.py     # Loads tasks from CSV or Hugging Face datasets
│   ├── ewc.py             # EWC implementation
│   └── replay_buffer.py   # Replay buffer implementation
├── train_baseline.py      # No forgetting mitigation
├── train_ewc.py           # With EWC penalty
├── train_replay.py        # With replay buffer
├── requirements.txt       # Required Python libraries
├── config.json            # Training hyperparameters
└── README.md              # This file
```

---

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run any of the three learning approaches:

```bash
python train_baseline.py
python train_ewc.py
python train_replay.py
```

---

## Results and Evaluation

Each script prints task-wise accuracy. You can add evaluation plots in the `plots/` folder (coming soon).

---

## Data Notes

You can either:
- Use the small CSV files in /data/ for quick tests
- Or modify `utils/data_loader.py` to load full datasets from Hugging Face like so:

```python
from datasets import load_dataset
ag_news = load_dataset("ag_news", split="train")
```


---



## License

MIT License. Use freely with attribution.
