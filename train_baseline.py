import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from utils.data_loader import load_all_tasks
from sklearn.metrics import accuracy_score

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

def train_baseline():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    task_data = load_all_tasks()
    for task_idx, (train_data, test_data) in enumerate(task_data):
        print(f"Training on Task {task_idx + 1}")

        train_data = train_data.map(lambda x: tokenize(x, tokenizer), batched=True)
        test_data = test_data.map(lambda x: tokenize(x, tokenizer), batched=True)

        training_args = TrainingArguments(
            output_dir=f"./baseline_task{task_idx+1}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_dir="./logs",
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()
        print(f"Task {task_idx + 1} Accuracy: {eval_result['eval_accuracy']:.4f}")

if __name__ == "__main__":
    train_baseline()
