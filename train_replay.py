import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from utils.data_loader import load_all_tasks
from utils.replay_buffer import ReplayBuffer
from sklearn.metrics import accuracy_score
from datasets import Dataset

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

def train_replay():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    task_data = load_all_tasks()
    buffer = ReplayBuffer(max_size=20)

    for task_idx, (train_data, test_data) in enumerate(task_data):
        print(f"Training on Task {task_idx + 1}")

        if len(buffer.buffer) > 0:
            replay_samples = buffer.sample(10)
            train_data = train_data.add_item(replay_samples)

        train_data = train_data.map(lambda x: tokenize(x, tokenizer), batched=True)
        test_data = test_data.map(lambda x: tokenize(x, tokenizer), batched=True)

        training_args = TrainingArguments(
            output_dir=f"./replay_task{task_idx+1}",
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
        print(f"Replay Task {task_idx + 1} Accuracy: {eval_result['eval_accuracy']:.4f}")

        buffer.add_batch(train_data)

if __name__ == "__main__":
    train_replay()
