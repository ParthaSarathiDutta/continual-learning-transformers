import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from utils.data_loader import load_all_tasks
from utils.ewc import EWC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

def train_ewc():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.train()

    task_data = load_all_tasks()
    ewc_obj = None
    lambda_ewc = 0.4

    for task_idx, (train_data, test_data) in enumerate(task_data):
        print(f"Training on Task {task_idx + 1}")

        train_data = train_data.map(lambda x: tokenize(x, tokenizer), batched=True)
        test_data = test_data.map(lambda x: tokenize(x, tokenizer), batched=True)

        if task_idx > 0:
            ewc_loss = ewc_obj.penalty(model)
        else:
            ewc_loss = 0

        def compute_loss(model, inputs, return_outputs=False):
            outputs = model(**inputs)
            ce_loss = outputs.loss
            loss = ce_loss + (lambda_ewc * ewc_loss if task_idx > 0 else 0)
            return (loss, outputs) if return_outputs else loss

        training_args = TrainingArguments(
            output_dir=f"./ewc_task{task_idx+1}",
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
            compute_loss=compute_loss,
        )

        trainer.train()
        eval_result = trainer.evaluate()
        print(f"EWC Task {task_idx + 1} Accuracy: {eval_result['eval_accuracy']:.4f}")

        dl = DataLoader(train_data.remove_columns(['text']), batch_size=8)
        ewc_obj = EWC(model, dl)

if __name__ == "__main__":
    train_ewc()
