from transformers import TrainingArguments, Trainer
from datasets.dataset_dict import DatasetDict
import os


def train_model(train_ds: DatasetDict,
          validation_ds: DatasetDict,
          data_collator,
          model, 
          output_dir,
          batch_size:int,
          epochs:int=10,
          lr:float=1e-6,
          decay:float=1e-4,
          save_steps:int=200,
          logging_steps:int=50
          ):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=2,
        fp16=True,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=lr,
        weight_decay=decay,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = validation_ds,
        data_collator = data_collator
    )
    def train():
        trainer.train()
        trainer.save_model(output_dir)
        trainer.evaluate()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ""
    train()