import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Suppress wandb and warnings
os.environ["WANDB_DISABLED"] = "true"
import warnings
warnings.filterwarnings("ignore")

# Define paths
DATA_DIR = r"D:\eOCR\full_training_dataset"
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
MODEL_PATH = r"D:\eOCR\trocr-small"
OUTPUT_DIR = r"D:\eOCR\trained_model"

class CADDrawingDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=64):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file name and text label
        file_name = self.df.iloc[idx, 0]
        text = str(self.df.iloc[idx, 1]).strip()
        if text == "nan":
            text = ""

        # Prepare image
        image_path = os.path.join(self.root_dir, "images", file_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images
            image = Image.new('RGB', (100, 32), color = (255, 255, 255))
            text = ""

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        # Important: Replace padding token id with -100 to ignore it in loss calculation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

def main():
    print("Loading dataset metadata...")
    df = pd.read_csv(CSV_PATH)
    
    # Shuffle and split dataset (90% train, 10% eval)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    eval_df = df.iloc[split_idx:]
    
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Evaluation samples: {len(eval_df)}")

    print(f"Loading local TrOCR model from {MODEL_PATH}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

    # Model configuration for training
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 64
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Remove them from main config to prevent save_pretrained crash
    for key in ['max_length', 'early_stopping', 'num_beams', 'length_penalty', 'no_repeat_ngram_size']:
        if hasattr(model.config, key):
            delattr(model.config, key)

    print("Preparing PyTorch datasets...")
    train_dataset = CADDrawingDataset(root_dir=DATA_DIR, df=train_df, processor=processor)
    eval_dataset = CADDrawingDataset(root_dir=DATA_DIR, df=eval_df, processor=processor)

    # Training arguments optimized for RTX 3050 (4GB VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        fp16=True, # USE FP16 for RTX 3050 (saves memory and is 3x faster)
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none"
    )

    print("Initializing HuggingFace Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Starting Fine-Tuning on GPU...")
    trainer.train()

    print(f"Training complete! Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
