import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

class LiverDataset:
    def __init__(self, filename):
        self.data = []
        
        # Load the training data we created
        with open(filename, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} training examples")
    
    def format_for_training(self, tokenizer):
        """Format data for GPT training"""
        formatted_data = []
        
        for example in self.data:
            # Create training text
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}<|endoftext|>"
            formatted_data.append(text)
        
        # Tokenize all texts
        tokenized = tokenizer(
            formatted_data,
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None
        )
        
        return Dataset.from_dict(tokenized)

def main():
    print("Starting fine-tuning process...")
    
    # Model settings
    model_name = "microsoft/DialoGPT-medium"  # Using a smaller model that works better
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,  # Use 4-bit to save memory
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration - this is the key to efficient training
    lora_config = LoraConfig(
        r=16,                 # Rank
        lora_alpha=32,        # Alpha parameter
        target_modules=["c_attn"],  # Target modules for DialoGPT
        lora_dropout=0.1,     # Dropout
        bias="none",          # Bias
        task_type="CAUSAL_LM" # Task type
    )
    
    model = get_peft_model(model, lora_config)
    
    print("Loading training data...")
    dataset_loader = LiverDataset('training_data.jsonl')
    train_dataset = dataset_loader.format_for_training(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./liver-model-output',
        overwrite_output_dir=True,
        num_train_epochs=2,              # Few epochs to start
        per_device_train_batch_size=1,   # Small batch size
        gradient_accumulation_steps=4,    # Accumulate gradients
        warmup_steps=50,
        logging_steps=10,
        save_steps=200,
        save_strategy="steps",
        evaluation_strategy="no",
        learning_rate=2e-4,
        fp16=True,                       # Mixed precision
        dataloader_drop_last=True,
        gradient_checkpointing=True,     # Save memory
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model('./liver-model-final')
    tokenizer.save_pretrained('./liver-model-final')
    
    print("Fine-tuning complete!")
    print("Model saved to ./liver-model-final")

if __name__ == "__main__":
    main()
