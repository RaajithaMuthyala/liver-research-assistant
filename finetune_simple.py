import torch
import json
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import os

def simple_fine_tune():
    print("Starting SIMPLE fine-tuning process...")
    print("This version avoids complex trainer issues")
    
    # Load training data
    training_data = []
    print("Loading training data...")
    
    with open('training_data.jsonl', 'r') as f:
        for line in f:
            training_data.append(json.loads(line))
    
    print(f"Loaded {len(training_data)} training examples")
    
    # Use a simpler, more compatible model
    model_name = "gpt2-medium"
    print(f"Loading {model_name}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    model.train()
    total_loss = 0
    num_epochs = 2
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        # Process in small batches to avoid memory issues
        batch_size = 4
        num_batches = len(training_data) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(training_data))
            batch = training_data[start_idx:end_idx]
            
            # Prepare batch texts
            batch_texts = []
            for example in batch:
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}<|endoftext|>"
                batch_texts.append(text)
            
            # Tokenize
            try:
                encoding = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,  # Shorter to avoid memory issues
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Print progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Batch {batch_idx + 1}/{num_batches}, Avg Loss: {avg_loss:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Skipping batch {batch_idx} due to memory constraints")
                    torch.cuda.empty_cache()  # Clear GPU memory
                    continue
                else:
                    raise e
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Save the model
    print("Saving fine-tuned model...")
    
    # Create output directory
    output_dir = './liver-model-final'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    print("Fine-tuning complete!")
    
    # Test the model quickly
    print("\nTesting the fine-tuned model...")
    test_question = "How do social determinants affect liver disease?"
    
    model.eval()
    with torch.no_grad():
        prompt = f"### Instruction:\nAnswer the research question about liver disease and social determinants of health based on scientific evidence.\n\n### Input:\n{test_question}\n\n### Response:\n"
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        if "### Response:" in response:
            answer = response.split("### Response:")[-1].strip()
            print(f"Test Question: {test_question}")
            print(f"Model Answer: {answer}")
        else:
            print("Model generated response but format unclear")
    
    print("All done! Your model is ready.")

if __name__ == "__main__":
    simple_fine_tune()
