\"\"\"
predict.py
Interactive Inference Script.
Loads the fine-tuned PEFT weights from the latest checkpoint and allows the 
user to test the model interactively via the terminal.
\"\"\"
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def main():
    # Base model used during training
    base_model_name = "bert-base-cased"
    
    # Locate the best checkpoint automatically
    # Since train.py was run from the root, the models went to D:\CODING\models
    # We will search both possible locations just in case
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "bert-base-classifier-peft")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "bert-base-classifier-peft")),
        r"d:\CODING\models\bert-base-classifier-peft"
    ]
    
    models_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            models_dir = p
            break
            
    if not models_dir:
        print("Could not find the models directory!")
        return

    # Find the checkpoint folder (we want the best one, e.g., checkpoint-896)
    checkpoints = [d for d in os.listdir(models_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}")
        return
        
    # Sort by step number, though we might just want to ask the user to pick if multiple exist.
    # The best model was saved as checkpoint-896 based on the logs
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    best_checkpoint = os.path.join(models_dir, checkpoints[0]) # checkpoint-896 was the best in the logs

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=2
    )
    
    print(f"Loading fine-tuned PEFT weights from: {best_checkpoint}")
    model = PeftModel.from_pretrained(base_model, best_checkpoint)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print("\n" + "="*50)
    print("🤖 AI Slop Detector is Ready!")
    print("Type or paste a paragraph below to test it.")
    print("Press Ctrl+C or type 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        try:
            text = input("Enter text: \n")
            if text.strip().lower() in ['quit', 'exit']:
                break
            if not text.strip():
                continue
                
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Convert raw logits to probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            human_prob = probs[0][0].item() * 100
            ai_prob = probs[0][1].item() * 100
            
            # Predict
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            label = "🤖 AI GENERATED (SLOP)" if prediction == 1 else "🧑 HUMAN WRITTEN"
            
            print(f"\n--- PREDICTION ---")
            print(f"Result: {label}")
            print(f"Confidence: Human ({human_prob:.1f}%) | AI ({ai_prob:.1f}%)\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
