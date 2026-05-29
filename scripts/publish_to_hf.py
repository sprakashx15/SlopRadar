\"\"\"
publish_to_hf.py
Deployment Script.
Automates the process of authenticating with Hugging Face and publishing both 
the 16,000-row dataset and the fine-tuned PEFT model weights to the Hub.
\"\"\"
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def main():
    # Load environment variables
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(env_path)
    
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    
    if not hf_token or not hf_username:
        print("❌ ERROR: Missing HF_TOKEN or HF_USERNAME in .env file.")
        print("Please add them to your .env file in the root folder like this:")
        print("HF_TOKEN=hf_your_token_here")
        print("HF_USERNAME=your_username_here")
        sys.exit(1)
        
    print("🔑 Authenticating with Hugging Face...")
    login(token=hf_token)
    
    # Define repository names
    dataset_repo = f"{hf_username}/ai-slop-dataset"
    model_repo = f"{hf_username}/bert-ai-slop-detector"
    
    # ---------------------------------------------------------
    # 1. Publish Dataset
    # ---------------------------------------------------------
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset.jsonl"))
    if os.path.exists(dataset_path):
        print(f"\n🚀 Pushing Dataset to Hub: {dataset_repo}")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        dataset.push_to_hub(dataset_repo)
        print("✅ Dataset successfully published!")
    else:
        print(f"\n⚠️ WARNING: Could not find dataset at {dataset_path}. Skipping dataset publish.")

    # ---------------------------------------------------------
    # 2. Publish Model
    # ---------------------------------------------------------
    print(f"\n🚀 Pushing Model to Hub: {model_repo}")
    
    base_model_name = "bert-base-cased"
    
    # Find best checkpoint
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "bert-base-classifier-peft")),
        r"d:\CODING\models\bert-base-classifier-peft"
    ]
    
    models_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            models_dir = p
            break
            
    if not models_dir:
        print("❌ ERROR: Could not find the trained models directory!")
        sys.exit(1)

    checkpoints = [d for d in os.listdir(models_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print(f"❌ ERROR: No checkpoints found in {models_dir}")
        sys.exit(1)
        
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    best_checkpoint = os.path.join(models_dir, checkpoints[0]) 

    print(f"Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading Base Model and PEFT Weights from {best_checkpoint}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, best_checkpoint)

    # Push to hub
    print("Uploading Model Weights (this may take a few minutes)...")
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(model_repo)
    
    print("\n🎉 SUCCESS! Your project is now live on Hugging Face!")
    print(f"Dataset URL: https://huggingface.co/datasets/{dataset_repo}")
    print(f"Model URL: https://huggingface.co/{model_repo}")

if __name__ == "__main__":
    main()
