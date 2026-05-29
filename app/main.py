"""
main.py
FastAPI backend for SlopRadar.
Loads the fine-tuned PEFT BERT model into memory on startup and exposes a 
/predict endpoint to classify text as Human or AI generated.
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

app = FastAPI(title="SlopRadar API")

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class PredictRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global tokenizer, model
    base_model_name = "bert-base-cased"
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    possible_paths = [
        os.path.join(root_dir, "..", "models", "bert-base-classifier-peft"),
        os.path.join(root_dir, "models", "bert-base-classifier-peft"),
        r"d:\CODING\models\bert-base-classifier-peft"
    ]
    
    models_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            models_dir = p
            break
            
    if not models_dir:
        print("WARNING: Could not find the models directory! Startup will fail.")
        return

    checkpoints = [d for d in os.listdir(models_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print(f"WARNING: No checkpoints found in {models_dir}")
        return
        
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    best_checkpoint = os.path.join(models_dir, checkpoints[0]) 

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=2
    )
    
    print(f"Loading fine-tuned PEFT weights from: {best_checkpoint}")
    model = PeftModel.from_pretrained(base_model, best_checkpoint)
    model.to(device)
    model.eval()
    print("Model successfully loaded!")

@app.post("/api/predict")
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")

    inputs = tokenizer(request.text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        human_prob = float(probs[0][0].item() * 100)
        ai_prob = float(probs[0][1].item() * 100)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        
    return {
        "is_ai": bool(prediction == 1),
        "ai_probability": ai_prob,
        "human_probability": human_prob
    }

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))
