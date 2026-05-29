"""
eval_raid.py
Model Evaluation Module.
Evaluates the fine-tuned PEFT model against the industry-standard RAID benchmark 
(Robust AI Detection) to calculate True Positive Rate at a 1% False Positive Rate.
"""
import os
import torch
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import roc_curve

def get_tpr_at_fpr(y_true, y_probs, target_fpr=0.01):
    # If there are no positive or negative samples, return NaN
    if len(np.unique(y_true)) < 2:
        return float('nan')
        
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Find the maximum TPR where FPR <= target_fpr
    valid_idx = fpr <= target_fpr
    if not any(valid_idx):
        return 0.0
    return tpr[valid_idx][-1]

def main():
    base_model_name = "bert-base-cased"
    
    # Locate the best checkpoint automatically
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

    checkpoints = [d for d in os.listdir(models_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        print(f"No checkpoints found in {models_dir}")
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
    
    # Use CPU by default if no CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load RAID dataset stream
    print("Loading RAID dataset (streaming)...")
    ds = load_dataset('liamdugan/raid', split='train', streaming=True)
    
    # We want a mixed sample. We'll grab 2000 total: roughly 1000 human and 1000 AI.
    # To get domain diversity from a stream, we shuffle the buffer.
    ds = ds.shuffle(buffer_size=10000, seed=42)
    
    samples = []
    human_count = 0
    ai_count = 0
    target_each = 1000
    
    print("Collecting 2,000 samples...")
    for row in ds:
        # Avoid adversarial attacks for the baseline evaluation if possible, or just include them
        # The RAID benchmark tests adversarial separately, but let's just take all
        
        is_human = (row['model'] == 'human')
        
        if is_human and human_count < target_each:
            samples.append(row)
            human_count += 1
        elif not is_human and ai_count < target_each:
            samples.append(row)
            ai_count += 1
            
        if human_count == target_each and ai_count == target_each:
            break

    print(f"Collected {len(samples)} samples. Running inference...")
    
    results = []
    
    # Run Inference
    batch_size = 16
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i+batch_size]
        texts = [row['generation'] for row in batch]
        
        inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_probs = probs[:, 1].cpu().numpy()
            
        for row, prob in zip(batch, ai_probs):
            results.append({
                "model": row['model'],
                "domain": row['domain'],
                "attack": row['attack'],
                "y_true": 0 if row['model'] == 'human' else 1,
                "y_prob": float(prob)
            })

    # Calculate metrics
    y_true = np.array([r['y_true'] for r in results])
    y_prob = np.array([r['y_prob'] for r in results])
    
    global_tpr = get_tpr_at_fpr(y_true, y_prob, 0.01)
    
    print("\n" + "="*50)
    print("RAID BENCHMARK RESULTS (2,000 Samples)")
    print("="*50)
    print(f"Global TPR@FPR=1%: {global_tpr*100:.1f}%\n")
    
    print("--- By Domain ---")
    domains = set(r['domain'] for r in results)
    for d in domains:
        d_true = np.array([r['y_true'] for r in results if r['domain'] == d])
        d_prob = np.array([r['y_prob'] for r in results if r['domain'] == d])
        tpr = get_tpr_at_fpr(d_true, d_prob, 0.01)
        count = len(d_true)
        if not np.isnan(tpr):
            print(f"{d.capitalize()}: {tpr*100:.1f}% (N={count})")
            
    print("\n--- By Generator (Model) ---")
    generators = set(r['model'] for r in results if r['model'] != 'human')
    # For generator slicing, we need negative samples (human). 
    # Usually we compare generator X vs ALL human samples.
    all_human_results = [r for r in results if r['y_true'] == 0]
    
    for gen in generators:
        gen_results = [r for r in results if r['model'] == gen]
        subset = all_human_results + gen_results
        
        g_true = np.array([r['y_true'] for r in subset])
        g_prob = np.array([r['y_prob'] for r in subset])
        tpr = get_tpr_at_fpr(g_true, g_prob, 0.01)
        if not np.isnan(tpr):
            print(f"{gen}: {tpr*100:.1f}% (N={len(gen_results)} AIs)")
            
    print("="*50)

    # Save results
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raid"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)
    print(f"Saved full predictions to {out_dir}/results.json")

if __name__ == "__main__":
    main()
