\"\"\"
vis_train.py
Visualization Module.
Parses the `trainer_state.json` from the Hugging Face Trainer output and 
generates matplotlib graphs showing training loss and evaluation accuracy over time.
\"\"\"
import os
import json
import matplotlib.pyplot as plt

# Dynamic path resolution to find the models folder
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
    exit()

checkpoints = [d for d in os.listdir(models_dir) if d.startswith("checkpoint-")]
checkpoints.sort(key=lambda x: int(x.split("-")[1]))
checkpoint = os.path.join(models_dir, checkpoints[0])

with open(os.path.join(checkpoint, "trainer_state.json"), "r") as file:
    trainer_state = json.load(file)

history = trainer_state["log_history"]

train_losses = []
train_epochs = []

test_losses = []
test_accuracies = []
test_epochs = []

for item in history:
    if "eval_loss" in item.keys():
        test_losses.append(item["eval_loss"])
        test_accuracies.append(item["eval_accuracy"])
        test_epochs.append(item["epoch"])
    elif "loss" in item.keys():
        train_losses.append(item["loss"])
        train_epochs.append(item["epoch"])

plt.plot(train_epochs, train_losses, label="Train")
plt.plot(test_epochs, test_losses, label="Test")
plt.show()


plt.plot(test_epochs, test_accuracies, label="Test")
plt.show()