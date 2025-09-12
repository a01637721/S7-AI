# ============================================================
# MNIST en PyTorch + Weights & Biases (W&B)
# Comparación: Baseline, Dropout y L2 (weight decay)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb

# --------------------------
# 0. Inicializar W&B
# --------------------------
wandb.login()

# ⚙️ Configura aquí tu experimento:
wandb.init(
    project="mnist-ajuste",
    name="baseline_con_escalado",  # cambia por "baseline", "dropout", "l2", etc.
    config={
        "dropout": 0.1,        # 0.0 para baseline
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 10,
        "weight_decay": 0.0  # 0.0 para baseline o solo dropout
    }
)

config = wandb.config

# --------------------------
# 1. Preparar dataset
# --------------------------
transform = transforms.ToTensor()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --------------------------
# 2. Definir red neuronal
# --------------------------
class Net(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(dropout_rate=config.dropout).to(device)

# --------------------------
# 3. Definir optimizador y loss
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay  # 👈 regularización L2
)

# --------------------------
# 4. Función para evaluar
# --------------------------
def evaluate(model, loader, mode="eval"):
    if mode == "eval":
        model.eval()
    else:
        model.train()

    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    loss_avg = loss_total / len(loader)
    return loss_avg, acc

# --------------------------
# 5. Entrenamiento
# --------------------------
train_acc_dropout = []
train_acc_eval = []
val_acc = []

for epoch in range(config.epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluaciones
    train_loss_d, train_acc_d = evaluate(model, train_loader, mode="train")  # con dropout
    train_loss_e, train_acc_e = evaluate(model, train_loader, mode="eval")   # sin dropout
    val_loss, val_acc_epoch = evaluate(model, test_loader, mode="eval")      # validación

    train_acc_dropout.append(train_acc_d)
    train_acc_eval.append(train_acc_e)
    val_acc.append(val_acc_epoch)

    # Log en W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss_dropout": train_loss_d,
        "train_acc_dropout": train_acc_d,
        "train_loss_eval": train_loss_e,
        "train_acc_eval": train_acc_e,
        "val_loss": val_loss,
        "val_acc": val_acc_epoch
    })

    print(f"Epoch {epoch+1}: Train acc (dropout)={train_acc_d:.4f}, "
          f"Train acc (eval)={train_acc_e:.4f}, Val acc={val_acc_epoch:.4f}")

wandb.finish()

# --------------------------
# 6. Graficar localmente
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc_dropout, label="Train (con dropout)")
plt.plot(train_acc_eval, label="Train (sin dropout)")
plt.plot(val_acc, label="Validación")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title(f"MNIST - Dropout={config.dropout}, L2={config.weight_decay}")
plt.legend()
plt.show()