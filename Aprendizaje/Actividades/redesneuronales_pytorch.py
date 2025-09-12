import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

# Transformaciones (con normalización)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Modelo Baseline
class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Modelo con Dropout y Regularización L2
class RegularizedNet(nn.Module):
    def __init__(self):
        super(RegularizedNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def train(model, optimizer, run, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        run.log({"train_loss": loss.item()})

def test(model, run, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    run.log({"test_loss": test_loss, "accuracy": accuracy})
    print(f"Epoch {epoch}: Test Loss={test_loss:.4f}, Accuracy={accuracy:.2f}%")

# Entrenamiento para ambos modelos
for model_name, model_class, wd in [
    ("baseline", BaselineNet, 0.0),
    ("regularized", RegularizedNet, 1e-4)
]:
    run = wandb.init(project="mnist-multiclase", name=model_name, group="comparison", reinit=True)
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)
    for epoch in range(1, 6):
        train(model, optimizer, run, epoch)
        test(model, run, epoch)
    run.finish()
