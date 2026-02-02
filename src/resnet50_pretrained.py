import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# =====================
# Configurações
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 10
learning_rate = 1e-3
num_classes = 10

print("===================================")
print("Inicializando treinamento")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")
print("===================================")

# =====================
# Transforms
# =====================
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# Datasets
# =====================
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

print(f"Tamanho do dataset de treino: {len(train_dataset)}")
print(f"Tamanho do dataset de teste: {len(test_dataset)}")

# =====================
# DataLoaders (Windows)
# =====================
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# =====================
# Modelo
# =====================
model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V1
)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =====================
# Treino
# =====================
def train_one_epoch(model, loader, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch+1} [Treino]",
        leave=False
    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.2f}%"
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# =====================
# Avaliação
# =====================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Validação",
        leave=False
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            acc = 100.0 * correct / total
            progress_bar.set_postfix({
                "acc": f"{acc:.2f}%"
            })

    return 100.0 * correct / total

# =====================
# Main
# =====================
def main():
    print("Entrou na função main...")

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, epoch
        )
        test_acc = evaluate(model, test_loader)

        print(
            f"Resumo Epoch {epoch+1}: "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    print("\nTreinamento finalizado.")

if __name__ == "__main__":
    main()
