import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import SimpleCNN
from models.learning_rules import HebbianLearning
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def train_hebbian(model, learning_rule, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = output
            y_true = F.one_hot(target, num_classes=output.shape[1]).float()
            learning_rule(data.view(data.size(0), -1), y_true)

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return test_loss, accuracy

def main():
    batch_size = 64
    epochs = 10
    learning_rate = 1e-3
    alpha = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    learning_rule = HebbianLearning(input_dim=64*16*16, output_dim=10, lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train_loss = train_hebbian(model, learning_rule, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        train_losses.append(train_loss)
        test_accuracies.append(accuracy)

    os.makedirs('models/saved', exist_ok=True)
    torch.save(model.state_dict(), 'models/saved/cnn_hebbian.pth')

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs +1), train_losses, marker='o')
    plt.title('Training Loss (Hebbian)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs +1), test_accuracies, marker='o', color='orange')
    plt.title('Test Accuracy (Hebbian)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('visuals/accuracy_plots/hebbian_training.png')
    plt.show()

if __name__ == '__main__':
    main()
