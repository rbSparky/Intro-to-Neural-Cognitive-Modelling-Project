# train_lpl_librispeech.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import SimpleCNN
from models.learning_rules import LPLLearning
from datasets.librispeech_phoneme import LibriSpeechPhoneme
import torchaudio.transforms as T
import torchvision.transforms as TT  # Import torchvision.transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch.nn.functional as F
import wandb

def train_lpl(model, learning_rule, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x_flat, output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = output
            y_true = F.one_hot(target, num_classes=output.shape[1]).float()
            learning_rule(x_flat, y_pred, y_true)

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            x_flat, output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return test_loss, accuracy

def main():
    wandb.init(
        project="Neural_Cognitive_Modeling",
        name="LPL_LibriSpeech_Training",
        config={
            "learning_rate": 1e-3,
            "epochs": 20,  # Increased epochs for better convergence
            "batch_size": 64,
            "alpha": 0.1,
            "beta": 0.01,
            "model": "SimpleCNN",
            "learning_rule": "LPL",
        }
    )
    config = wandb.config

    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    alpha = config.alpha
    beta = config.beta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the feature extractor
    composed_transform = TT.Compose([
        T.MelSpectrogram(sample_rate=16000, n_mels=64),  # Reduced n_mels to 64
        T.AmplitudeToDB(),
        TT.Resize((64, 64))  # Resize to match CNN input dimensions
    ])

    # Initialize datasets
    train_dataset = LibriSpeechPhoneme(
        root="/content/librispeech-clean/",
        url="train-clean-100",
        download=False,
        transform=composed_transform
    )
    test_dataset = LibriSpeechPhoneme(
        root="/content/librispeech-clean/",
        url="test-clean",
        download=False,
        transform=composed_transform
    )

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = SimpleCNN(input_channels=1, num_classes=41).to(device)
    
    # Calculate input_dim for LPLLearning based on CNN architecture
    input_dim = 64 * 16 * 16  # 64 channels * 16 height * 16 width
    learning_rule = LPLLearning(input_dim=input_dim, output_dim=41, lr=learning_rate, alpha=alpha, beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, log="all")

    train_losses = []
    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train_loss = train_lpl(model, learning_rule, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        train_losses.append(train_loss)
        test_accuracies.append(accuracy)

        wandb.log({
            "Epoch": epoch,
            "Training Loss": train_loss,
            "Test Loss": test_loss,
            "Test Accuracy": accuracy
        })

    # Save the trained model
    os.makedirs('models/saved', exist_ok=True)
    model_path = 'models/saved/cnn_lpl_librispeech.pth'
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    # Plot training loss and test accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs +1), train_losses, marker='o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs +1), test_accuracies, marker='o', color='orange', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    os.makedirs('visuals/accuracy_plots', exist_ok=True)
    plot_path = 'visuals/accuracy_plots/lpl_librispeech_training.png'
    plt.savefig(plot_path)
    plt.show()

    wandb.log({"Training and Test Metrics": wandb.Image(plot_path)})

    wandb.finish()

if __name__ == '__main__':
    main()
