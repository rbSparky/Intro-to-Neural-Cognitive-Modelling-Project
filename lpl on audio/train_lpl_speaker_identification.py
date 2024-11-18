# train_lpl_speaker_identification.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import ResNetSpeaker  # Updated model
from models.learning_rules import LPLLearning
from datasets.librispeech_speaker import LibriSpeechSpeaker
import torchaudio.transforms as T
import torchvision.transforms as TT  # Import torchvision.transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch.nn.functional as F
import wandb

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        representation, output = model(data)  # Get representation and output
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')  # Sum to calculate average
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)
            representation, output = model(data)  # Get representation and output
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return test_loss, accuracy

def main():
    # Initialize Weights & Biases
    wandb.init(
        project="Neural_Cognitive_Modeling",
        name="LPL_LibriSpeech_Speaker_Training",
        config={
            "learning_rate": 1e-3,
            "epochs": 50,  # Increased epochs for better learning
            "batch_size": 64,
            "alpha": 0.05,  # Adjusted alpha
            "beta": 0.005,   # Adjusted beta
            "model": "ResNetSpeaker",
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

    # Set random seeds for reproducibility (optional but recommended)
    import random
    import numpy as np

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed()

    # Define the feature extractor
    composed_transform = TT.Compose([
        T.MelSpectrogram(sample_rate=16000, n_mels=64),  # Adjust n_mels as needed
        T.AmplitudeToDB(),
        TT.Resize((224, 224)),  # Resize to match ResNet input dimensions (224x224)
        # Normalize using ImageNet mean and std as ResNet was pretrained on ImageNet
        TT.Normalize(mean=[-80.0], std=[10.0])  # Example values; adjust based on data
    ])

    # Initialize datasets
    train_dataset = LibriSpeechSpeaker(
        root="/content/librispeech-clean/",
        url="train-clean-100",
        download=False,
        transform=composed_transform
    )
    test_dataset = LibriSpeechSpeaker(
        root="/content/librispeech-clean/",
        url="test-clean",
        download=False,
        transform=composed_transform
    )

    # Initialize data loaders
    # Set num_workers=2 to adhere to the system's recommendation
    # Set pin_memory=True for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    num_classes = len(train_dataset.speaker_to_class)
    model = ResNetSpeaker(num_classes=num_classes).to(device)

    # Separate the final layer's parameters
    final_layer_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'fc.weight' in name or 'fc.bias' in name:
            final_layer_params.append(param)
        else:
            other_params.append(param)

    # Initialize optimizer for all parameters except the final layer
    optimizer = optim.Adam(other_params, lr=learning_rate)

    # Initialize LPL Learning Rule with correct input_dim
    input_dim = model.resnet.fc.in_features  # Typically 512 for ResNet-18
    learning_rule = LPLLearning(model=model, device=device, lr=learning_rate, alpha=alpha, beta=beta)

    # Watch the model with Weights & Biases
    wandb.watch(model, log="all")

    train_losses = []
    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        train_losses.append(train_loss)
        test_accuracies.append(accuracy)

        # Apply LPL Learning Rule to update final layer
        with torch.no_grad():
            # Iterate through the entire training set to update weights
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                representation, y_pred = model(data)
                y_true = F.one_hot(target, num_classes=num_classes).float()
                learning_rule.update_weights(representation, y_pred, y_true)

        wandb.log({
            "Epoch": epoch,
            "Training Loss": train_loss,
            "Test Loss": test_loss,
            "Test Accuracy": accuracy
        })

        # Log weight statistics for the final layer
        for name, param in model.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                wandb.log({
                    f"{name}_mean": param.data.mean().item(),
                    f"{name}_std": param.data.std().item()
                })
                wandb.log({f"{name}_hist": wandb.Histogram(param.data.cpu().numpy())})

    # Save the trained model
    os.makedirs('models/saved', exist_ok=True)
    model_path = 'models/saved/resnet_lpl_librispeech_speaker.pth'
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
    plot_path = 'visuals/accuracy_plots/resnet_lpl_librispeech_speaker_training.png'
    plt.savefig(plot_path)
    plt.show()

    wandb.log({"Training and Test Metrics": wandb.Image(plot_path)})

    wandb.finish()

if __name__ == '__main__':
    main()