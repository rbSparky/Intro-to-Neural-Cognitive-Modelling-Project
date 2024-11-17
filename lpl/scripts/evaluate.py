import torch
from torch.utils.data import DataLoader
from models.cnn_model import SimpleCNN
from models.learning_rules import LPLLearning, BcmLearning, HebbianLearning
from torchvision import datasets, transforms
import os
import argparse
from sklearn.metrics import accuracy_score

def load_model(model_path, device):
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def compute_accuracy(model, device, data_loader):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            x_flat, output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds) * 100
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate Models')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--learning_rule', type=str, required=True, choices=['lpl', 'bcm', 'hebbian'], help='Learning rule used')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_model(args.model, device)

    accuracy = compute_accuracy(model, device, test_loader)
    print(f"Test Accuracy for {args.learning_rule.upper()} model: {accuracy:.2f}%")

    os.makedirs('visuals/accuracy_plots', exist_ok=True)
    with open(f'visuals/accuracy_plots/{args.learning_rule}_accuracy.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    main()
