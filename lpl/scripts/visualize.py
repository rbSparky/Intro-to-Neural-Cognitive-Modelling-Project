import torch
from torch.utils.data import DataLoader
from models.cnn_model import SimpleCNN
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os
import argparse
import numpy as np

def load_model(model_path, device):
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_representations(model, device, data_loader):
    representations = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            x_flat, output = model(data)
            representations.append(x_flat.cpu().numpy())
            labels.extend(target.cpu().numpy())
    representations = np.concatenate(representations, axis=0)
    labels = np.array(labels)
    return representations, labels

def plot_tsne(representations, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(representations)

    plt.figure(figsize=(10,8))
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels, palette="tab10", legend='full', alpha=0.6)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_pca(representations, labels, title, save_path):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(representations)

    plt.figure(figsize=(10,8))
    sns.scatterplot(x=pca_results[:,0], y=pca_results[:,1], hue=labels, palette="tab10", legend='full', alpha=0.6)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Model Representations')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--learning_rule', type=str, required=True, choices=['lpl', 'bcm', 'hebbian'], help='Learning rule used')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for representation extraction')
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

    representations, labels = get_representations(model, device, test_loader)

    os.makedirs('visuals/representations', exist_ok=True)
    tsne_save_path = f'visuals/representations/{args.learning_rule}_tsne.png'
    plot_tsne(representations, labels, f't-SNE Representation ({args.learning_rule.upper()})', tsne_save_path)
    print(f"t-SNE plot saved to {tsne_save_path}")

    pca_save_path = f'visuals/representations/{args.learning_rule}_pca.png'
    plot_pca(representations, labels, f'PCA Representation ({args.learning_rule.upper()})', pca_save_path)
    print(f"PCA plot saved to {pca_save_path}")

if __name__ == "__main__":
    main()
