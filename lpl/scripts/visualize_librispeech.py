import torch
from torch.utils.data import DataLoader
from models.cnn_model import SimpleCNN
from datasets.librispeech_phoneme import LibriSpeechPhoneme
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os
import argparse
import numpy as np

def load_model(model_path, device):
    model = SimpleCNN(input_channels=1, num_classes=41).to(device)
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
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(representations)

    plt.figure(figsize=(10,8))
    palette = sns.color_palette("hsv", 41)  
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels, palette=palette, legend='full', alpha=0.6, s=10)
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title='Phoneme')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca(representations, labels, title, save_path):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(representations)

    plt.figure(figsize=(10,8))
    palette = sns.color_palette("hsv", 41)  
    sns.scatterplot(x=pca_results[:,0], y=pca_results[:,1], hue=labels, palette=palette, legend='full', alpha=0.6, s=10)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title='Phoneme')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Model Representations for LibriSpeech')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--url', type=str, default='test-clean', help='Dataset subset to use (e.g., train-clean-100, test-clean)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = T.MelSpectrogram(sample_rate=16000, n_mels=128)
    normalization = T.AmplitudeToDB()

    composed_transform = T.Compose([
        transform,
        normalization,
        T.Resize((64, 64)) 
    ])

    dataset = LibriSpeechPhoneme(
        root="./librispeech-clean",
        url=args.url,
        download=False,
        transform=composed_transform
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(args.model, device)

    print("Extracting representations...")
    representations, labels = get_representations(model, device, data_loader)
    print("Extraction complete.")

    os.makedirs('visuals/representations', exist_ok=True)
    tsne_save_path = f'visuals/representations/tsne_librispeech_{args.url}.png'
    plot_tsne(representations, labels, f't-SNE Representation ({args.url})', tsne_save_path)
    print(f"t-SNE plot saved to {tsne_save_path}")

    pca_save_path = f'visuals/representations/pca_librispeech_{args.url}.png'
    plot_pca(representations, labels, f'PCA Representation ({args.url})', pca_save_path)
    print(f"PCA plot saved to {pca_save_path}")

if __name__ == "__main__":
    main()
