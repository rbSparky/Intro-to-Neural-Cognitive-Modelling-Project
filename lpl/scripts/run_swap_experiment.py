import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.cnn_model import SimpleCNN
from models.learning_rules import LPLLearning, BcmLearning, HebbianLearning
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import seaborn as sns
from tqdm import tqdm

class SwapNonswapSequenceDataset(Dataset):
    def __init__(self, sequences_dir, condition='swap', seq_length=20, transform=None, p_digit=0, n_digit=1):
        """
        Args:
            sequences_dir (str): Directory with all the sequences.
            condition (str): 'swap' or 'nonswap'.
            seq_length (int): Number of frames per sequence.
            transform (callable, optional): Optional transform to be applied on a frame.
            p_digit (int): Preferred digit.
            n_digit (int): Non-preferred digit.
        """
        self.sequences_dir = os.path.join(sequences_dir, condition)
        self.condition = condition
        self.seq_length = seq_length
        self.transform = transform
        self.p_digit = p_digit
        self.n_digit = n_digit
        self.sequence_ids = sorted(list(set([f.split('_')[1] for f in os.listdir(self.sequences_dir) if f.endswith('.png')])))
    
    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        sequence_id = self.sequence_ids[idx]
        sequence = []
        for t in range(self.seq_length):
            img_path = os.path.join(self.sequences_dir, f'seq_{sequence_id}_{self.condition}_frame_{t:02d}.png')
            img = Image.open(img_path).convert('L')  
            if self.transform:
                img = self.transform(img)
            sequence.append(img)
        sequence = torch.stack(sequence)
        return sequence, self.condition

def load_model(model_path, device):
    model = SimpleCNN(input_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def compute_selectivity(model, device, data_loader, p_digit=0, n_digit=1):
    """
    Computes object selectivity (P - N) for swap and nonswap conditions.
    """
    selectivity_swap = []
    selectivity_nonswap = []
    
    for sequences, condition in tqdm(data_loader, desc='Processing Sequences'):
        sequences = sequences.to(device)  
        batch_size, seq_length, C, H, W = sequences.shape
        for i in range(batch_size):
            for t in range(seq_length):
                img = sequences[i, t]
                x_flat, output = model(img.unsqueeze(0))  
                p_act = output[0, p_digit].item()
                n_act = output[0, n_digit].item()
                selectivity = p_act - n_act
                if condition == 'swap':
                    selectivity_swap.append(selectivity)
                else:
                    selectivity_nonswap.append(selectivity)
    
    avg_selectivity_swap = np.mean(selectivity_swap)
    avg_selectivity_nonswap = np.mean(selectivity_nonswap)
    
    return avg_selectivity_swap, avg_selectivity_nonswap

def main():
    parser = argparse.ArgumentParser(description='Run Swap and Nonswap Experiments')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--sequences_dir', type=str, default='data/swap_nonswap_images', help='Directory containing swap and nonswap sequences')
    parser.add_argument('--seq_length', type=int, default=20, help='Number of frames per sequence')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--p_digit', type=int, default=0, help='Preferred digit')
    parser.add_argument('--n_digit', type=int, default=1, help='Non-preferred digit')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    swap_dataset = SwapNonswapSequenceDataset(
        sequences_dir=args.sequences_dir,
        condition='swap',
        seq_length=args.seq_length,
        transform=transform,
        p_digit=args.p_digit,
        n_digit=args.n_digit
    )
    nonswap_dataset = SwapNonswapSequenceDataset(
        sequences_dir=args.sequences_dir,
        condition='nonswap',
        seq_length=args.seq_length,
        transform=transform,
        p_digit=args.p_digit,
        n_digit=args.n_digit
    )

    swap_loader = DataLoader(swap_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    nonswap_loader = DataLoader(nonswap_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    from itertools import chain
    class CombinedLoader:
        def __init__(self, loaders):
            self.loaders = loaders
            self.iterators = [iter(loader) for loader in loaders]

        def __iter__(self):
            return self

        def __next__(self):
            for i in range(len(self.loaders)):
                try:
                    return next(self.iterators[i])
                except StopIteration:
                    self.iterators[i] = iter(self.loaders[i])
                    return next(self.iterators[i])

    combined_loader = CombinedLoader([swap_loader, nonswap_loader])

    avg_selectivity_swap, avg_selectivity_nonswap = compute_selectivity(model, device, combined_loader, 
                                                                         p_digit=args.p_digit, 
                                                                         n_digit=args.n_digit)
    
    print(f"Average Selectivity (Swap): {avg_selectivity_swap:.4f}")
    print(f"Average Selectivity (Nonswap): {avg_selectivity_nonswap:.4f}")

    os.makedirs('visuals/selectivity', exist_ok=True)
    with open('visuals/selectivity/selectivity_results.txt', 'w') as f:
        f.write(f"Average Selectivity (Swap): {avg_selectivity_swap:.4f}\n")
        f.write(f"Average Selectivity (Nonswap): {avg_selectivity_nonswap:.4f}\n")
    
    plt.figure(figsize=(8,6))
    conditions = ['Swap', 'Nonswap']
    selectivities = [avg_selectivity_swap, avg_selectivity_nonswap]
    sns.barplot(x=conditions, y=selectivities, palette="viridis")
    plt.ylabel('Selectivity (P - N)')
    plt.title('Object Selectivity under Swap and Nonswap Exposures')
    plt.savefig('visuals/selectivity/selectivity_comparison.png')
    plt.show()
    print("Selectivity plot saved to visuals/selectivity/selectivity_comparison.png")

if __name__ == "__main__":
    main()
