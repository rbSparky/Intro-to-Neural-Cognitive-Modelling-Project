# datasets/librispeech_phoneme.py
import torch
from torch.utils.data import Dataset
import torchaudio
import os
import csv

class LibriSpeechPhoneme(Dataset):
    def __init__(self, root, url, transform=None, download=False):
        """
        Args:
            root (str): Root directory of dataset where LibriSpeech is stored.
            url (str): Dataset subset to use (e.g., 'train-clean-100', 'test-clean').
            transform (callable, optional): Optional transform to be applied on a sample.
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory.
        """
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=download)
        self.transform = transform
        self.labels = self.load_labels(os.path.join(root, 'phoneme_labels', f'{url}_labels.csv'))
    
    def load_labels(self, csv_path):
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row['file_id']
                phoneme_label = int(row['phoneme_label'])  # Ensure labels are integers 0-40
                labels[file_id] = phoneme_label
        return labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        # Construct file_id based on dataset indexing
        file_path = self.dataset._walker[idx]  # e.g., 'train-clean-100/1089/128104/1089-128104-0000.flac'
        file_id = os.path.splitext(os.path.basename(file_path))[0]  # e.g., '1089-128104-0000'
        label = self.labels.get(file_id, -1)  # Assign -1 if label not found
        
        if label == -1:
            raise ValueError(f"Label for file_id {file_id} not found.")
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label
