# datasets/librispeech_speaker.py
import torch
from torch.utils.data import Dataset
import torchaudio
import os
import csv

class LibriSpeechSpeaker(Dataset):
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
        self.speaker_ids = self._get_unique_speaker_ids()
        self.speaker_to_class = {speaker_id: idx for idx, speaker_id in enumerate(sorted(self.speaker_ids))}
        
    def _get_unique_speaker_ids(self):
        speaker_ids = set()
        for i in range(len(self.dataset)):
            _, _, _, speaker_id, _, _ = self.dataset[i]
            speaker_ids.add(speaker_id)
        return speaker_ids

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        label = self.speaker_to_class[speaker_id]
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label
