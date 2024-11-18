# datasets/librispeech_speaker.py
import os
import torch
from torch.utils.data import Dataset
import torchaudio

class LibriSpeechSpeaker(Dataset):
    def __init__(self, root, url, download=False, transform=None):
        """
        Args:
            root (string): Root directory of dataset.
            url (string): Dataset subset to load ('train-clean-100', 'test-clean', etc.).
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory.
            transform (callable, optional): A function/transform that takes in an audio waveform and returns a transformed version.
        """
        super(LibriSpeechSpeaker, self).__init__()
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=download)
        self.transform = transform
        self.speaker_to_class = self._create_speaker_mapping()

    def _create_speaker_mapping(self):
        speaker_set = set()
        for _, _, _, speaker_id, _, _ in self.dataset:
            speaker_set.add(speaker_id)
        speaker_list = sorted(list(speaker_set))
        speaker_to_class = {speaker: idx for idx, speaker in enumerate(speaker_list)}
        return speaker_to_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        label = self.speaker_to_class[speaker_id]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label
