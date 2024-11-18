import os
import torch
from torch.utils.data import Dataset
import torchaudio

class LibriSpeechSpeaker(Dataset):
    def __init__(self, root, url, download=False, transform=None, max_speakers=5):
        """
        Args:
            root (string): Root directory of dataset.
            url (string): Dataset subset to load ('train-clean-100', 'test-clean', etc.).
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory.
            transform (callable, optional): A function/transform that takes in an audio waveform and returns a transformed version.
            max_speakers (int, optional): Maximum number of speakers to include.
        """
        super(LibriSpeechSpeaker, self).__init__()
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=download)
        self.transform = transform
        self.max_speakers = max_speakers
        self.speaker_to_class, self.selected_speakers = self._create_speaker_mapping()
        self.filtered_indices = self._filter_dataset()

    def _create_speaker_mapping(self):
        speaker_set = set()
        for _, _, _, speaker_id, _, _ in self.dataset:
            speaker_set.add(speaker_id)
            if len(speaker_set) >= self.max_speakers:
                break
        selected_speakers = sorted(list(speaker_set))
        speaker_to_class = {speaker: idx for idx, speaker in enumerate(selected_speakers)}
        return speaker_to_class, selected_speakers

    def _filter_dataset(self):
        filtered = []
        for idx, (_, _, _, speaker_id, _, _) in enumerate(self.dataset):
            if speaker_id in self.speaker_to_class:
                filtered.append(idx)
        return filtered

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[actual_idx]
        label = self.speaker_to_class[speaker_id]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label
