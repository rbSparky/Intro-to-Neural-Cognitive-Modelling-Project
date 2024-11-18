import torch
from datasets.librispeech_speaker import LibriSpeechSpeaker
import torchaudio.transforms as T
import torchvision.transforms as TT

def main():
    composed_transform = TT.Compose([
        T.MelSpectrogram(sample_rate=16000, n_mels=64),
        T.AmplitudeToDB(),
        TT.Resize((224, 224)),
        TT.Normalize(mean=[0.485], std=[0.229])
    ])

    train_dataset = LibriSpeechSpeaker(
        root="/content/librispeech-clean/",
        url="train-clean-100",
        download=False,
        transform=composed_transform,
        max_speakers=41
    )

    print(f"Total samples: {len(train_dataset)}")
    print(f"Total speakers: {len(train_dataset.speaker_to_class)}")

    for i in range(5):
        waveform, label = train_dataset[i]
        print(f"Sample {i}: Waveform shape: {waveform.shape}, Label: {label}")

if __name__ == "__main__":
    main()
