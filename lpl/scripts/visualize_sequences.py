import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize_sequence(sequence_id=0, save_dir='data/images', num_frames=20):
    frames = []
    for t in range(num_frames):
        img_path = os.path.join(save_dir, f'seq_{sequence_id:04d}_frame_{t:02d}.png')
        img = Image.open(img_path)
        frames.append(img)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 1.5, 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_sequence(sequence_id=0)
