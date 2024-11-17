import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

def create_moving_mnist(num_sequences=1000, seq_length=20, image_size=64, digit_size=28, save_dir='data/images'):
    os.makedirs(save_dir, exist_ok=True)
    mnist = MNIST(root='data/mnist', download=True, transform=transforms.ToTensor())
    digits = mnist.data.numpy()
    labels = mnist.targets.numpy()

    for i in tqdm(range(num_sequences), desc="Generating Sequences"):
        sequence = np.zeros((seq_length, image_size, image_size), dtype=np.uint8)
        digit1 = digits[np.random.randint(0, len(digits))]
        digit2 = digits[np.random.randint(0, len(digits))]
        x1, y1 = np.random.randint(0, image_size - digit_size, size=2)
        x2, y2 = np.random.randint(0, image_size - digit_size, size=2)
        vx1, vy1 = np.random.choice([-1, 1], size=2)
        vx2, vy2 = np.random.choice([-1, 1], size=2)

        for t in range(seq_length):
            frame = np.zeros((image_size, image_size), dtype=np.uint8)
            frame[y1:y1+digit_size, x1:x1+digit_size] = np.maximum(
                frame[y1:y1+digit_size, x1:x1+digit_size], digit1)
            frame[y2:y2+digit_size, x2:x2+digit_size] = np.maximum(
                frame[y2:y2+digit_size, x2:x2+digit_size], digit2)
            sequence[t] = frame
            
            x1 += vx1
            y1 += vy1
            x2 += vx2
            y2 += vy2
            
            if x1 < 0:
                x1 = 0
                vx1 *= -1
            elif x1 > image_size - digit_size:
                x1 = image_size - digit_size
                vx1 *= -1

            if y1 < 0:
                y1 = 0
                vy1 *= -1
            elif y1 > image_size - digit_size:
                y1 = image_size - digit_size
                vy1 *= -1

            if x2 < 0:
                x2 = 0
                vx2 *= -1
            elif x2 > image_size - digit_size:
                x2 = image_size - digit_size
                vx2 *= -1

            if y2 < 0:
                y2 = 0
                vy2 *= -1
            elif y2 > image_size - digit_size:
                y2 = image_size - digit_size
                vy2 *= -1

        for t in range(seq_length):
            img = Image.fromarray(sequence[t], mode='L')
            img.save(os.path.join(save_dir, f'seq_{i:04d}_frame_{t:02d}.png'))

if __name__ == "__main__":
    create_moving_mnist()
