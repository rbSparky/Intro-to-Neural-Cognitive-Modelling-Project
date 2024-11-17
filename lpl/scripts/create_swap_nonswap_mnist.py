import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

def create_swap_nonswap_mnist(num_sequences=500, seq_length=20, image_size=64, digit_size=28, 
                              save_dir='data/swap_nonswap_images', swap=True, swap_frame=10, 
                              p_digit=0, n_digit=1):
    """
    Generates swap and nonswap sequences for the experiment.

    Args:
        num_sequences (int): Number of sequences per condition.
        seq_length (int): Number of frames per sequence.
        image_size (int): Size of the canvas.
        digit_size (int): Size of each digit.
        save_dir (str): Directory to save generated sequences.
        swap (bool): Whether to create swap or nonswap sequences.
        swap_frame (int): Frame at which to perform the swap.
        p_digit (int): Preferred digit.
        n_digit (int): Non-preferred digit.
    """
    os.makedirs(save_dir, exist_ok=True)
    mnist = MNIST(root='data/mnist', download=True, transform=transforms.ToTensor())
    digits = mnist.data.numpy()
    labels = mnist.targets.numpy()

    p_indices = np.where(labels == p_digit)[0]
    n_indices = np.where(labels == n_digit)[0]

    for i in tqdm(range(num_sequences), desc=f"Generating {'Swap' if swap else 'Nonswap'} Sequences"):
        sequence = np.zeros((seq_length, image_size, image_size), dtype=np.uint8)
        
        p_digit_image = digits[np.random.choice(p_indices)]
        n_digit_image = digits[np.random.choice(n_indices)]
        
        x_p, y_p = np.random.randint(0, image_size - digit_size, size=2)
        x_n, y_n = np.random.randint(0, image_size - digit_size, size=2)
        
        vx_p, vy_p = np.random.choice([-1, 1], size=2)
        vx_n, vy_n = np.random.choice([-1, 1], size=2)
        
        for t in range(seq_length):
            frame = np.zeros((image_size, image_size), dtype=np.uint8)
            
            frame[y_p:y_p+digit_size, x_p:x_p+digit_size] = np.maximum(
                frame[y_p:y_p+digit_size, x_p:x_p+digit_size], p_digit_image)
            
            frame[y_n:y_n+digit_size, x_n:x_n+digit_size] = np.maximum(
                frame[y_n:y_n+digit_size, x_n:x_n+digit_size], n_digit_image)
            
            sequence[t] = frame.copy()
            
            x_p += vx_p
            y_p += vy_p
            x_n += vx_n
            y_n += vy_n
            
            if x_p < 0:
                x_p = 0
                vx_p *= -1
            elif x_p > image_size - digit_size:
                x_p = image_size - digit_size
                vx_p *= -1

            if y_p < 0:
                y_p = 0
                vy_p *= -1
            elif y_p > image_size - digit_size:
                y_p = image_size - digit_size
                vy_p *= -1

            if x_n < 0:
                x_n = 0
                vx_n *= -1
            elif x_n > image_size - digit_size:
                x_n = image_size - digit_size
                vx_n *= -1

            if y_n < 0:
                y_n = 0
                vy_n *= -1
            elif y_n > image_size - digit_size:
                y_n = image_size - digit_size
                vy_n *= -1

            if swap and t == swap_frame:
                p_digit_image, n_digit_image = n_digit_image, p_digit_image
                x_p, x_n = x_n, x_p
                y_p, y_n = y_n, y_p

        for t in range(seq_length):
            img = Image.fromarray(sequence[t], mode='L')
            exposure = 'swap' if swap else 'nonswap'
            img.save(os.path.join(save_dir, f'seq_{i:04d}_{exposure}_frame_{t:02d}.png'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create Swap and Nonswap Moving MNIST Sequences')
    parser.add_argument('--num_sequences', type=int, default=500, help='Number of sequences per condition')
    parser.add_argument('--seq_length', type=int, default=20, help='Number of frames per sequence')
    parser.add_argument('--image_size', type=int, default=64, help='Canvas size')
    parser.add_argument('--digit_size', type=int, default=28, help='Digit size')
    parser.add_argument('--save_dir', type=str, default='data/swap_nonswap_images', help='Directory to save sequences')
    parser.add_argument('--swap_frame', type=int, default=10, help='Frame at which to perform the swap')
    parser.add_argument('--p_digit', type=int, default=0, help='Preferred digit')
    parser.add_argument('--n_digit', type=int, default=1, help='Non-preferred digit')

    args = parser.parse_args()

    create_swap_nonswap_mnist(num_sequences=args.num_sequences, seq_length=args.seq_length, 
                              image_size=args.image_size, digit_size=args.digit_size, 
                              save_dir=os.path.join(args.save_dir, 'swap'), swap=True, 
                              swap_frame=args.swap_frame, p_digit=args.p_digit, n_digit=args.n_digit)

    create_swap_nonswap_mnist(num_sequences=args.num_sequences, seq_length=args.seq_length, 
                              image_size=args.image_size, digit_size=args.digit_size, 
                              save_dir=os.path.join(args.save_dir, 'nonswap'), swap=False, 
                              swap_frame=args.swap_frame, p_digit=args.p_digit, n_digit=args.n_digit)
