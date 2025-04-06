import numpy as np
from PIL import Image
import os
import argparse


def save_random_images(num_images, folder, num_classes):
    images_per_class = num_images // num_classes
    for class_label in range(num_classes):
        os.makedirs(os.path.join(folder, str(class_label)), exist_ok=True)
        for i in range(images_per_class):
            noise_image = (np.random.rand(*img_size) * 255).astype(np.uint8)
            img = Image.fromarray(noise_image)
            img.save(os.path.join(folder, str(class_label), f'image_{class_label}_{i}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random noise images.')
    parser.add_argument('--num_train', type=int, default=60000, help='Number of training images')
    parser.add_argument('--num_test', type=int, default=10000, help='Number of test images')
    args = parser.parse_args()

    # Directory setup
    train_path = f"random_mnist_{args.num_train}/train"
    test_path = f"random_mnist_{args.num_train}/test"

    # Configuration
    img_size = (28, 28)

    # Generate and save
    save_random_images(args.num_train, train_path, 10)
    save_random_images(args.num_test, test_path, 10)