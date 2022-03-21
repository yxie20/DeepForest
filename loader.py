import torch
from torch.utils.data import Dataset
import os
import PIL
from PIL import Image
import numpy as np

class TreeDataset(Dataset):
    def __init__(self, data_dir, resolution=256):
        self.resolution = resolution
        self.data_dir = data_dir
        self.data = []
        self.load_data()


    def load_data(self):
        for file in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, file)
            img = Image.open(filepath)
            img = self.pad_image(img)
            img = img.resize((self.resolution, self.resolution), resample=PIL.Image.BICUBIC)
            self.data.append(img)

    

    @staticmethod
    def pad_image(image):
        np_img = np.asarray(image)
        #image is taller than it is wide
        if np_img.shape[0] > np_img.shape[1]:
            padding = np_img.shape[0] - np_img.shape[1]
            left_padding_width = padding // 2
            right_padding_width = padding - left_padding_width
            if left_padding_width > 0:
                left_padding = np.zeros((np_img.shape[0], left_padding_width, 3))
                right_padding = np.zeros((np_img.shape[0], right_padding_width, 3))
                padded_image = np.hstack([left_padding, np_img, right_padding])
            else:
                right_padding = np.zeros((np_img.shape[0], right_padding_width, 3))
                padded_image = np.hstack([np_img, right_padding])
        # image is wider than it is tall
        elif np_img.shape[1] > np_img.shape[0]:
            padding = np_img.shape[1] - np_img.shape[0]
            top_padding_width = padding // 2
            bottom_padding_width = padding - top_padding_width
            if top_padding_width > 0:
                top_padding = np.zeros((top_padding_width, np_img.shape[1], 3))
                bottom_padding = np.zeros((bottom_padding_width, np_img.shape[1], 3))
                padded_image = np.vstack([top_padding, np_img, bottom_padding])
            else:
                bottom_padding = np.zeros((bottom_padding_width, np_img.shape[1], 3))
                padded_image = np.vstack([np_img, bottom_padding])
        # image is already square
        else:
            padded_image = np_img
        return Image.fromarray(padded_image.astype(np.uint8))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Show image resizing and crops")
    parser.add_argument("--data-dir", type=str, help="The directory with sample images")
    args = parser.parse_args()

    dataset = TreeDataset(args.data_dir)

    for img in dataset:
        f, ax = plt.subplots()
        ax.imshow(img)
        plt.show()