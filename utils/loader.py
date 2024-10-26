import torch
import torchvision
import pandas as pd
import os
import random
from torch.utils.data import DataLoader, Dataset
from config.config import config
from utils.common import load_image

root = config['root']
image_dir = config['image_dir']
label_key = config['label_key']
train_batch_size = config['train_batch_size']
val_batch_size = config['val_batch_size']
data_types = config['data_types']

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224), antialias=True),
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])
transform_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224), antialias=True),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 归一化
])


class OCTADataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.transform = transform
        self.data = data
        if config['shuffle']:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        images, label = self.data[item]
        images = [self.transform(i) for i in images]
        image = torch.cat(images, dim=0)
        return image, label


def load_data(label_file):
    labels = pd.read_csv(os.path.join(root, label_file), index_col=0)
    data = []
    for sample in labels.index:
        dir_ = os.path.join(root, image_dir, sample)
        images = [load_image(os.path.join(dir_, file)) for file in os.listdir(dir_) if file in data_types]
        label = torch.tensor(labels[label_key][sample]).long()
        data.append((images, label))
    return data


def get_loader(label_file, train=True):
    data = load_data(label_file)
    dataset = OCTADataset(data, transform_train if train else transform_val)
    loader = DataLoader(
        dataset,
        batch_size=train_batch_size if train else val_batch_size,
        shuffle=True if train else val_batch_size
    )
    return (loader, get_class_weights(label_file)) if train else loader


def get_class_weights(label_file):
    labels = pd.read_csv(os.path.join(root, label_file), index_col=0)
    class_counts = labels[label_key].value_counts()

    total_samples = sum(class_counts)
    weights = [total_samples / count for count in class_counts]
    weights = [w / sum(weights) for w in weights]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights
