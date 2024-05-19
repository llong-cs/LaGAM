import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
import copy


class DatasetClass(Dataset):
    def __init__(
        self, images, pu_labels, true_labels, data_stats, transform=None, train=True
    ):
        self.images = images
        self.pu_labels = pu_labels
        self.true_labels = true_labels
        self.transform = transform
        self.train = train

        if self.transform is None:
            self.weak_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(data_stats["size"], padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(data_stats["mean"], data_stats["std"]),
                ]
            )
            self.strong_transform = copy.deepcopy(self.weak_transform)
            self.strong_transform.transforms.insert(1, RandomAugment(3, 5))

    def update_targets(self, new_labels, idxes):
        self.pu_labels[idxes] = new_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        if not self.train:
            image = self.transform(self.images[index])
            true_label = self.true_labels[index]
            return image, true_label
        else:
            if self.transform is None:
                image_w = self.weak_transform(self.images[index])
                image_s = self.strong_transform(self.images[index])
                label = self.pu_labels[index]
                true_label = self.true_labels[index]

                return image_w, image_s, label, true_label, index
            else:
                label = self.pu_labels[index]
                image = self.transform(self.images[index])
                true_label = self.true_labels[index]
                return image, image, label, true_label, index
