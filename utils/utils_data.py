import os
import cv2
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from .Dataset import DatasetClass
from .utils_algo import *


def load_dataset(args):
    batch_size = args.batch_size

    if args.dataset == "cifar10":
        mean, std, size = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261), 32
        test_temp_dataset = dsets.CIFAR10(
            root="./data", train=False, transform=None, download=True
        )
    elif args.dataset == "cifar100":
        mean, std, size = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 32
        test_temp_dataset = dsets.CIFAR100(
            root="./data", train=False, transform=None, download=True
        )
        test_temp_dataset.targets = sparse2coarse(test_temp_dataset.targets)
    elif args.dataset == "stl10":
        mean, std, size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 96
        test_temp_dataset = dsets.STL10(
            root="./data", split="test", transform=None, download=True
        )
        test_temp_dataset.targets = test_temp_dataset.labels
        test_temp_dataset.data = transpose(
            np.array(test_temp_dataset.data), source="NCHW", target="NHWC"
        )
    elif args.dataset == "alz":
        mean, std, size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 224
        test_temp_dataset = dsets.CIFAR10(
            root="./data", train=False, transform=None, download=True
        )
        test_temp_dataset.data, test_temp_dataset.targets = load_alz(train=False)
    else:
        raise NotImplementedError("Wrong dataset arguments.")

    data_stats = {"mean": mean, "std": std, "size": size}

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    data_test, test_labels_temp = np.array(test_temp_dataset.data), np.array(
        test_temp_dataset.targets
    )
    test_labels = binarize_labels(args, test_labels_temp)
    test_dataset = DatasetClass(
        data_test, None, test_labels, data_stats, transform=test_transform, train=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=4
    )

    all_data, all_labels = get_train_data(args.dataset)
    all_labels_pu = binarize_labels(args, all_labels)
    train_labeled_idxs, train_unlabeled_idxs, valid_idxs, prior = train_val_split(
        all_labels, args.n_positive, args.positive_list, args.n_valid
    )

    train_idxs = []
    for t in train_labeled_idxs:
        train_idxs.append(t)
    for t in train_unlabeled_idxs:
        train_idxs.append(t)
    train_data = all_data[train_idxs]
    train_pu_labels = np.zeros(len(train_idxs))
    train_pu_labels[: len(train_labeled_idxs)] = 1
    train_labels = all_labels_pu[train_idxs]

    print("Training data num: ", len(train_pu_labels))
    print("Positive label num: ", train_pu_labels.sum())

    if args.using_cont:
        train_transform = None
    else:
        train_transform = test_transform

    train_dataset = DatasetClass(
        train_data, train_pu_labels, train_labels, data_stats, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_data = all_data[valid_idxs]
    valid_labels = all_labels_pu[valid_idxs]
    valid_dataset = DatasetClass(
        valid_data,
        None,
        valid_labels,
        data_stats,
        transform=test_transform,
        train=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    eval_dataset = DatasetClass(
        train_data, train_pu_labels, train_labels, data_stats, transform=test_transform
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    dim = train_dataset.images.size / len(train_dataset.images)

    return train_loader, test_loader, valid_loader, eval_loader, dim


def get_train_data(dataset):
    if dataset == "cifar10":
        temp_train_dataset = dsets.CIFAR10(
            root="./data", train=True, download=True, transform=None
        )
    elif dataset == "cifar100":
        temp_train_dataset = dsets.CIFAR100(
            root="./data", train=True, download=True, transform=None
        )
        temp_train_dataset.targets = sparse2coarse(temp_train_dataset.targets)
    elif dataset == "stl10":
        temp_train_dataset = dsets.STL10(
            root="./data", split="train+unlabeled", download=True, transform=None
        )
        temp_train_dataset.targets = temp_train_dataset.labels
        temp_train_dataset.data = transpose(
            np.array(temp_train_dataset.data), source="NCHW", target="NHWC"
        )
    elif dataset == "alz":
        return load_alz()
    data, labels = np.array(temp_train_dataset.data), np.array(
        temp_train_dataset.targets
    )
    return data, labels


def train_val_split(labels, n_labeled, positive_label_list, val_num):
    labels = np.array(labels)
    label_types = np.unique(labels)

    train_labeled_idxs = []
    train_unlabeled_idxs = []
    valid_idxs = []
    n_labeled_per_class = n_labeled // len(positive_label_list)

    num_positive = 0
    label_num = len(label_types)
    if -1 in label_types:
        label_num -= 1
    val_num_per_clss = val_num // label_num

    for i in label_types:
        idxs = np.where(labels == i)[0]

        if i == -1:
            print(
                "[Warning] Label {} detected, collected to unlabeled data. ".format(i)
            )
        elif i != -1:
            np.random.shuffle(idxs)

            valid_piece = idxs[:val_num_per_clss]
            valid_idxs.extend(valid_piece)

            idxs = np.array(list(set(idxs.tolist()) - set(valid_piece.tolist())))

            if i in positive_label_list:
                pos_piece = idxs[:n_labeled_per_class]
                train_labeled_idxs.extend(pos_piece)
                num_positive += len(idxs)
                idxs = list(set(idxs.tolist()) - set(pos_piece.tolist()))

        train_unlabeled_idxs.extend(idxs)

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    prior = num_positive / (len(train_unlabeled_idxs) + len(train_labeled_idxs))

    return train_labeled_idxs, train_unlabeled_idxs, valid_idxs, prior


def binarize_labels(args, labels):
    if not args.reverse:
        return np.array([1 if l in args.positive_list else 0 for l in labels])
    else:
        return np.array([1 if l not in args.positive_list else 0 for l in labels])


def transpose(x, source="NCHW", target="NHWC"):
    """
    N: batch size
    H: height
    W: weight
    C: channel
    """
    return x.transpose([source.index(d) for d in target])


def normalise(x, mean, std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array(
        [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
    )
    return coarse_labels[targets]


def load_alz(train=True):
    class_list = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    data = []
    targets = []
    for class_name in class_list:
        if train:
            baseDir = "./data/Alzheimer/train/" + class_name
        else:
            baseDir = "./data/Alzheimer/test/" + class_name
        for file in os.listdir(baseDir):
            dir = baseDir + "/" + file
            img = cv2.imread(dir)
            img = cv2.resize(img, (224, 224))
            data.append(img)
            targets.append(class_list.index(class_name))

    return np.array(data), np.array(targets)
