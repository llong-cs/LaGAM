# LaGAM: Positive-Unlabeled Learning by Latent Group-Aware Meta Disambiguation

This is the implementation of our CVPR 2024 paper: Positive-Unlabeled Learning by Latent Group-Aware Meta Disambiguation.

## Training
### CIFAR-10-1

```python
python train.py --exp-dir experiment --arch resnet18 --dataset cifar10 --positive_list 0,1,8,9 --warmup_epoch 20 --n_positive 1000 --n_valid 500 --num_cluster 5 --cont_cutoff --identifier classifier --knn_aug --num_neighbors 10 --epochs 400
```

### CIFAR-10-2

```python
python train.py --exp-dir experiment --arch resnet18 --dataset cifar10 --positive_list 0,1,8,9 --warmup_epoch 20 --n_positive 1000 --n_valid 500 --num_cluster 5 --cont_cutoff --identifier classifier --knn_aug --num_neighbors 10 --epochs 400 --reverse 1
```

### CIFAR-100-1

```python
python train.py --exp-dir experiment --arch resnet18 --dataset cifar100 --positive_list 18,19 --warmup_epoch 20 --n_positive 1000 --n_valid 500 --num_cluster 100 --cont_cutoff --identifier classifier --knn_aug --num_neighbors 10 --epochs 400
```

### CIFAR-100-2

```python
python train.py --exp-dir experiment --arch resnet18 --dataset cifar100 --positive_list 0,1,7,8,11,12,13,14,15,16 --warmup_epoch 20 --n_positive 1000 --n_valid 500 --num_cluster 100 --cont_cutoff --identifier classifier --knn_aug --num_neighbors 10 --epochs 400
```

### STL-10-1

```python
python train.py --exp-dir experiment --arch resnet18 --dataset stl10 --positive_list 0,2,3,8,9 --warmup_epoch 20 --n_positive 1000 --n_valid 500 --num_cluster 100 --cont_cutoff --identifier classifier --knn_aug --num_neighbors 10 --epochs 400
```

### STL-10-2

```python
python train.py --exp-dir experiment --arch resnet18 --dataset stl10 --positive_list 0,2,3,8,9 --warmup_epoch 20 --n_positive 1000 --n_valid 500 --num_cluster 100 --cont_cutoff --identifier classifier --knn_aug --num_neighbors 10 --epochs 400 --reverse 1
```

