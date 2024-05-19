import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import numpy as np
import dill
from tqdm import tqdm
from models.models import *
from models.resnet import *
from models.wideresnet import WideResNet
from utils.utils_algo import *
from utils.utils_data import *
from utils.utils_loss import *
import warnings

warnings.filterwarnings("ignore")

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(
    description="PyTorch implementation of LaGAM"
)
parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    choices=["cifar10", "cifar100", "stl10", "alz"],
    help="dataset name (cifar10)",
)
parser.add_argument(
    "--exp-dir",
    default="experiment",
    type=str,
    help="experiment directory for saving checkpoints and logs",
)
parser.add_argument(
    "--no_verbose", action="store_true", help="disable showing running statics"
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=["resnet50", "resnet18", "CNN", "WRN", "CNN13"],
    help="network architecture",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    help="number of data loading workers",
)
parser.add_argument(
    "--epochs", default=400, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    help="mini-batch size (default: 64), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--lr_decay_epochs",
    type=str,
    default="250,300,350",
    help="where to decay lr, can be a list",
)
parser.add_argument(
    "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
)
parser.add_argument(
    "--cosine", action="store_true", default=False, help="use cosine lr schedule"
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p", "--print-freq", default=100, type=int, help="print frequency (default: 100)"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument("--num-class", default=10, type=int, help="number of class")

parser.add_argument("--n_positive", default=1000, type=int, help="num_labeled data")
parser.add_argument("--n_valid", default=500, type=int, help="number of valid examples")
parser.add_argument(
    "--positive_list", default="0,1,8,9", type=str, help="list of positive labels"
)
# Standatd setups:
#   CIFAR-10-1:   0,1,8,9
#   CIFAR-10-2:   2,3,4,5,6,7
#   STL-10-1:     0,2,3,8,9
#   STL-10-2:     1,4,5,6,7
#   CIFAR-100-1:  18,19
#   CIFAR-100-2:  0,1,7,8,11,12,13,14,15,16
parser.add_argument(
    "--ent_loss", action="store_true", help="whether enable entropy loss"
)
parser.add_argument("--mix_weight", default=1.0, type=float, help="mixup loss weight")
parser.add_argument(
    "--rho_range", default="0.95,0.8", type=str, help="momentum updating parameter"
)
parser.add_argument(
    "--warmup_epoch", default=20, type=int, help="epoch number of warm up"
)
parser.add_argument(
    "--using_cont", type=int, default=1, help="whether using contrastive loss"
)

parser.add_argument("--num_cluster", default=100, type=int, help="number of clusters")
parser.add_argument("--temperature", default=0.07, type=float, help="mixup loss weight")
parser.add_argument(
    "--cont_cutoff", action="store_true", help="whether cut off by classifier"
)
parser.add_argument("--knn_aug", action="store_true", help="whether using kNN for CL")
parser.add_argument("--num_neighbors", default=10, type=int, help="number of neighbors")
parser.add_argument(
    "--identifier",
    default=None,
    type=str,
    help="identifier for meta layers, e.g. classifier",
)
parser.add_argument(
    "--contrastive_clustering",
    default=1,
    type=int,
    help="whether using contrastive clustering",
)

parser.add_argument("--reverse", default=0, type=int, help="whether inverse label")
parser.add_argument("--tag", default="", type=str, help="special identifier")
parser.add_argument("--save", default=0, type=int, help="whether save model")


class Trainer:
    def __init__(self, args, model_func=None):
        self.args = args
        model_path = "{ds}_ep{ep}_we{we}_pos{pl}_nl{nl}_rho{rs}~{re}_co{co}_knn{knn}{k}_sd_{seed}".format(
            ds=args.dataset,
            ep=args.epochs,
            pl=str(args.positive_list),
            nl=args.n_positive,
            rs=args.rho_start,
            re=args.rho_end,
            knn=args.knn_aug,
            co=args.cont_cutoff,
            k=args.num_neighbors,
            we=args.warmup_epoch,
            seed=args.seed,
        )
        args.exp_dir = os.path.join(args.exp_dir, model_path)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            cudnn.deterministic = True

        train_loader, test_loader, valid_loader, eval_loader, dim = load_dataset(
            args=args
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader
        self.dim = dim
        print("=> creating model '{}'".format(args.arch))
        model = create_model(args, self.dim)

        optimizer = torch.optim.SGD(
            model.params(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        self.model = model
        self.optimizer = optimizer
        self.bce_loss = BCELoss(args.ent_loss)
        self.contrastive_loss = ContLoss(
            temperature=args.temperature,
            cont_cutoff=args.cont_cutoff,
            knn_aug=args.knn_aug,
            num_neighbors=args.num_neighbors,
            contrastive_clustering=args.contrastive_clustering,
        )

    def train(self):
        args = self.args
        optimizer = self.optimizer

        best_acc = 0

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(args, optimizer, epoch)

            if epoch < args.warmup_epoch or args.using_cont == 0:
                self.train_loop(epoch)
            else:
                features = self.compute_features()
                cluster_result = run_kmeans(features, args)
                self.train_loop(epoch, cluster_result)

            acc_test = self.test()

            with open(os.path.join(args.exp_dir, "result.log"), "a+") as f:
                f.write(
                    "Epoch {}: Acc {:.2f}, Best Acc {:.2f}. (lr {:.5f})\n".format(
                        epoch, acc_test, best_acc, optimizer.param_groups[0]["lr"]
                    )
                )

            if acc_test > best_acc:
                best_acc = acc_test

        if args.save:
            file_name = "{}_{}_{}_{}_{}".format(
                args.dataset, args.arch, args.n_valid, args.num_cluster, args.tag
            )

            torch.save(self.model, file_name + ".pth")

            with open(file_name + ".pkl", "wb") as f:
                dill.dump(self.train_loader.dataset, f)

    def train_loop(self, epoch, cluster_result=None):
        args = self.args
        train_loader = self.train_loader
        model = self.model
        optimizer = self.optimizer
        bce_loss = self.bce_loss
        contrastive_loss = self.contrastive_loss

        batch_time = AverageMeter("Time", ":1.2f")
        data_time = AverageMeter("Data", ":1.2f")
        acc_cls = AverageMeter("Acc@Cls", ":2.2f")
        loss_cls_log = AverageMeter("Loss@Cls", ":2.2f")
        loss_cont_log = AverageMeter("Loss@Cont", ":2.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, acc_cls, loss_cls_log, loss_cont_log],
            prefix="Epoch: [{}]".format(epoch),
        )

        model.train()

        updated_label_list = []
        true_label_list = []
        index_list = []
        ema_param = (
            1.0 * epoch / args.epochs * (args.rho_end - args.rho_start) + args.rho_start
        )

        end = time.time()

        for i, (images, images_s, labels_, true_labels, index) in enumerate(
            train_loader
        ):
            data_time.update(time.time() - end)

            if labels_.sum() == 0:
                continue
            true_label_list.append(true_labels)
            index_list.append(index)

            images, images_s, labels_, index = (
                images.cuda(),
                images_s.cuda(),
                labels_.cuda(),
                index.cuda(),
            )
            labels_ = labels_.unsqueeze(1)
            labels = torch.cat([1 - labels_, labels_], dim=1).detach()
            Y_true = true_labels.long().detach().cuda()
            bs = len(labels)
            cluster_idxes = (
                None if cluster_result is None else cluster_result["im2cluster"][index]
            )

            if epoch < args.warmup_epoch:
                labels_final = labels
            else:
                meta_model = create_model(args, self.dim)
                meta_model.load_state_dict(model.state_dict())

                preds_meta = meta_model(images)

                eps = to_var(torch.zeros(bs, 2).cuda())
                labels_meta = labels + eps
                loss = bce_loss(preds_meta, labels_meta)

                meta_model.zero_grad()

                params = []
                for name, p in meta_model.named_params(meta_model):
                    if args.identifier in name and len(p.shape) > 1:
                        params.append(p)
                grads = torch.autograd.grad(
                    loss, params, create_graph=True, allow_unused=True
                )
                meta_lr = 0.001
                meta_model.update_params(
                    meta_lr, source_params=grads, identifier=args.identifier
                )

                try:
                    images_v, labels_v = next(valid_loder_iter)
                except:
                    valid_loder_iter = iter(self.valid_loader)
                    images_v, labels_v = next(valid_loder_iter)

                images_v = images_v.cuda()
                labels_v = F.one_hot(labels_v.cuda(), 2).float()

                preds_v = meta_model(images_v)

                loss_meta_v = bce_loss(preds_v, labels_v)
                grad_eps = torch.autograd.grad(
                    loss_meta_v, eps, only_inputs=True, allow_unused=True
                )[0]

                eps = eps - grad_eps
                meta_detected_labels = eps.argmax(dim=1)
                meta_detected_labels[labels_.squeeze() == 1] = 1
                meta_detected_labels = F.one_hot(meta_detected_labels, 2)
                meta_detected_labels = meta_detected_labels.detach()

                updated_labels = labels
                updated_labels = updated_labels * ema_param + meta_detected_labels * (
                    1 - ema_param
                )
                labels_final = updated_labels.detach()

                updated_label_list.append(updated_labels[:, 1].cpu())

                del grad_eps, grads, params

            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            X_w_c = images
            pseudo_label_c = labels_final
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            pseudo_label_c_rand = pseudo_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
            pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
            logits_mix = model(X_w_c_mix)
            loss_mix = bce_loss(logits_mix, pseudo_label_c_mix)

            preds_final, feat_cont = model(images, flag_feature=True)
            loss_cls = bce_loss(preds_final, labels_final)

            loss_final = loss_cls + args.mix_weight * loss_mix

            if args.using_cont:
                _, feat_cont_s = model(images_s, flag_feature=True)
                loss_cont = contrastive_loss(
                    feat_cont,
                    feat_cont_s,
                    cluster_idxes,
                    preds_final,
                    start_knn_aug=epoch > 50,
                )
                loss_final = loss_final + loss_cont

                loss_cont_log.update(loss_cont.item())
            loss_cls_log.update(loss_final.item())

            acc = accuracy(
                torch.cat([1 - preds_final, preds_final], dim=1), Y_true
            )
            acc = acc[0]
            acc_cls.update(acc[0])

            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        if epoch >= args.warmup_epoch and not args.no_verbose:
            true_label_list = torch.cat(true_label_list, dim=0)
            updated_label_list = torch.cat(updated_label_list, dim=0)
            index_list = torch.cat(index_list, dim=0)

            print(updated_label_list[:10])
            print(true_label_list[:10])

            update_label_cate = (updated_label_list > 0.5) * 1
            compare = update_label_cate == true_label_list
            print(
                "New target accuracy: ",
                compare.sum() / len(compare),
                "; ema param: ",
                ema_param,
            )

            self.train_loader.dataset.update_targets(
                updated_label_list.numpy(), index_list
            )

    def test(self):
        model = self.model
        test_loader = self.test_loader

        with torch.no_grad():
            print("==> Evaluation...")
            model.eval()
            pred_list = []
            true_list = []
            for _, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                outputs = model(images)
                pred = torch.sigmoid(outputs)
                pred = torch.cat([1 - pred, pred], dim=1)
                pred_list.append(pred.cpu())
                true_list.append(labels)

            pred_list = torch.cat(pred_list, dim=0)
            true_list = torch.cat(true_list, dim=0)

            acc1 = accuracy(pred_list, true_list, topk=(1,))
            acc1 = acc1[0]
            print("==> Test Accuracy is %.2f%%" % (acc1))
            # print("==> AUC, F1, Recall, Precision are: ")
            # print(auc, f1, recall, precision)
        return float(acc1)

    def compute_features(self):
        model = self.model
        model.eval()
        feat_list = torch.zeros(len(self.eval_loader.dataset), 128)
        with torch.no_grad():
            for i, (images, _, _, _, index) in enumerate(self.eval_loader):
                images = images.cuda(non_blocking=True)
                _, feat = model(images, flag_feature=True)
                feat_list[index] = feat.cpu()
        return feat_list.numpy()

    def save_checkpoint(
        self,
        state,
        is_best,
        filename="checkpoint.pth.tar",
        best_file_name="model_best.pth.tar",
    ):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_file_name)


def create_model(args, dim=0):
    if args.arch == "CNN":
        if args.dataset == "stl10":
            model = MixCNNSTL_CL(dim).cuda()
        else:
            model = MixCNNCIFAR_CL(dim).cuda()
    elif args.arch == "WRN":
        model = WideResNet()
    elif args.arch == "CNN13":
        model = MetaCNN()
    else:
        if args.dataset == "stl10":
            model = PreActResNetMeta(
                PreActBlockMeta, [2, 2, 2, 2], num_classes=1, use_checkpoint=False
            ).cuda()
        elif args.dataset == "alz":
            model = PreActResNetMeta(
                PreActBottleneckMeta, [3, 4, 6, 3], num_classes=1, use_checkpoint=False
            ).cuda()
        else:
            model = PreActResNetMeta(
                PreActBlockMeta, [2, 2, 2, 2], num_classes=1, use_checkpoint=False
            ).cuda()
    model.cuda()
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    [args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(",")]
    args.positive_list = [int(item) for item in args.positive_list.split(",")]
    iterations = args.lr_decay_epochs.split(",")
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)
    trainer = Trainer(args)
    trainer.train()
