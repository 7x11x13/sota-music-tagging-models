# coding: utf-8
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from datasets import SplitType, get_dataset
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_utils import build_model, load_model, to_device, to_var


class Solver(object):
    def __init__(self, config):
        # data loader
        self.data_path = config.data_path
        self.dataset_name = config.dataset

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.load_model = config.load_model
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # build model
        self.model, self.input_length = build_model(
            self.model_type, self.dataset_name, config.n_stems, self.model_load_path
        )

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr, weight_decay=config.weight_decay
        )

        # Tensorboard
        self.writer = SummaryWriter()

        self.dataset = get_dataset(
            config.dataset,
            config.data_path,
            self.input_length,
            config.batch_size,
            SplitType.VALIDATE,
        )
        self.data_loader = DataLoader(
            dataset=get_dataset(
                config.dataset,
                config.data_path,
                self.input_length,
                config.batch_size,
                SplitType.TRAIN,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.num_workers,
        )

    def get_loss_function(self):
        return nn.BCELoss()

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = "adam"
        reconst_loss = self.get_loss_function()
        best_metric = 0
        drop_counter = 0

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = to_var(x)
                y = to_var(y)
                out = self.model(x)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar("Loss/train", loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch)

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(
                current_optimizer, drop_counter
            )

        print(
            "[%s] Train finished. Elapsed: %s"
            % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                datetime.timedelta(seconds=time.time() - start_t),
            )
        )

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == "adam" and drop_counter == 80:
            load_model(self.model, self.model_save_path)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                0.001,
                momentum=0.9,
                weight_decay=0.0001,
                nesterov=True,
            )
            current_optimizer = "sgd_1"
            drop_counter = 0
            print("sgd 1e-3")
        # first drop
        if current_optimizer == "sgd_1" and drop_counter == 20:
            load_model(self.model, self.model_save_path)
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0001
            current_optimizer = "sgd_2"
            drop_counter = 0
            print("sgd 1e-4")
        # second drop
        if current_optimizer == "sgd_2" and drop_counter == 20:
            load_model(self.model, self.model_save_path)
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.00001
            current_optimizer = "sgd_3"
            print("sgd 1e-5")
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({"model": model}, filename)

    def get_auc(self, est_array, gt_array):
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        print("roc_auc: %.4f" % roc_aucs)
        print("pr_auc: %.4f" % pr_aucs)
        return roc_aucs, pr_aucs

    def print_log(self, epoch, ctr, loss, start_t):
        if (ctr) % self.log_step == 0:
            print(
                "[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s"
                % (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch + 1,
                    self.n_epochs,
                    ctr,
                    len(self.data_loader),
                    loss.item(),
                    datetime.timedelta(seconds=time.time() - start_t),
                )
            )

    def validation(self, best_metric, epoch):
        roc_auc, pr_auc, loss = self.get_validation_score(epoch)
        score = 1 - loss
        if score > best_metric:
            print("best model!")
            best_metric = score
            torch.save(
                self.model.state_dict(),
                self.model_save_path,
            )
        return best_metric

    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = self.get_loss_function()
        index = 0
        for data in tqdm.tqdm(self.dataset.data_list):
            # load and split
            x = self.dataset.get_tensor(data)

            # ground truth
            ground_truth = self.dataset.get_ground_truth(data)

            # forward
            x = to_var(x)
            y = to_device(
                torch.tensor(
                    np.tile(ground_truth.astype("float32"), (self.batch_size, 1))
                )
            )
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)

            gt_array.append(ground_truth)
            index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print("loss: %.4f" % loss)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        self.writer.add_scalar("Loss/valid", loss, epoch)
        self.writer.add_scalar("AUC/ROC", roc_auc, epoch)
        self.writer.add_scalar("AUC/PR", pr_auc, epoch)
        return roc_auc, pr_auc, loss
