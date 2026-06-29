import os
import time
import warnings

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.atssd import atssd
from utils.tools import EarlyStopping, adjustment, adjust_learning_rate, visual

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.loss_path = os.path.join("Loss", args.model_id)
        os.makedirs(self.loss_path, exist_ok=True)
        self.lambda_contrastive = args.lambda_contrastive

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _forward_model(self, batch_x, batch_m):
        if self.args.model == "MSTGCNet":
            return self.model(batch_x, batch_m)
        outputs = self.model(batch_x, batch_m, None, None)
        balance_loss = torch.zeros((), device=batch_x.device)
        return outputs, balance_loss

    def vali(self, vali_data, vali_loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _, batch_m in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_m = batch_m.float().to(self.device)
                outputs, _ = self._forward_model(batch_x, batch_m)
                losses.append(criterion(outputs, batch_x).item())
        self.model.train()
        return float(np.average(losses)) if losses else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _, batch_m) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_m = batch_m.float().to(self.device)

                outputs, balance_loss = self._forward_model(batch_x, batch_m)
                loss = criterion(outputs, batch_x) + balance_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.average(train_loss)) if train_loss else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                "Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _point_scores(self, values, marks, windows):
        score_sum = np.zeros(len(values), dtype=np.float64)
        score_count = np.zeros(len(values), dtype=np.float64)
        if len(windows) == 0:
            return np.array([]), np.array([], dtype=bool)

        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(windows), self.args.batch_size):
                starts = windows[offset: offset + self.args.batch_size]
                batch_x = np.stack(
                    [values[start:start + self.args.seq_len] for start in starts],
                    axis=0,
                )
                batch_m = np.stack(
                    [marks[start:start + self.args.seq_len] for start in starts],
                    axis=0,
                )
                batch_x = torch.from_numpy(batch_x).float().to(self.device)
                batch_m = torch.from_numpy(batch_m).float().to(self.device)
                outputs, _ = self._forward_model(batch_x, batch_m)
                scores = torch.mean((batch_x - outputs) ** 2, dim=-1)
                scores = scores.detach().cpu().numpy()

                for start, score in zip(starts, scores):
                    end = start + self.args.seq_len
                    score_sum[start:end] += score
                    score_count[start:end] += 1

        covered = score_count > 0
        scores = np.zeros(len(values), dtype=np.float64)
        scores[covered] = score_sum[covered] / score_count[covered]
        return scores[covered], covered

    def _evaluate_predictions(self, gt, pred):
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(gt, pred)
        return accuracy, precision, recall, f_score, cm

    def test(self, setting, test=0):
        test_data, _ = self._get_data(flag="test")
        train_data, _ = self._get_data(flag="train")

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./checkpoints", setting, "checkpoint.pth"),
                    map_location=self.device,
                )
            )

        folder_path = os.path.join("./figure", setting)
        os.makedirs(folder_path, exist_ok=True)

        train_energy, _ = self._point_scores(
            train_data.train, train_data.train_mark, train_data.train_windows
        )
        test_energy, covered = self._point_scores(
            test_data.test, test_data.test_mark, test_data.test_windows
        )
        test_labels = test_data.test_labels[covered]

        if self.args.abl_thre == 1:
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
            raw_pred = (test_energy > threshold).astype(int)
        else:
            raw_pred, threshold = atssd(
                test_energy, window_size=self.args.winsize, alpha=self.args.alpha
            )

        gt = test_labels.astype(int)
        gt[gt != 0] = 1
        adjusted_gt, adjusted_pred = adjustment(gt.copy(), raw_pred.copy())

        res_path = os.path.join("./label_results", setting)
        os.makedirs(res_path, exist_ok=True)
        np.save(os.path.join(res_path, "threshold.npy"), threshold)
        np.save(os.path.join(res_path, "test_energy.npy"), test_energy)
        np.save(os.path.join(res_path, "test_labels.npy"), gt)
        np.save(os.path.join(res_path, "raw_pred.npy"), raw_pred)
        np.save(os.path.join(res_path, "adjusted_pred.npy"), adjusted_pred)

        visual(gt, test_energy, os.path.join(folder_path, "scores.pdf"))

        raw = self._evaluate_predictions(gt, raw_pred)
        adjusted = self._evaluate_predictions(adjusted_gt, adjusted_pred)

        raw_line = (
            "Raw      Accuracy : {:0.4f}, Precision : {:0.4f}, "
            "Recall : {:0.4f}, F-score : {:0.4f}"
        ).format(raw[0], raw[1], raw[2], raw[3])
        adjusted_line = (
            "Adjusted Accuracy : {:0.4f}, Precision : {:0.4f}, "
            "Recall : {:0.4f}, F-score : {:0.4f}"
        ).format(adjusted[0], adjusted[1], adjusted[2], adjusted[3])

        print("pred: ", raw_pred.shape)
        print("gt:   ", gt.shape)
        print(raw_line)
        print("Raw confusion matrix:", raw[4].tolist())
        print(adjusted_line)
        print("Adjusted confusion matrix:", adjusted[4].tolist())

        with open("result_anomaly_detection.txt", "a") as f:
            f.write(setting + "\n")
            f.write(raw_line + "\n")
            f.write("Raw confusion matrix: " + str(raw[4].tolist()) + "\n")
            f.write(adjusted_line + "\n")
            f.write("Adjusted confusion matrix: " + str(adjusted[4].tolist()) + "\n")
            f.write("\n")
