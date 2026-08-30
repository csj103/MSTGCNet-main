import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.metrics import precision_recall_fscore_support
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.atssd import atssd, causal_atssd
from utils.alfa_anomaly_types import (
    anomaly_type_sort_key,
    canonical_anomaly_type_from_row,
)
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

    @staticmethod
    def _reconstruction_loss(outputs, targets):
        return (outputs - targets).pow(2).flatten(start_dim=1).sum(dim=1).mean()

    @classmethod
    def _model_loss_from_output(cls, model_output, targets, args):
        if not isinstance(model_output, dict):
            reconstruction, aux_loss = model_output
            loss = cls._reconstruction_loss(reconstruction, targets) + aux_loss
            return loss, {
                "reconstruction_loss": float(
                    cls._reconstruction_loss(reconstruction, targets).detach().cpu()
                ),
                "nll_loss": 0.0,
                "dynamic_loss": 0.0,
                "last_loss": 0.0,
                "balance_loss": float(aux_loss.detach().cpu()),
            }

        if "observation_score" in model_output:
            observation_score = model_output["observation_score"]
            nll_loss = observation_score.flatten(start_dim=1).sum(dim=1).mean()
            last_loss = observation_score[:, -1, :].sum(dim=-1).mean()
        else:
            reconstruction = model_output["reconstruction"]
            nll_loss = cls._reconstruction_loss(reconstruction, targets)
            last_loss = (reconstruction[:, -1, :] - targets[:, -1, :]).pow(2).sum(
                dim=-1
            ).mean()

        dynamic_score = model_output.get("dynamic_score")
        if dynamic_score is None:
            dynamic_loss = torch.zeros((), dtype=targets.dtype, device=targets.device)
        else:
            dynamic_loss = dynamic_score.mean()
        balance_loss = model_output.get(
            "balance_loss",
            torch.zeros((), dtype=targets.dtype, device=targets.device),
        )

        loss = (
            nll_loss
            + getattr(args, "dynamic_loss_weight", 0.0) * dynamic_loss
            + getattr(args, "last_loss_weight", 0.0) * last_loss
            + getattr(args, "balance_loss_weight", getattr(args, "loss_coef", 0.0))
            * balance_loss
        )
        return loss, {
            "reconstruction_loss": float(nll_loss.detach().cpu()),
            "nll_loss": float(nll_loss.detach().cpu()),
            "dynamic_loss": float(dynamic_loss.detach().cpu()),
            "last_loss": float(last_loss.detach().cpu()),
            "balance_loss": float(balance_loss.detach().cpu()),
        }

    def _select_criterion(self):
        return self._reconstruction_loss

    def _forward_model(self, batch_x, batch_m):
        if self.args.model == "DTSGAD":
            return self.model(batch_x, batch_m, return_dict=True)
        if self.args.model == "MSTGCNet":
            outputs, balance_loss = self.model(batch_x, batch_m)
            return {
                "reconstruction": outputs,
                "balance_loss": balance_loss,
            }
        outputs = self.model(batch_x, batch_m, None, None)
        return {
            "reconstruction": outputs,
            "balance_loss": torch.zeros((), device=batch_x.device),
        }

    @staticmethod
    def _reconstruction_from_output(model_output):
        if isinstance(model_output, dict):
            return model_output["reconstruction"]
        return model_output[0]

    @staticmethod
    def _full_scores_from_output(model_output, batch_x):
        if isinstance(model_output, dict) and "total_score" in model_output:
            return model_output["total_score"]
        reconstruction = Exp_Anomaly_Detection._reconstruction_from_output(
            model_output
        )
        return torch.mean((batch_x - reconstruction) ** 2, dim=-1)

    def vali(self, vali_data, vali_loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _, batch_m in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_m = batch_m.float().to(self.device)
                model_output = self._forward_model(batch_x, batch_m)
                loss, _ = self._model_loss_from_output(
                    model_output, batch_x, self.args
                )
                losses.append(loss.item())
        self.model.train()
        return float(np.average(losses)) if losses else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        config = {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in vars(self.args).items()
        }
        with open(
            os.path.join(path, "experiment_config.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {"setting": setting, "args": config},
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            reconstruction_losses = []
            balance_losses = []
            model_for_stats = (
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )
            blocks_for_stats = getattr(model_for_stats, "blocks", [])
            routing_loads = [
                np.zeros(block.num_experts, dtype=np.float64)
                for block in blocks_for_stats
            ]
            routing_entropy = np.zeros(len(blocks_for_stats), dtype=np.float64)
            routing_samples = 0
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _, batch_m) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_m = batch_m.float().to(self.device)

                model_output = self._forward_model(batch_x, batch_m)
                loss, loss_parts = self._model_loss_from_output(
                    model_output, batch_x, self.args
                )
                train_loss.append(loss.item())
                reconstruction_losses.append(loss_parts["reconstruction_loss"])
                balance_losses.append(loss_parts["balance_loss"])
                routing_samples += batch_x.size(0)
                for block_idx, block in enumerate(blocks_for_stats):
                    router = block.router
                    if router.last_expert_load is not None:
                        routing_loads[block_idx] += (
                            router.last_expert_load.detach().cpu().numpy()
                        )
                    if router.last_entropy is not None:
                        routing_entropy[block_idx] += (
                            float(router.last_entropy) * batch_x.size(0)
                        )

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
            reconstruction_loss = (
                float(np.average(reconstruction_losses))
                if reconstruction_losses
                else 0.0
            )
            balance_loss = (
                float(np.average(balance_losses)) if balance_losses else 0.0
            )
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                "Rec Loss: {3:.7f} Balance Loss: {4:.7f} "
                "Vali Loss: {5:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    reconstruction_loss,
                    balance_loss,
                    vali_loss,
                )
            )
            if routing_samples:
                for block_idx, load in enumerate(routing_loads):
                    total = load.sum()
                    fractions = load / total if total else load
                    entropy = routing_entropy[block_idx] / routing_samples
                    print(
                        "  Router block {} | load={} fractions={} entropy={:.6f}".format(
                            block_idx + 1,
                            load.astype(int).tolist(),
                            np.round(fractions, 4).tolist(),
                            entropy,
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
                model_output = self._forward_model(batch_x, batch_m)
                scores = self._full_scores_from_output(model_output, batch_x)
                scores = scores.detach().cpu().numpy()

                for start, score in zip(starts, scores):
                    end = start + self.args.seq_len
                    score_sum[start:end] += score
                    score_count[start:end] += 1

        covered = score_count > 0
        scores = np.zeros(len(values), dtype=np.float64)
        scores[covered] = score_sum[covered] / score_count[covered]
        return scores[covered], covered

    def _causal_last_feature_scale(self, values, marks, windows):
        error_sum = np.zeros(values.shape[1], dtype=np.float64)
        count = 0
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(windows), self.args.batch_size):
                starts = windows[offset: offset + self.args.batch_size]
                batch_x = torch.from_numpy(
                    np.stack(
                        [values[start:start + self.args.seq_len] for start in starts]
                    )
                ).float().to(self.device)
                batch_m = torch.from_numpy(
                    np.stack(
                        [marks[start:start + self.args.seq_len] for start in starts]
                    )
                ).float().to(self.device)
                model_output = self._forward_model(batch_x, batch_m)
                outputs = self._reconstruction_from_output(model_output)
                errors = (batch_x[:, -1, :] - outputs[:, -1, :]).pow(2)
                error_sum += errors.sum(dim=0).cpu().numpy()
                count += errors.size(0)
        if count == 0:
            return np.ones(values.shape[1], dtype=np.float64)
        return np.maximum(error_sum / count, 1e-8)

    def _causal_last_scores(
        self,
        values,
        marks,
        labels,
        windows,
        meta=None,
        feature_scale=None,
    ):
        energies = []
        point_indices = []
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
                model_output = self._forward_model(batch_x, batch_m)
                if (
                    isinstance(model_output, dict)
                    and "total_score" in model_output
                    and feature_scale is None
                ):
                    scores = model_output["total_score"][:, -1]
                else:
                    outputs = self._reconstruction_from_output(model_output)
                    errors = (batch_x[:, -1, :] - outputs[:, -1, :]) ** 2
                    if feature_scale is not None:
                        scale = torch.as_tensor(
                            feature_scale, device=errors.device, dtype=errors.dtype
                        )
                        errors = errors / scale
                    scores = torch.mean(errors, dim=-1)
                energies.append(scores.detach().cpu().numpy())
                point_indices.extend(
                    start + self.args.seq_len - 1 for start in starts
                )

        if not energies:
            empty = np.array([], dtype=int)
            return np.array([]), empty, empty, empty

        indices = np.asarray(point_indices, dtype=int)
        targets = np.asarray(labels)[indices]
        if meta is not None and "segment_id" in meta.columns:
            segment_ids = meta.iloc[indices]["segment_id"].to_numpy()
        else:
            segment_ids = np.zeros(len(indices), dtype=int)
        return np.concatenate(energies), targets, indices, segment_ids

    @staticmethod
    def _evaluate_predictions(gt, pred):
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(gt, pred)
        return accuracy, precision, recall, f_score, cm

    def _paper_nonoverlap_scores(self, values, marks, labels):
        starts = list(
            range(0, max(len(values) - self.args.seq_len + 1, 0), self.args.seq_len)
        )
        energies = []
        targets = []
        indices = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(starts), self.args.batch_size):
                batch_starts = starts[offset: offset + self.args.batch_size]
                batch_x = np.stack(
                    [
                        values[start:start + self.args.seq_len]
                        for start in batch_starts
                    ],
                    axis=0,
                )
                batch_m = np.stack(
                    [marks[start:start + self.args.seq_len] for start in batch_starts],
                    axis=0,
                )
                batch_x = torch.from_numpy(batch_x).float().to(self.device)
                batch_m = torch.from_numpy(batch_m).float().to(self.device)
                model_output = self._forward_model(batch_x, batch_m)
                scores = self._full_scores_from_output(model_output, batch_x)
                energies.append(scores.detach().cpu().numpy())
                targets.append(
                    np.stack(
                        [
                            labels[start:start + self.args.seq_len]
                            for start in batch_starts
                        ],
                        axis=0,
                    )
                )
                indices.append(
                    np.stack(
                        [
                            np.arange(start, start + self.args.seq_len)
                            for start in batch_starts
                        ],
                        axis=0,
                    )
                )

        if not energies:
            empty = np.array([], dtype=int)
            return np.array([]), empty, empty
        return (
            np.concatenate(energies, axis=0).reshape(-1),
            np.concatenate(targets, axis=0).reshape(-1),
            np.concatenate(indices, axis=0).reshape(-1),
        )

    @staticmethod
    def _segment_boundaries(segment_ids, total):
        if total == 0:
            return np.array([0], dtype=int)
        if segment_ids is None:
            return np.array([0, total], dtype=int)
        segment_ids = np.asarray(segment_ids)
        return np.flatnonzero(np.r_[True, segment_ids[1:] != segment_ids[:-1], True])

    @classmethod
    def _adjust_by_segment(cls, gt, pred, segment_ids=None):
        adjusted_gt = gt.copy()
        adjusted_pred = pred.copy()
        boundaries = cls._segment_boundaries(segment_ids, len(gt))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment_gt, segment_pred = adjustment(
                adjusted_gt[start:end], adjusted_pred[start:end]
            )
            adjusted_gt[start:end] = segment_gt
            adjusted_pred[start:end] = segment_pred
        return adjusted_gt, adjusted_pred

    @classmethod
    def _event_hits(cls, gt, pred, segment_ids=None):
        if segment_ids is not None:
            hits = 0
            total_events = 0
            boundaries = cls._segment_boundaries(segment_ids, len(gt))
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                segment_hits, segment_total = cls._event_hits(
                    gt[start:end], pred[start:end]
                )
                hits += segment_hits
                total_events += segment_total
            return hits, total_events

        padded = np.pad(gt.astype(int), (1, 1))
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        hits = sum(np.any(pred[start:end] == 1) for start, end in zip(starts, ends))
        return int(hits), int(len(starts))

    @staticmethod
    def _canonical_anomaly_type(row):
        return canonical_anomaly_type_from_row(row)

    @classmethod
    def _score_anomaly_types(cls, meta, score_indices):
        if meta is None or score_indices is None:
            return None
        score_indices = np.asarray(score_indices, dtype=int)
        if score_indices.size == 0:
            return np.array([], dtype=object)
        if score_indices.min() < 0 or score_indices.max() >= len(meta):
            raise IndexError("score_indices are not aligned with test metadata")
        rows = meta.iloc[score_indices]
        return rows.apply(cls._canonical_anomaly_type, axis=1).to_numpy()

    @classmethod
    def _metrics_by_anomaly_type(cls, gt, pred, meta, score_indices):
        anomaly_types = cls._score_anomaly_types(meta, score_indices)
        if anomaly_types is None:
            return []

        rows = []
        for name in sorted(set(anomaly_types), key=anomaly_type_sort_key):
            mask = anomaly_types == name
            if int(gt[mask].sum()) == 0:
                continue
            accuracy, precision, recall, f_score, cm = cls._evaluate_predictions(
                gt[mask], pred[mask]
            )
            rows.append(
                {
                    "Anomaly Type": name,
                    "Points": int(mask.sum()),
                    "Anomalies": int(gt[mask].sum()),
                    "Predicted": int(pred[mask].sum()),
                    "Accuracy": float(accuracy),
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F-score": float(f_score),
                    "Confusion Matrix": cm.tolist(),
                }
            )
        return rows

    @staticmethod
    def _format_anomaly_type_table(title, rows):
        if not rows:
            return [title + ": unavailable"]
        lines = [title]
        header = (
            "  {name:<30} {acc:>8} {pre:>9} {rec:>8} {f1:>8} "
            "{points:>8} {anom:>9} {pred:>9}"
        ).format(
            name="Anomaly Type",
            acc="Accuracy",
            pre="Precision",
            rec="Recall",
            f1="F-score",
            points="Points",
            anom="Anomalies",
            pred="Pred",
        )
        lines.append(header)
        for row in rows:
            lines.append(
                "  {name:<30} {acc:8.4f} {pre:9.4f} {rec:8.4f} {f1:8.4f} "
                "{points:8d} {anom:9d} {pred:9d}".format(
                    name=row["Anomaly Type"][:30],
                    acc=row["Accuracy"],
                    pre=row["Precision"],
                    rec=row["Recall"],
                    f1=row["F-score"],
                    points=row["Points"],
                    anom=row["Anomalies"],
                    pred=row["Predicted"],
                )
            )
        return lines

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

        folder_path = os.path.join("./figure", setting, self.args.score_mode)
        os.makedirs(folder_path, exist_ok=True)

        score_indices = None
        score_segments = None
        feature_scale = None
        if self.args.score_normalization == "train_feature":
            if self.args.score_mode != "causal_last":
                raise ValueError(
                    "train_feature score normalization requires causal_last mode"
                )
            feature_scale = self._causal_last_feature_scale(
                train_data.train,
                train_data.train_mark,
                train_data.train_windows,
            )
        if self.args.score_mode == "causal_last":
            test_energy, test_labels, score_indices, score_segments = (
                self._causal_last_scores(
                    test_data.test,
                    test_data.test_mark,
                    test_data.test_labels,
                    test_data.test_windows,
                    test_data.test_meta,
                    feature_scale,
                )
            )
        elif self.args.score_mode == "paper_nonoverlap":
            test_energy, test_labels, score_indices = self._paper_nonoverlap_scores(
                test_data.test,
                test_data.test_mark,
                test_data.test_labels,
            )
        else:
            test_energy, covered = self._point_scores(
                test_data.test, test_data.test_mark, test_data.test_windows
            )
            test_labels = test_data.test_labels[covered]
            score_indices = np.flatnonzero(covered)

        if self.args.threshold_method == "percentile":
            if self.args.score_mode == "causal_last":
                train_energy, _, _, _ = self._causal_last_scores(
                    train_data.train,
                    train_data.train_mark,
                    np.zeros(len(train_data.train), dtype=int),
                    train_data.train_windows,
                    train_data.train_meta,
                    feature_scale,
                )
            else:
                train_energy, _ = self._point_scores(
                    train_data.train, train_data.train_mark, train_data.train_windows
                )
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
            raw_pred = (test_energy > threshold).astype(int)
        elif self.args.threshold_method == "atssd":
            raw_pred, threshold = atssd(
                test_energy,
                window_size=self.args.winsize,
                alpha=self.args.alpha,
                segment_ids=score_segments,
            )
        else:
            raw_pred, threshold = causal_atssd(
                test_energy,
                window_size=self.args.winsize,
                alpha=self.args.alpha,
                segment_ids=score_segments,
                confirmation=self.args.alarm_confirmation,
                latch_alarm=bool(self.args.latch_alarm),
                adaptation_clip=self.args.threshold_adaptation_clip,
            )

        gt = test_labels.astype(int)
        gt[gt != 0] = 1
        adjusted_gt, adjusted_pred = self._adjust_by_segment(
            gt, raw_pred, score_segments
        )

        res_path = os.path.join("./label_results", setting, self.args.score_mode)
        os.makedirs(res_path, exist_ok=True)
        np.save(os.path.join(res_path, "threshold.npy"), threshold)
        np.save(os.path.join(res_path, "test_energy.npy"), test_energy)
        np.save(os.path.join(res_path, "test_labels.npy"), gt)
        np.save(os.path.join(res_path, "test_indices.npy"), score_indices)
        np.save(os.path.join(res_path, "raw_pred.npy"), raw_pred)
        np.save(os.path.join(res_path, "adjusted_pred.npy"), adjusted_pred)
        if feature_scale is not None:
            np.save(os.path.join(res_path, "feature_scale.npy"), feature_scale)

        visual(gt, test_energy, os.path.join(folder_path, "scores.pdf"))

        raw = self._evaluate_predictions(gt, raw_pred)
        adjusted = self._evaluate_predictions(adjusted_gt, adjusted_pred)
        roc_auc = roc_auc_score(gt, test_energy)
        pr_auc = average_precision_score(gt, test_energy)
        event_hits, event_total = self._event_hits(gt, raw_pred, score_segments)
        raw_type_rows = self._metrics_by_anomaly_type(
            gt, raw_pred, test_data.test_meta, score_indices
        )
        adjusted_type_rows = self._metrics_by_anomaly_type(
            adjusted_gt, adjusted_pred, test_data.test_meta, score_indices
        )
        if raw_type_rows:
            pd.DataFrame(raw_type_rows).to_csv(
                os.path.join(res_path, "raw_metrics_by_anomaly_type.csv"),
                index=False,
            )
        if adjusted_type_rows:
            pd.DataFrame(adjusted_type_rows).to_csv(
                os.path.join(res_path, "adjusted_metrics_by_anomaly_type.csv"),
                index=False,
            )

        raw_line = (
            "Raw      Accuracy : {:0.4f}, Precision : {:0.4f}, "
            "Recall : {:0.4f}, F-score : {:0.4f}"
        ).format(raw[0], raw[1], raw[2], raw[3])
        adjusted_line = (
            "Point-adjusted (GT-assisted) Accuracy : {:0.4f}, Precision : {:0.4f}, "
            "Recall : {:0.4f}, F-score : {:0.4f}"
        ).format(adjusted[0], adjusted[1], adjusted[2], adjusted[3])
        quality_line = "Score ROC-AUC : {:0.4f}, PR-AUC : {:0.4f}".format(
            roc_auc, pr_auc
        )

        print("score mode:", self.args.score_mode)
        print("evaluated original points:", len(score_indices))
        print("pred: ", raw_pred.shape)
        print("gt:   ", gt.shape)
        print(quality_line)
        print(raw_line)
        print("Raw confusion matrix:", raw[4].tolist())
        print("Raw event hits: {}/{}".format(event_hits, event_total))
        print(adjusted_line)
        print("Point-adjusted confusion matrix:", adjusted[4].tolist())
        print("Note: point-adjusted metrics use ground-truth segment boundaries.")
        for line in self._format_anomaly_type_table(
            "Raw metrics by anomaly type", raw_type_rows
        ):
            print(line)
        for line in self._format_anomaly_type_table(
            "Point-adjusted metrics by anomaly type", adjusted_type_rows
        ):
            print(line)

        with open("result_anomaly_detection.txt", "a") as f:
            f.write(setting + " [score_mode=" + self.args.score_mode + "]\n")
            f.write(quality_line + "\n")
            f.write(raw_line + "\n")
            f.write("Raw confusion matrix: " + str(raw[4].tolist()) + "\n")
            f.write("Raw event hits: {}/{}\n".format(event_hits, event_total))
            f.write(adjusted_line + "\n")
            f.write(
                "Point-adjusted confusion matrix: "
                + str(adjusted[4].tolist())
                + "\n"
            )
            for line in self._format_anomaly_type_table(
                "Raw metrics by anomaly type", raw_type_rows
            ):
                f.write(line + "\n")
            for line in self._format_anomaly_type_table(
                "Point-adjusted metrics by anomaly type", adjusted_type_rows
            ):
                f.write(line + "\n")
            f.write("\n")
