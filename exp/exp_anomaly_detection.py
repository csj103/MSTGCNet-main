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
from utils.threshold_diagnostics import (
    freeze_atssd,
    lagged_atssd,
    median_mad_threshold,
    oracle_best_f1,
    val_quantile_threshold,
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

    @staticmethod
    def _masked_average(values, loss_mask):
        mask = loss_mask.to(device=values.device, dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _random_channel_masks(batch_x, mask_ratio):
        ratio = float(mask_ratio)
        if ratio <= 0.0:
            keep = torch.ones_like(batch_x)
            masked = torch.zeros_like(batch_x)
            return keep, masked

        bsz, seq_len, num_vars = batch_x.shape
        masked_channels = int(round(num_vars * ratio))
        masked_channels = max(1, min(masked_channels, num_vars))
        channel_scores = torch.rand(
            bsz,
            num_vars,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        masked_indices = torch.topk(
            channel_scores,
            k=masked_channels,
            dim=-1,
        ).indices
        channel_mask = torch.zeros(
            bsz,
            num_vars,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        channel_mask.scatter_(1, masked_indices, 1.0)
        loss_mask = channel_mask.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
        input_mask = 1.0 - loss_mask
        return input_mask, loss_mask

    @staticmethod
    def _is_next_step_prediction(args):
        return (
            getattr(args, "dtsgad_objective", "reconstruction")
            == "next_step_prediction"
        )

    @classmethod
    def _model_loss_from_output(cls, model_output, targets, args, loss_mask=None):
        if not isinstance(model_output, dict):
            reconstruction, aux_loss = model_output
            if loss_mask is None:
                reconstruction_loss = cls._reconstruction_loss(
                    reconstruction,
                    targets,
                )
            else:
                reconstruction_loss = cls._masked_average(
                    (reconstruction - targets).pow(2),
                    loss_mask,
                )
            loss = reconstruction_loss + aux_loss
            return loss, {
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                "nll_loss": 0.0,
                "dynamic_loss": 0.0,
                "last_loss": 0.0,
                "balance_loss": float(aux_loss.detach().cpu()),
            }

        if "prediction_score" in model_output:
            prediction_loss = model_output["prediction_score"].mean()
            balance_loss = model_output.get(
                "balance_loss",
                torch.zeros((), dtype=targets.dtype, device=targets.device),
            )
            return prediction_loss, {
                "reconstruction_loss": float(prediction_loss.detach().cpu()),
                "nll_loss": 0.0,
                "dynamic_loss": 0.0,
                "last_loss": 0.0,
                "balance_loss": float(balance_loss.detach().cpu()),
            }

        if "observation_score" in model_output:
            observation_score = model_output["observation_score"]
            if loss_mask is None:
                nll_loss = observation_score.flatten(start_dim=1).sum(dim=1).mean()
                last_loss = observation_score[:, -1, :].sum(dim=-1).mean()
            else:
                nll_loss = cls._masked_average(observation_score, loss_mask)
                last_loss = cls._masked_average(
                    observation_score[:, -1, :],
                    loss_mask[:, -1, :],
                )
        else:
            reconstruction = model_output["reconstruction"]
            if loss_mask is None:
                nll_loss = cls._reconstruction_loss(reconstruction, targets)
                last_loss = (
                    reconstruction[:, -1, :] - targets[:, -1, :]
                ).pow(2).sum(dim=-1).mean()
            else:
                nll_loss = cls._masked_average(
                    (reconstruction - targets).pow(2),
                    loss_mask,
                )
                last_loss = cls._masked_average(
                    (reconstruction[:, -1, :] - targets[:, -1, :]).pow(2),
                    loss_mask[:, -1, :],
                )

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

    def _training_masks(self, batch_x):
        mask_ratio = float(getattr(self.args, "mask_ratio", 0.0))
        if self.args.model != "DTSGAD" or mask_ratio <= 0.0:
            return None, None
        return self._random_channel_masks(batch_x, mask_ratio)

    def _loss_targets(self, batch_x, batch_y):
        if self._is_next_step_prediction(self.args):
            return batch_y
        return batch_x

    def _forward_model(
        self,
        batch_x,
        batch_m,
        collect_diagnostics=False,
        force_expert_index=None,
        input_mask=None,
        target_next=None,
    ):
        if self.args.model == "DTSGAD":
            return self.model(
                batch_x,
                batch_m,
                return_dict=True,
                collect_diagnostics=collect_diagnostics,
                force_expert_index=force_expert_index,
                input_mask=input_mask,
                target_next=target_next,
            )
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

    def _score_forward_model(
        self,
        batch_x,
        batch_m,
        collect_diagnostics=False,
        force_expert_index=None,
        target_next=None,
    ):
        if (
            self.args.model != "DTSGAD"
            or getattr(self.args, "mask_eval_mode", "none") != "channelwise"
        ):
            return self._forward_model(
                batch_x,
                batch_m,
                collect_diagnostics=collect_diagnostics,
                force_expert_index=force_expert_index,
                target_next=target_next,
            )

        outputs = []
        bsz, seq_len, num_vars = batch_x.shape
        for feature_index in range(num_vars):
            input_mask = torch.ones_like(batch_x)
            input_mask[:, :, feature_index] = 0.0
            outputs.append(
                self._forward_model(
                    batch_x,
                    batch_m,
                    collect_diagnostics=collect_diagnostics and feature_index == 0,
                    force_expert_index=force_expert_index,
                    input_mask=input_mask,
                    target_next=target_next,
                )
            )

        first = outputs[0]
        composed = {}
        for key in (
            "observation_score",
            "reconstruction_norm",
            "reconstruction_logvar",
            "reconstruction",
        ):
            if key not in first:
                continue
            value = torch.zeros_like(first[key])
            for feature_index, output in enumerate(outputs):
                value[:, :, feature_index] = output[key][:, :, feature_index]
            composed[key] = value

        composed["target_norm"] = first.get("target_norm", batch_x)
        composed["input_norm"] = first.get("input_norm", batch_x)
        if "dynamic_score" in first and first["dynamic_score"] is not None:
            composed["dynamic_score"] = torch.stack(
                [output["dynamic_score"] for output in outputs],
                dim=0,
            ).mean(dim=0)
        if "observation_score" in composed:
            topk = max(1, min(getattr(self.args, "obs_topk", 1), num_vars))
            channel_score = torch.topk(
                composed["observation_score"],
                k=topk,
                dim=-1,
            ).values.mean(dim=-1)
            dynamic_score = composed.get(
                "dynamic_score",
                torch.zeros_like(channel_score),
            )
            if getattr(self.args, "score_fusion", "dual") == "obs":
                composed["total_score"] = channel_score
            elif getattr(self.args, "score_fusion", "dual") == "dyn":
                composed["total_score"] = dynamic_score
            else:
                composed["total_score"] = (
                    channel_score
                    + getattr(self.args, "dynamic_score_weight", 0.0)
                    * dynamic_score
                )
        composed["balance_loss"] = first.get(
            "balance_loss",
            torch.zeros((), dtype=batch_x.dtype, device=batch_x.device),
        )
        for key in (
            "posterior_mu",
            "posterior_logvar",
            "prior_mu",
            "prior_logvar",
            "adjacencies",
            "router_diagnostics",
        ):
            if key in first:
                composed[key] = first[key]
        return composed

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

    @staticmethod
    def _dtsgad_causal_last_components(model_output, obs_topk):
        if not isinstance(model_output, dict):
            return {}
        observation_score = model_output.get("observation_score")
        if observation_score is None:
            return {}

        topk = max(1, min(int(obs_topk), observation_score.size(-1)))
        last_observation = observation_score[:, -1, :]
        last_topk = torch.topk(last_observation, k=topk, dim=-1).values.mean(dim=-1)
        components = {
            "s_obs_feature": last_observation.detach().cpu().numpy(),
            "s_obs_topk": last_topk.detach().cpu().numpy(),
        }

        dynamic_score = model_output.get("dynamic_score")
        if dynamic_score is not None:
            components["s_dyn"] = dynamic_score[:, -1].detach().cpu().numpy()

        total_score = model_output.get("total_score")
        if total_score is not None:
            components["s_total"] = total_score[:, -1].detach().cpu().numpy()

        target_norm = model_output.get("target_norm")
        reconstruction_norm = model_output.get(
            "reconstruction_norm",
            model_output.get("prediction_norm"),
        )
        logvar = model_output.get("reconstruction_logvar")
        prediction_score = model_output.get("prediction_score")
        if prediction_score is not None:
            last_mse = prediction_score
            components.update(
                {
                    "s_mse_feature": last_mse.detach().cpu().numpy(),
                    "s_mse_topk": torch.topk(
                        last_mse,
                        k=topk,
                        dim=-1,
                    ).values.mean(dim=-1).detach().cpu().numpy(),
                }
            )
        if (
            prediction_score is None
            and target_norm is not None
            and reconstruction_norm is not None
            and logvar is not None
        ):
            if target_norm.ndim == 3:
                last_target = target_norm[:, -1, :]
            else:
                last_target = target_norm
            if reconstruction_norm.ndim == 3:
                last_mu = reconstruction_norm[:, -1, :]
            else:
                last_mu = reconstruction_norm
            last_logvar = logvar[:, -1, :] if logvar.ndim == 3 else logvar
            last_mse = (last_target - last_mu).pow(2)
            last_stdres = last_mse / last_logvar.exp().clamp_min(1e-6)
            last_nll = last_observation
            components.update(
                {
                    "s_mse_feature": last_mse.detach().cpu().numpy(),
                    "s_stdres_feature": last_stdres.detach().cpu().numpy(),
                    "s_unc_feature": last_logvar.detach().cpu().numpy(),
                    "s_nll_feature": last_nll.detach().cpu().numpy(),
                    "s_mse_topk": torch.topk(
                        last_mse,
                        k=topk,
                        dim=-1,
                    ).values.mean(dim=-1).detach().cpu().numpy(),
                    "s_stdres_topk": torch.topk(
                        last_stdres,
                        k=topk,
                        dim=-1,
                    ).values.mean(dim=-1).detach().cpu().numpy(),
                    "s_unc_mean": last_logvar.mean(dim=-1).detach().cpu().numpy(),
                    "s_unc_topk": torch.topk(
                        last_logvar,
                        k=topk,
                        dim=-1,
                    ).values.mean(dim=-1).detach().cpu().numpy(),
                    "s_nll_topk": torch.topk(
                        last_nll,
                        k=topk,
                        dim=-1,
                    ).values.mean(dim=-1).detach().cpu().numpy(),
                }
            )

        router_diagnostics = model_output.get("router_diagnostics", [])
        if router_diagnostics:
            components["router_gates"] = torch.stack(
                [block["gates"] for block in router_diagnostics],
                dim=1,
            ).detach().cpu().numpy()
            components["expert_pair_cosine"] = torch.stack(
                [block["expert_pair_cosine"] for block in router_diagnostics],
                dim=1,
            ).detach().cpu().numpy()
            components["expert_pre_graph_pair_cosine"] = torch.stack(
                [
                    block["expert_pre_graph_pair_cosine"]
                    for block in router_diagnostics
                ],
                dim=1,
            ).detach().cpu().numpy()
            components["expert_post_graph_pair_cosine"] = torch.stack(
                [
                    block["expert_post_graph_pair_cosine"]
                    for block in router_diagnostics
                ],
                dim=1,
            ).detach().cpu().numpy()
            components["graph_trace_pair_cosine"] = torch.stack(
                [
                    block["graph_trace_pair_cosine"]
                    for block in router_diagnostics
                ],
                dim=1,
            ).detach().cpu().numpy()
            adjacency = torch.stack(
                [block["adjacency"] for block in router_diagnostics],
                dim=0,
            )
            adjacency = adjacency.unsqueeze(0).expand(
                observation_score.size(0),
                -1,
                -1,
                -1,
                -1,
            )
            components["adjacency"] = adjacency.detach().cpu().numpy()

        return components

    @staticmethod
    def _latent_vectors_from_output(model_output):
        if not isinstance(model_output, dict):
            raise ValueError("latent diagnostics require return_dict model output")
        posterior_mu = model_output.get("posterior_mu")
        if posterior_mu is None:
            raise ValueError(
                "latent diagnostics require posterior_mu. Use the original "
                "DTSGAD Full reconstruction model with dynamic score enabled."
            )
        if posterior_mu.ndim == 3:
            return posterior_mu[:, -1, :]
        if posterior_mu.ndim == 2:
            return posterior_mu
        raise ValueError("posterior_mu must have shape [B, L, D] or [B, D]")

    @staticmethod
    def _latent_center(latents):
        latents = np.asarray(latents, dtype=np.float64)
        if latents.ndim != 2 or latents.shape[0] == 0:
            raise ValueError("latents must have shape [num_points, latent_dim]")
        return latents.mean(axis=0)

    @staticmethod
    def _latent_distance_scores(latents, center):
        latents = np.asarray(latents, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        if latents.ndim != 2:
            raise ValueError("latents must have shape [num_points, latent_dim]")
        if center.shape != (latents.shape[1],):
            raise ValueError("center must have shape [latent_dim]")
        return np.square(latents - center).sum(axis=1)

    @classmethod
    def _latent_metrics_table(cls, labels, latent_scores):
        labels = np.asarray(labels).astype(int)
        labels[labels != 0] = 1
        latent_scores = np.asarray(latent_scores, dtype=np.float64)
        return [
            {
                "score_name": "S_latent_center_l2",
                "ROC-AUC": cls._safe_roc_auc(labels, latent_scores),
                "PR-AUC": cls._safe_pr_auc(labels, latent_scores),
                "normal_mean": float(np.mean(latent_scores[labels == 0]))
                if np.any(labels == 0)
                else float("nan"),
                "anomaly_mean": float(np.mean(latent_scores[labels == 1]))
                if np.any(labels == 1)
                else float("nan"),
            }
        ]

    def _causal_last_latents(self, values, marks, labels, windows, meta=None):
        latents = []
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
                model_output = self._score_forward_model(batch_x, batch_m)
                z_t = self._latent_vectors_from_output(model_output)
                latents.append(z_t.detach().cpu().numpy())
                point_indices.extend(start + self.args.seq_len - 1 for start in starts)

        if not latents:
            empty = np.array([], dtype=int)
            return np.empty((0, 0), dtype=np.float64), empty, empty, empty

        indices = np.asarray(point_indices, dtype=int)
        targets = np.asarray(labels)[indices]
        if meta is not None and "segment_id" in meta.columns:
            segment_ids = meta.iloc[indices]["segment_id"].to_numpy()
        else:
            segment_ids = np.zeros(len(indices), dtype=int)
        return np.concatenate(latents, axis=0), targets, indices, segment_ids

    def _save_latent_diagnostics(
        self,
        res_path,
        train_data,
        test_data,
        score_indices,
    ):
        if self.args.model != "DTSGAD" or self.args.score_mode != "causal_last":
            return []

        train_latents, _, _, _ = self._causal_last_latents(
            train_data.train,
            train_data.train_mark,
            np.zeros(len(train_data.train), dtype=int),
            train_data.train_windows,
            train_data.train_meta,
        )
        center = self._latent_center(train_latents)
        test_latents, test_labels, latent_indices, _ = self._causal_last_latents(
            test_data.test,
            test_data.test_mark,
            test_data.test_labels,
            test_data.test_windows,
            test_data.test_meta,
        )
        latent_scores = self._latent_distance_scores(test_latents, center)
        gt = np.asarray(test_labels).astype(int)
        gt[gt != 0] = 1

        output_dir = os.path.join(res_path, "latent_diagnostics")
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "latent_center.npy"), center)
        np.save(os.path.join(output_dir, "train_latent.npy"), train_latents)
        np.save(os.path.join(output_dir, "test_latent.npy"), test_latents)
        np.save(os.path.join(output_dir, "latent_scores.npy"), latent_scores)
        np.save(os.path.join(output_dir, "latent_test_labels.npy"), gt)
        np.save(os.path.join(output_dir, "latent_test_indices.npy"), latent_indices)

        rows = self._latent_metrics_table(gt, latent_scores)
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "latent_metrics.csv"),
            index=False,
        )

        type_rows = self._score_auc_by_anomaly_type(
            gt,
            latent_scores,
            test_data.test_meta,
            latent_indices if score_indices is None else latent_indices,
        )
        if type_rows:
            pd.DataFrame(type_rows).to_csv(
                os.path.join(output_dir, "latent_metrics_by_anomaly_type.csv"),
                index=False,
            )
        return rows

    def vali(self, vali_data, vali_loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_m in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_m = batch_m.float().to(self.device)
                input_mask, loss_mask = self._training_masks(batch_x)
                model_output = self._forward_model(
                    batch_x,
                    batch_m,
                    input_mask=input_mask,
                    target_next=batch_y
                    if self._is_next_step_prediction(self.args)
                    else None,
                )
                loss, _ = self._model_loss_from_output(
                    model_output,
                    self._loss_targets(batch_x, batch_y),
                    self.args,
                    loss_mask=loss_mask,
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
        printed_prediction_shape = False
        if self._is_next_step_prediction(self.args) and train_data.train_windows:
            start = int(train_data.train_windows[0])
            end = start + self.args.seq_len - 1
            target_index = end + int(getattr(self.args, "target_horizon", 1))
            print(
                "M2 window check | input index: {}~{} | target index: {}".format(
                    start,
                    end,
                    target_index,
                )
            )

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

            for i, (batch_x, batch_y, batch_m) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_m = batch_m.float().to(self.device)

                input_mask, loss_mask = self._training_masks(batch_x)
                model_output = self._forward_model(
                    batch_x,
                    batch_m,
                    input_mask=input_mask,
                    target_next=batch_y
                    if self._is_next_step_prediction(self.args)
                    else None,
                )
                if (
                    self._is_next_step_prediction(self.args)
                    and not printed_prediction_shape
                ):
                    print(
                        "M2 shape check | input.shape={} | target.shape={} | "
                        "prediction.shape={}".format(
                            list(batch_x.shape),
                            list(batch_y.shape),
                            list(model_output["prediction"].shape),
                        )
                    )
                    printed_prediction_shape = True
                loss, loss_parts = self._model_loss_from_output(
                    model_output,
                    self._loss_targets(batch_x, batch_y),
                    self.args,
                    loss_mask=loss_mask,
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
                model_output = self._score_forward_model(batch_x, batch_m)
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
        prediction_mode = self._is_next_step_prediction(self.args)
        horizon = int(getattr(self.args, "target_horizon", 1))
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(windows), self.args.batch_size):
                starts = windows[offset: offset + self.args.batch_size]
                target_indices = np.asarray(starts, dtype=int) + self.args.seq_len + horizon - 1
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
                target_next = None
                if prediction_mode:
                    target_next = torch.from_numpy(
                        values[target_indices]
                    ).float().to(self.device)
                model_output = self._score_forward_model(
                    batch_x,
                    batch_m,
                    target_next=target_next,
                )
                if prediction_mode:
                    errors = model_output["prediction_score"]
                else:
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
        return_components=False,
    ):
        energies = []
        point_indices = []
        component_parts = {}
        prediction_mode = self._is_next_step_prediction(self.args)
        horizon = int(getattr(self.args, "target_horizon", 1))
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, len(windows), self.args.batch_size):
                starts = windows[offset: offset + self.args.batch_size]
                target_indices = np.asarray(starts, dtype=int) + self.args.seq_len + horizon - 1
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
                target_next = None
                if prediction_mode:
                    target_next = torch.from_numpy(
                        values[target_indices]
                    ).float().to(self.device)
                model_output = self._score_forward_model(
                    batch_x,
                    batch_m,
                    collect_diagnostics=bool(
                        getattr(self.args, "router_diagnostics", False)
                    ),
                    target_next=target_next,
                )
                if return_components:
                    components = self._dtsgad_causal_last_components(
                        model_output,
                        getattr(self.args, "obs_topk", 1),
                    )
                    for name, values_part in components.items():
                        component_parts.setdefault(name, []).append(values_part)
                    if bool(
                        getattr(self.args, "router_diagnostics", False)
                    ) and bool(getattr(self.args, "single_expert_diagnostics", True)):
                        single_mse = []
                        single_nll = []
                        model_for_experts = (
                            self.model.module
                            if isinstance(self.model, nn.DataParallel)
                            else self.model
                        )
                        num_experts = max(
                            block.num_experts
                            for block in getattr(model_for_experts, "blocks", [])
                        )
                        for expert_index in range(num_experts):
                            forced_output = self._score_forward_model(
                                batch_x,
                                batch_m,
                                force_expert_index=expert_index,
                                target_next=target_next,
                            )
                            forced_components = (
                                self._dtsgad_causal_last_components(
                                    forced_output,
                                    getattr(self.args, "obs_topk", 1),
                                )
                            )
                            if "s_mse_topk" in forced_components:
                                single_mse.append(forced_components["s_mse_topk"])
                            if "s_nll_topk" in forced_components:
                                single_nll.append(forced_components["s_nll_topk"])
                        if single_mse:
                            component_parts.setdefault(
                                "single_expert_mse_topk",
                                [],
                            ).append(np.stack(single_mse, axis=1))
                        if single_nll:
                            component_parts.setdefault(
                                "single_expert_nll_topk",
                                [],
                            ).append(np.stack(single_nll, axis=1))
                if (
                    isinstance(model_output, dict)
                    and "total_score" in model_output
                    and feature_scale is None
                ):
                    scores = model_output["total_score"][:, -1]
                elif prediction_mode:
                    errors = model_output["prediction_score"]
                    if feature_scale is not None:
                        scale = torch.as_tensor(
                            feature_scale, device=errors.device, dtype=errors.dtype
                        )
                        errors = errors / scale
                    topk = max(1, min(getattr(self.args, "obs_topk", 1), errors.size(-1)))
                    scores = torch.topk(errors, k=topk, dim=-1).values.mean(dim=-1)
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
                if prediction_mode:
                    point_indices.extend(target_indices.tolist())
                else:
                    point_indices.extend(
                        start + self.args.seq_len - 1 for start in starts
                    )

        if not energies:
            empty = np.array([], dtype=int)
            if return_components:
                return np.array([]), empty, empty, empty, {}
            return np.array([]), empty, empty, empty

        indices = np.asarray(point_indices, dtype=int)
        targets = np.asarray(labels)[indices]
        if meta is not None and "segment_id" in meta.columns:
            segment_ids = meta.iloc[indices]["segment_id"].to_numpy()
        else:
            segment_ids = np.zeros(len(indices), dtype=int)
        scores = np.concatenate(energies)
        if return_components:
            score_components = {
                name: np.concatenate(values_parts, axis=0)
                for name, values_parts in component_parts.items()
            }
            return scores, targets, indices, segment_ids, score_components
        return scores, targets, indices, segment_ids

    @staticmethod
    def _evaluate_predictions(gt, pred):
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(gt, pred, labels=[0, 1])
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

    @classmethod
    def _score_auc_by_anomaly_type(cls, gt, scores, meta, score_indices):
        anomaly_types = cls._score_anomaly_types(meta, score_indices)
        if anomaly_types is None:
            return []

        rows = []
        gt = np.asarray(gt).astype(int)
        gt[gt != 0] = 1
        scores = np.asarray(scores, dtype=np.float64)
        for name in sorted(set(anomaly_types), key=anomaly_type_sort_key):
            mask = anomaly_types == name
            if int(gt[mask].sum()) == 0:
                continue
            rows.append(
                {
                    "Anomaly Type": name,
                    "Points": int(mask.sum()),
                    "Anomalies": int(gt[mask].sum()),
                    "ROC-AUC": cls._safe_roc_auc(gt[mask], scores[mask]),
                    "PR-AUC": cls._safe_pr_auc(gt[mask], scores[mask]),
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

    @staticmethod
    def _safe_pr_auc(gt, scores):
        if len(np.unique(gt)) < 2:
            return float("nan")
        return float(average_precision_score(gt, scores))

    @staticmethod
    def _safe_roc_auc(gt, scores):
        if len(np.unique(gt)) < 2:
            return float("nan")
        return float(roc_auc_score(gt, scores))

    @staticmethod
    def _threshold_summary(threshold):
        values = np.asarray(threshold, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("inf")
        if values.size == 1:
            return float(values[0])
        return float(np.mean(finite))

    @staticmethod
    def _float_tag(value):
        text = "{:g}".format(float(value))
        return text

    @staticmethod
    def _sweep_values(args, plural_name, singular_name, default_values):
        values = list(getattr(args, plural_name, default_values) or default_values)
        singular = getattr(args, singular_name, None)
        if singular is not None and not any(np.isclose(singular, value) for value in values):
            values.append(float(singular))

        cleaned = []
        for value in values:
            value = float(value)
            if not any(np.isclose(value, existing) for existing in cleaned):
                cleaned.append(value)
        return cleaned

    def _build_threshold_diagnostics(
        self,
        test_energy,
        gt,
        score_segments=None,
        val_energy=None,
        test_meta=None,
        score_indices=None,
    ):
        test_energy = np.asarray(test_energy, dtype=np.float64).reshape(-1)
        gt = np.asarray(gt).astype(int).reshape(-1)
        gt[gt != 0] = 1
        if test_energy.shape[0] != gt.shape[0]:
            raise ValueError("test_energy and gt must have the same length")

        methods = []
        methods.append(
            (
                "atssd",
                *atssd(
                    test_energy,
                    window_size=self.args.winsize,
                    alpha=self.args.alpha,
                    segment_ids=score_segments,
                ),
                False,
            )
        )
        methods.append(
            (
                "causal_atssd",
                *causal_atssd(
                    test_energy,
                    window_size=self.args.winsize,
                    alpha=self.args.alpha,
                    segment_ids=score_segments,
                    confirmation=self.args.alarm_confirmation,
                    latch_alarm=bool(self.args.latch_alarm),
                    adaptation_clip=self.args.threshold_adaptation_clip,
                ),
                False,
            )
        )
        oracle_pred, oracle_threshold, _ = oracle_best_f1(test_energy, gt)
        methods.append(("oracle", oracle_pred, oracle_threshold, True))
        if val_energy is not None and len(val_energy) > 0:
            for quantile in self._sweep_values(
                self.args,
                "val_quantiles",
                "val_quantile",
                [0.95, 0.975, 0.99, 0.995, 0.999],
            ):
                methods.append(
                    (
                        "val_quantile_" + self._float_tag(quantile),
                        *val_quantile_threshold(
                            test_energy,
                            val_energy,
                            quantile=quantile,
                        ),
                        False,
                    )
                )
        methods.append(
            (
                "lagged_atssd",
                *lagged_atssd(
                    test_energy,
                    window_size=self.args.winsize,
                    alpha=self.args.alpha,
                    segment_ids=score_segments,
                ),
                False,
            )
        )
        methods.append(
            (
                "freeze_atssd",
                *freeze_atssd(
                    test_energy,
                    window_size=self.args.winsize,
                    alpha=self.args.alpha,
                    segment_ids=score_segments,
                ),
                False,
            )
        )
        for mad_k in self._sweep_values(
            self.args,
            "mad_ks",
            "mad_k",
            [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        ):
            methods.append(
                (
                    "median_mad_k" + self._float_tag(mad_k),
                    *median_mad_threshold(
                        test_energy,
                        window_size=self.args.winsize,
                        mad_k=mad_k,
                        segment_ids=score_segments,
                    ),
                    False,
                )
            )

        pr_auc = self._safe_pr_auc(gt, test_energy)
        rows = []
        type_rows = []
        predictions = {}
        thresholds = {}
        seen = set()
        for method_name, pred, threshold, uses_test_labels in methods:
            if method_name in seen:
                continue
            seen.add(method_name)
            pred = np.asarray(pred).astype(int).reshape(-1)
            predictions[method_name] = pred
            thresholds[method_name] = np.asarray(threshold)
            accuracy, precision, recall, f_score, cm = self._evaluate_predictions(
                gt, pred
            )
            event_hits, event_total = self._event_hits(gt, pred, score_segments)
            rows.append(
                {
                    "Threshold Method": method_name,
                    "Uses Test Labels": bool(uses_test_labels),
                    "Accuracy": float(accuracy),
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "Raw F1": float(f_score),
                    "F-score": float(f_score),
                    "PR-AUC": pr_auc,
                    "Event Hits": int(event_hits),
                    "Event Total": int(event_total),
                    "Predicted": int(pred.sum()),
                    "Threshold Mean": self._threshold_summary(threshold),
                    "Confusion Matrix": cm.tolist(),
                }
            )
            for row in self._metrics_by_anomaly_type(
                gt,
                pred,
                test_meta,
                score_indices,
            ):
                row = {"Threshold Method": method_name, **row}
                type_rows.append(row)

        return rows, type_rows, predictions, thresholds

    @staticmethod
    def _format_threshold_diagnostics(rows):
        if not rows:
            return ["Threshold diagnostics: unavailable"]
        lines = ["Threshold diagnostics"]
        lines.append(
            "  {method:<22} {acc:>8} {pre:>9} {rec:>8} {f1:>8} "
            "{prauc:>8} {hits:>9} {pred:>9} {oracle:>8}".format(
                method="Method",
                acc="Accuracy",
                pre="Precision",
                rec="Recall",
                f1="Raw F1",
                prauc="PR-AUC",
                hits="Hits",
                pred="Pred",
                oracle="Oracle",
            )
        )
        for row in rows:
            lines.append(
                "  {method:<22} {acc:8.4f} {pre:9.4f} {rec:8.4f} {f1:8.4f} "
                "{prauc:8.4f} {hits:>4}/{total:<4} {pred:9d} {oracle:>8}".format(
                    method=row["Threshold Method"][:22],
                    acc=row["Accuracy"],
                    pre=row["Precision"],
                    rec=row["Recall"],
                    f1=row["Raw F1"],
                    prauc=row["PR-AUC"],
                    hits=row["Event Hits"],
                    total=row["Event Total"],
                    pred=row["Predicted"],
                    oracle="yes" if row["Uses Test Labels"] else "no",
                )
            )
        return lines

    def _save_score_components(self, res_path, score_components):
        if not score_components:
            return
        component_path = os.path.join(res_path, "score_components")
        os.makedirs(component_path, exist_ok=True)
        np.savez(
            os.path.join(component_path, "dtsgad_score_components.npz"),
            **score_components,
        )
        for name, values in score_components.items():
            np.save(os.path.join(component_path, name + ".npy"), values)

        if any(
            name in score_components
            for name in (
                "router_gates",
                "expert_pair_cosine",
                "expert_pre_graph_pair_cosine",
                "expert_post_graph_pair_cosine",
                "graph_trace_pair_cosine",
                "adjacency",
                "single_expert_mse_topk",
            )
        ):
            metadata = {
                "patch_size_list": getattr(self.args, "patch_size_list", []),
                "num_experts_list": getattr(self.args, "num_experts_list", []),
                "top_k": getattr(self.args, "top_k", None),
                "knn_k": getattr(self.args, "knn_k", None),
                "graph_mode": getattr(self.args, "graph_mode", "learned"),
                "graph_residual_mode": getattr(
                    self.args,
                    "graph_residual_mode",
                    "shared",
                ),
                "graph_residual_alpha": getattr(
                    self.args,
                    "graph_residual_alpha",
                    1.0,
                ),
                "dtsgad_objective": getattr(
                    self.args,
                    "dtsgad_objective",
                    "reconstruction",
                ),
                "target_horizon": getattr(self.args, "target_horizon", 1),
                "prediction_score": "topk_mse"
                if self._is_next_step_prediction(self.args)
                else None,
                "mask_ratio": getattr(self.args, "mask_ratio", 0.0),
                "mask_eval_mode": getattr(self.args, "mask_eval_mode", "none"),
                "graph_trace_positions": [
                    "input",
                    "after_adjacency",
                    "residual_base",
                    "after_residual",
                    "after_norm",
                ],
            }
            with open(
                os.path.join(component_path, "router_metadata.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)

        feature_scores = score_components.get("s_obs_feature")
        if feature_scores is None:
            return
        feature_names = [
            "feature_{}".format(index)
            for index in range(feature_scores.shape[1])
        ]
        metadata_path = os.path.join(self.args.root_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            names = metadata.get("features", [])
            if len(names) == feature_scores.shape[1]:
                feature_names = names
        pd.DataFrame(feature_scores, columns=feature_names).to_csv(
            os.path.join(component_path, "s_obs_feature.csv"),
            index=False,
        )

    def _validation_energy(self, train_data, feature_scale=None):
        if self.args.score_mode == "causal_last":
            val_energy, _, _, _ = self._causal_last_scores(
                train_data.val,
                train_data.val_mark,
                np.zeros(len(train_data.val), dtype=int),
                train_data.val_windows,
                train_data.val_meta,
                feature_scale,
            )
            return val_energy
        if self.args.score_mode == "overlap_mean":
            val_energy, _ = self._point_scores(
                train_data.val,
                train_data.val_mark,
                train_data.val_windows,
            )
            return val_energy
        return None

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
        diagnostic_feature_scale = None
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
            diagnostic_feature_scale = feature_scale
        elif self.args.score_mode == "causal_last" and bool(
            getattr(self.args, "save_train_feature_scale", False)
        ):
            diagnostic_feature_scale = self._causal_last_feature_scale(
                train_data.train,
                train_data.train_mark,
                train_data.train_windows,
            )
        if self.args.score_mode == "causal_last":
            (
                test_energy,
                test_labels,
                score_indices,
                score_segments,
                score_components,
            ) = (
                self._causal_last_scores(
                    test_data.test,
                    test_data.test_mark,
                    test_data.test_labels,
                    test_data.test_windows,
                    test_data.test_meta,
                    feature_scale,
                    return_components=True,
                )
            )
        elif self.args.score_mode == "paper_nonoverlap":
            score_components = {}
            test_energy, test_labels, score_indices = self._paper_nonoverlap_scores(
                test_data.test,
                test_data.test_mark,
                test_data.test_labels,
            )
        else:
            score_components = {}
            test_energy, covered = self._point_scores(
                test_data.test, test_data.test_mark, test_data.test_windows
            )
            test_labels = test_data.test_labels[covered]
            score_indices = np.flatnonzero(covered)

        gt = test_labels.astype(int)
        gt[gt != 0] = 1
        val_energy = None
        if self.args.threshold_method == "val_quantile" or getattr(
            self.args,
            "threshold_diagnostics",
            True,
        ):
            val_energy = self._validation_energy(train_data, feature_scale)

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
        elif self.args.threshold_method == "causal_atssd":
            raw_pred, threshold = causal_atssd(
                test_energy,
                window_size=self.args.winsize,
                alpha=self.args.alpha,
                segment_ids=score_segments,
                confirmation=self.args.alarm_confirmation,
                latch_alarm=bool(self.args.latch_alarm),
                adaptation_clip=self.args.threshold_adaptation_clip,
            )
        elif self.args.threshold_method == "oracle":
            raw_pred, threshold, _ = oracle_best_f1(test_energy, gt)
        elif self.args.threshold_method == "val_quantile":
            if val_energy is None or len(val_energy) == 0:
                raise ValueError("val_quantile threshold requires validation scores")
            raw_pred, threshold = val_quantile_threshold(
                test_energy,
                val_energy,
                quantile=self.args.val_quantile,
            )
        elif self.args.threshold_method == "lagged_atssd":
            raw_pred, threshold = lagged_atssd(
                test_energy,
                window_size=self.args.winsize,
                alpha=self.args.alpha,
                segment_ids=score_segments,
            )
        elif self.args.threshold_method == "freeze_atssd":
            raw_pred, threshold = freeze_atssd(
                test_energy,
                window_size=self.args.winsize,
                alpha=self.args.alpha,
                segment_ids=score_segments,
            )
        else:
            raw_pred, threshold = median_mad_threshold(
                test_energy,
                window_size=self.args.winsize,
                mad_k=self.args.mad_k,
                segment_ids=score_segments,
            )

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
        if diagnostic_feature_scale is not None:
            np.save(
                os.path.join(res_path, "train_feature_error_scale.npy"),
                diagnostic_feature_scale,
            )
        self._save_score_components(res_path, score_components)

        visual(gt, test_energy, os.path.join(folder_path, "scores.pdf"))

        raw = self._evaluate_predictions(gt, raw_pred)
        adjusted = self._evaluate_predictions(adjusted_gt, adjusted_pred)
        roc_auc = self._safe_roc_auc(gt, test_energy)
        pr_auc = self._safe_pr_auc(gt, test_energy)
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
        latent_rows = []
        if getattr(self.args, "latent_diagnostics", False):
            latent_rows = self._save_latent_diagnostics(
                res_path,
                train_data,
                test_data,
                score_indices,
            )
        diagnostic_rows = []
        diagnostic_type_rows = []
        if getattr(self.args, "threshold_diagnostics", True):
            (
                diagnostic_rows,
                diagnostic_type_rows,
                diagnostic_predictions,
                diagnostic_thresholds,
            ) = self._build_threshold_diagnostics(
                test_energy,
                gt,
                score_segments,
                val_energy,
                test_data.test_meta,
                score_indices,
            )
            pd.DataFrame(diagnostic_rows).to_csv(
                os.path.join(res_path, "threshold_diagnostics.csv"),
                index=False,
            )
            if diagnostic_type_rows:
                pd.DataFrame(diagnostic_type_rows).to_csv(
                    os.path.join(
                        res_path,
                        "threshold_metrics_by_anomaly_type.csv",
                    ),
                    index=False,
                )
            threshold_dir = os.path.join(res_path, "threshold_diagnostics")
            os.makedirs(threshold_dir, exist_ok=True)
            for method_name, values in diagnostic_predictions.items():
                np.save(
                    os.path.join(threshold_dir, method_name + "_pred.npy"),
                    values,
                )
            for method_name, values in diagnostic_thresholds.items():
                np.save(
                    os.path.join(threshold_dir, method_name + "_threshold.npy"),
                    values,
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
        if latent_rows:
            latent_row = latent_rows[0]
            print(
                "Latent normal-center ROC-AUC : {:0.4f}, PR-AUC : {:0.4f}".format(
                    latent_row["ROC-AUC"],
                    latent_row["PR-AUC"],
                )
            )
        for line in self._format_threshold_diagnostics(diagnostic_rows):
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
            if latent_rows:
                latent_row = latent_rows[0]
                f.write(
                    "Latent normal-center ROC-AUC : {:0.4f}, PR-AUC : {:0.4f}\n".format(
                        latent_row["ROC-AUC"],
                        latent_row["PR-AUC"],
                    )
                )
            for line in self._format_threshold_diagnostics(diagnostic_rows):
                f.write(line + "\n")
            f.write("\n")
