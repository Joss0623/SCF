"""Main training entry point for Source-Compositional Forecasting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
UTILITY_ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import Dataset_Custom
from models.Autoformer import Model as Autoformer
from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
from models.TimesNet import Model as TimesNet
from models.TimeXer import Model as TimeXer
from models.iTransformer import Model as iTransformer


BASE_MODES = {"off", "fixed"}
CONTINUATION_MODES = {
    "off_continue",
    "fixed_continue",
    "fixed_high_continue",
    "global_learned",
    "horizon_only",
    "sample_only",
    "gate_no_utility",
    "source_factorized",
    "source_router_trend",
    "prior_router_horizon_only",
    "prior_router_sample_only",
    "prior_router_full_mlp",
    "prior_source_router_no_context",
    "prior_source_router_trend",
    "prior_interaction_router",
    "prior_source_attention_router",
    "cross_attention",
    "utility",
    "utility_shuffle",
}

FOUR_EXPERT_ROUTER_MODES = {
    "prior_router_horizon_only",
    "prior_router_sample_only",
    "prior_router_full_mlp",
    "prior_source_router_no_context",
    "prior_source_router_trend",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FrozenBertTokenEncoder:
    """Encode raw text with frozen BERT input embeddings and cache token IDs."""

    def __init__(self, model_name: str, device: torch.device, max_length: int = 1024):
        self.device = device
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        bert = BertModel.from_pretrained(model_name, local_files_only=True)
        self.embedding = deepcopy(bert.get_input_embeddings()).to(device).eval()
        self.embedding.requires_grad_(False)
        self.dim = self.embedding.embedding_dim
        self.cache: Dict[str, torch.Tensor] = {}
        del bert

    @torch.no_grad()
    def encode(self, dataset, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_texts = dataset.get_text(indices)
        texts = [self._normalize_text(value) for value in raw_texts]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            encoded = self.tokenizer(
                missing, padding=False, truncation=True, max_length=self.max_length
            ).input_ids
            for text, token_ids in zip(missing, encoded):
                self.cache[text] = torch.tensor(token_ids, dtype=torch.long)
        token_ids = [self.cache[text] for text in texts]
        max_tokens = max(ids.numel() for ids in token_ids)
        padded = torch.full(
            (len(token_ids), max_tokens), self.tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros((len(token_ids), max_tokens), dtype=torch.float32)
        for row, ids in enumerate(token_ids):
            padded[row, : ids.numel()] = ids
            mask[row, : ids.numel()] = 1.0
        padded = padded.to(self.device)
        mask = mask.to(self.device)
        return self.embedding(padded), mask

    @staticmethod
    def _normalize_text(value: object) -> str:
        return " ".join(str(value).replace("\\n", " ").split()).strip()


class PrecomputedTextEncoder:
    """Load row-aligned frozen text embeddings with an explicit source mask."""

    def __init__(self, embedding_path: str, mask_path: str, device: torch.device, args):
        self.embedding = np.load(embedding_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        if self.embedding.shape[:2] != self.mask.shape:
            raise ValueError(
                f"Embedding shape {self.embedding.shape} is incompatible with mask {self.mask.shape}"
            )
        self.dim = int(self.embedding.shape[-1])
        self.device = device
        self.seq_len = args.seq_len
        self.train_ratio = float(args.train_ratio)
        self.test_ratio = float(args.test_ratio)

    def encode(self, dataset, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total = len(self.embedding)
        train_end = int(total * self.train_ratio)
        test_count = int(total * self.test_ratio)
        offsets = [0, train_end - self.seq_len, total - test_count - self.seq_len]
        local_starts = indices.numpy() % dataset.tot_len
        # Use the final observed timestamp, never the first target timestamp.
        rows = offsets[dataset.set_type] + local_starts + self.seq_len - 1
        tokens = torch.from_numpy(np.asarray(self.embedding[rows], dtype=np.float32).copy())
        mask = torch.from_numpy(np.asarray(self.mask[rows], dtype=np.float32).copy())
        return tokens.to(self.device), mask.to(self.device)

    def encode_absolute_rows(self, rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64) % len(self.embedding)
        tokens = torch.from_numpy(np.asarray(self.embedding[rows], dtype=np.float32).copy())
        mask = torch.from_numpy(np.asarray(self.mask[rows], dtype=np.float32).copy())
        return tokens.to(self.device), mask.to(self.device)


from scf.model import NativeUtilityFusion


def make_dataset(args, flag: str) -> Dataset_Custom:
    dataset_args = SimpleNamespace(
        use_closedllm=0,
        text_len=args.text_len,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
    )
    return Dataset_Custom(
        args=dataset_args,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features="S",
        data_path=args.data_path,
        target="OT",
        timeenc=1,
        freq=args.freq,
    )


def make_loader(dataset, args, shuffle: bool, order_seed: int, drop_last: bool = True):
    generator = torch.Generator()
    generator.manual_seed(order_seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        generator=generator,
    )


def prepare_batch(
    batch,
    dataset,
    text_pooler,
    args,
    device,
    text_permutation: np.ndarray | None = None,
    zero_text: bool = False,
    text_control: str | None = None,
    empirical_token_rng: np.random.Generator | None = None,
    text_noise_scale: float = 0.0,
    foreign_text_pooler=None,
    foreign_row_map: np.ndarray | None = None,
):
    batch_x, batch_y, x_mark, y_mark, indices = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    x_mark = x_mark.float().to(device)
    y_mark = y_mark.float().to(device)
    prior = torch.from_numpy(dataset.get_prior_y(indices)).float().to(device)
    if foreign_text_pooler is not None:
        if foreign_row_map is None:
            raise ValueError("foreign_text_pooler requires foreign_row_map")
        text_tokens, text_mask = foreign_text_pooler.encode_absolute_rows(
            foreign_row_map[indices.numpy()]
        )
    else:
        text_indices = indices
        if text_permutation is not None:
            text_indices = torch.from_numpy(text_permutation[indices.numpy()])
        text_tokens, text_mask = text_pooler.encode(dataset, text_indices)
    if empirical_token_rng is not None:
        split = min(args.source_tokens, text_tokens.size(1))
        for start, end in ((0, split), (split, text_tokens.size(1))):
            block = text_tokens[:, start:end]
            valid = text_mask[:, start:end].bool()
            values = block[valid]
            if values.size(0) > 1:
                permutation = torch.from_numpy(
                    empirical_token_rng.permutation(values.size(0))
                ).to(values.device)
                block[valid] = values[permutation]
                text_tokens[:, start:end] = block
    if text_noise_scale > 0:
        valid_values = text_tokens[text_mask.bool()]
        if valid_values.numel() > 0:
            feature_scale = valid_values.std(0, unbiased=False).clamp_min(1e-6)
            noise = torch.from_numpy(
                empirical_token_rng.standard_normal(text_tokens.shape).astype(np.float32)
                if empirical_token_rng is not None
                else np.random.default_rng(0).standard_normal(text_tokens.shape).astype(np.float32)
            ).to(text_tokens.device)
            text_tokens = text_tokens + text_noise_scale * noise * feature_scale
            text_tokens = text_tokens * text_mask[:, :, None]
    if zero_text:
        text_tokens = torch.zeros_like(text_tokens)
    if text_control is not None:
        split = min(args.source_tokens, text_tokens.size(1))
        if text_control == "report_only":
            text_tokens[:, split:] = 0.0
            text_mask[:, split:] = 0.0
        elif text_control == "search_only":
            text_tokens[:, :split] = 0.0
            text_mask[:, :split] = 0.0
        elif text_control == "no_text":
            text_tokens.zero_()
            text_mask.zero_()
        elif text_control == "source_swap":
            if text_tokens.size(1) != 2 * split:
                raise ValueError(
                    "source_swap requires equal report/search token counts; "
                    f"received {text_tokens.size(1)} tokens with split {split}"
                )
            text_tokens = torch.cat([text_tokens[:, split:], text_tokens[:, :split]], dim=1)
            text_mask = torch.cat([text_mask[:, split:], text_mask[:, :split]], dim=1)
        else:
            raise ValueError(f"Unknown text control: {text_control}")
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :])
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
    true = batch_y[:, -args.pred_len :, :]
    return batch_x, true, x_mark, y_mark, dec_inp, text_tokens, text_mask, prior, indices


def build_optimizer(model, args, mode: str):
    if mode == "off":
        return Adam(model.backbone.parameters(), lr=args.learning_rate)
    if mode == "fixed":
        return Adam(
            [
                {"params": model.backbone.parameters(), "lr": args.learning_rate},
                {"params": model.text_branch.parameters(), "lr": args.text_learning_rate},
            ]
        )
    if mode == "off_continue":
        return Adam(
            model.backbone.parameters(),
            lr=args.learning_rate * args.continue_lr_ratio,
        )
    if mode in {"fixed_continue", "fixed_high_continue"}:
        return Adam(
            [
                {
                    "params": model.backbone.parameters(),
                    "lr": args.learning_rate * args.continue_lr_ratio,
                },
                {
                    "params": model.text_branch.parameters(),
                    "lr": args.text_learning_rate * args.continue_lr_ratio,
                },
            ]
        )
    if mode == "cross_attention":
        adapter_parameters = (
            list(model.cross_text_projection.parameters())
            + list(model.decomposition_projection.parameters())
            + list(model.patch_cross_attention.parameters())
            + list(model.patch_injection_gate.parameters())
        )
        return Adam(
            [
                {"params": adapter_parameters, "lr": args.gate_learning_rate},
                {
                    "params": model.backbone.parameters(),
                    "lr": args.learning_rate,
                },
                {
                    "params": model.text_branch.parameters(),
                    "lr": args.text_learning_rate,
                },
            ]
        )
    if mode == "global_learned":
        gate_parameters = [model.global_gate_logit]
    elif mode == "horizon_only":
        gate_parameters = [model.horizon_gate_logits]
    elif mode == "sample_only":
        gate_parameters = list(model.text_context.parameters()) + list(model.sample_gate_net.parameters())
    elif mode == "source_factorized":
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.source_reliability_net.parameters())
            + [model.source_horizon_factors, model.source_horizon_logits]
        )
    elif mode == "source_router_trend":
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.router_reliability_net.parameters())
            + [model.router_horizon_factors, model.router_horizon_logits]
        )
    elif mode == "prior_router_horizon_only":
        gate_parameters = [model.prior_router_horizon_logits]
    elif mode == "prior_router_sample_only":
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.prior_router_sample_net.parameters())
            + [model.prior_router_global_logits]
        )
    elif mode == "prior_router_full_mlp":
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.prior_router_full_net.parameters())
            + [model.prior_router_horizon_logits]
        )
    elif mode in {"prior_source_router_trend", "prior_source_router_no_context"}:
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.prior_router_reliability_net.parameters())
            + [model.prior_router_horizon_factors, model.prior_router_horizon_logits]
        )
    elif mode == "prior_interaction_router":
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.interaction_router_reliability_net.parameters())
            + list(model.synergy_delta_net.parameters())
            + [
                model.interaction_router_horizon_factors,
                model.interaction_router_horizon_logits,
            ]
        )
    elif mode == "prior_source_attention_router":
        gate_parameters = (
            list(model.report_context.parameters())
            + list(model.search_context.parameters())
            + list(model.prior_router_reliability_net.parameters())
            + list(model.attention_text_projection.parameters())
            + list(model.attention_query_projection.parameters())
            + list(model.report_source_attention.parameters())
            + list(model.search_source_attention.parameters())
            + list(model.report_attention_head.parameters())
            + list(model.search_attention_head.parameters())
            + [
                model.prior_router_horizon_factors,
                model.prior_router_horizon_logits,
                model.attention_horizon_queries,
                model.attention_source_positions,
            ]
        )
    else:
        gate_parameters = list(model.text_context.parameters()) + list(model.gate_net.parameters())
    return Adam(
        [
            {"params": gate_parameters, "lr": args.gate_learning_rate},
            {
                "params": model.backbone.parameters(),
                "lr": args.learning_rate * args.continue_lr_ratio,
            },
            {
                "params": model.text_branch.parameters(),
                "lr": args.text_learning_rate * args.continue_lr_ratio,
            },
        ]
    )


def seasonal_hard_negative_permutation(length: int, period: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = np.arange(length, dtype=np.int64)
    for phase in range(min(period, length)):
        group = np.arange(phase, length, period, dtype=np.int64)
        if len(group) > 1:
            offsets = rng.integers(1, len(group), size=len(group))
            permutation[group] = group[(np.arange(len(group)) + offsets) % len(group)]
    return permutation


def semantic_source_ranking_loss(
    model,
    true: torch.Tensor,
    negative_tokens: torch.Tensor,
    negative_mask: torch.Tensor,
    prior: torch.Tensor,
    args,
) -> torch.Tensor:
    aligned_predictions = model.last_source_predictions
    aligned_availability = model.last_source_availability
    endo = model.last_numerical_prediction
    negative_report, negative_search, negative_report_ok, negative_search_ok = model._source_experts(
        negative_tokens, negative_mask, prior
    )
    losses = []
    for aligned, negative, aligned_ok, negative_ok in zip(
        aligned_predictions,
        (negative_report, negative_search),
        aligned_availability,
        (negative_report_ok, negative_search_ok),
    ):
        aligned_se = (aligned - true).square()
        negative_se = (negative - true).square()
        endo_se = (endo - true).square()
        helpfulness = torch.sigmoid(
            (endo_se - aligned_se) / max(args.semantic_help_temp, 1e-6)
        ).detach()
        helpfulness = args.semantic_help_floor + (1.0 - args.semantic_help_floor) * helpfulness
        valid = (aligned_ok & negative_ok).to(true.dtype)[:, None, None]
        rank = F.relu(args.semantic_rank_margin + aligned_se - negative_se)
        losses.append((rank * helpfulness * valid).sum() / valid.expand_as(rank).sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def apply_text_control_tensors(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    args,
    text_control: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = text_tokens.clone()
    mask = text_mask.clone()
    split = min(args.source_tokens, tokens.size(1))
    if text_control == "report_only":
        tokens[:, split:] = 0.0
        mask[:, split:] = 0.0
    elif text_control == "search_only":
        tokens[:, :split] = 0.0
        mask[:, :split] = 0.0
    elif text_control == "no_text":
        tokens.zero_()
        mask.zero_()
    else:
        raise ValueError(f"Unknown text control: {text_control}")
    return tokens, mask


def interaction_router_loss(
    model,
    x: torch.Tensor,
    x_mark: torch.Tensor,
    dec_inp: torch.Tensor,
    y_mark: torch.Tensor,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    prior: torch.Tensor,
    full_prediction: torch.Tensor,
    args,
) -> torch.Tensor:
    expert_outputs = model.last_interaction_expert_outputs
    if expert_outputs is None:
        return full_prediction.new_zeros(())
    expert_outputs = expert_outputs
    predictions = {}
    for control in ("no_text", "report_only", "search_only"):
        control_tokens, control_mask = apply_text_control_tensors(
            text_tokens, text_mask, args, control
        )
        with torch.no_grad():
            control_prediction, _, _, _ = model(
                x, x_mark, dec_inp, y_mark, control_tokens, control_mask, prior, "prior_interaction_router"
            )
        predictions[control] = control_prediction.detach()

    numerical = expert_outputs[:, :, 0]
    report_unique = expert_outputs[:, :, 2]
    search_unique = expert_outputs[:, :, 3]
    synergy = expert_outputs[:, :, 4]
    full_anchor = full_prediction.detach()
    no_text = predictions["no_text"]
    report_only = predictions["report_only"]
    search_only = predictions["search_only"]
    margin = float(args.interaction_margin)

    report_positive = F.mse_loss(report_unique, report_only)
    report_negative = 0.5 * (
        F.relu(margin - (report_unique - no_text).square().mean())
        + F.relu(margin - (report_unique - search_only).square().mean())
    )
    search_positive = F.mse_loss(search_unique, search_only)
    search_negative = 0.5 * (
        F.relu(margin - (search_unique - no_text).square().mean())
        + F.relu(margin - (search_unique - report_only).square().mean())
    )
    synergy_positive = F.mse_loss(synergy, full_anchor)
    synergy_negative = (
        F.relu(margin - (synergy - no_text).square().mean())
        + F.relu(margin - (synergy - report_only).square().mean())
        + F.relu(margin - (synergy - search_only).square().mean())
    ) / 3.0
    numerical_anchor = F.mse_loss(numerical, no_text)
    return (
        report_positive
        + search_positive
        + synergy_positive
        + 0.25 * numerical_anchor
        + args.interaction_negative_weight
        * (report_negative + search_negative + synergy_negative)
    )


def run_epoch(
    model,
    dataset,
    text_pooler,
    args,
    device,
    mode: str,
    epoch_index: int,
    optimizer=None,
    utility_targets: np.ndarray | None = None,
):
    training = optimizer is not None
    model.train(training)
    loader = make_loader(
        dataset,
        args,
        shuffle=training,
        order_seed=args.order_seed + epoch_index,
        # Keep every validation window. Small Time-MMD domains can have fewer
        # than one full validation batch, and dropping it makes model selection
        # undefined and batch-size dependent.
        drop_last=training,
    )
    total_se = 0.0
    total_ae = 0.0
    total_count = 0
    total_rank = 0.0
    rank_batches = 0
    total_interaction = 0.0
    interaction_batches = 0
    negative_permutation = None
    if (
        training
        and mode in {"source_router_trend", "prior_source_router_trend"}
        and args.semantic_rank_weight > 0
    ):
        period = args.negative_period
        if period <= 0:
            period = 52 if args.freq.lower().startswith("w") else 12
        negative_permutation = seasonal_hard_negative_permutation(
            len(dataset), period, args.order_seed + epoch_index + 900_001
        )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x, true, x_mark, y_mark, dec_inp, text_tokens, text_mask, prior, indices = prepare_batch(
                batch, dataset, text_pooler, args, device
            )
            prediction, gate, raw_gate, _ = model(
                x, x_mark, dec_inp, y_mark, text_tokens, text_mask, prior, mode
            )
            forecast_loss = torch.mean((prediction - true) ** 2)
            loss = forecast_loss
            if negative_permutation is not None:
                negative_indices = torch.from_numpy(negative_permutation[indices.numpy()])
                negative_tokens, negative_mask = text_pooler.encode(dataset, negative_indices)
                rank_loss = semantic_source_ranking_loss(
                    model, true, negative_tokens, negative_mask, prior, args
                )
                loss = loss + args.semantic_rank_weight * rank_loss
                total_rank += rank_loss.detach().item()
                rank_batches += 1
            if training and mode == "prior_interaction_router" and args.interaction_loss_weight > 0:
                int_loss = interaction_router_loss(
                    model,
                    x,
                    x_mark,
                    dec_inp,
                    y_mark,
                    text_tokens,
                    text_mask,
                    prior,
                    prediction,
                    args,
                )
                loss = loss + args.interaction_loss_weight * int_loss
                total_interaction += int_loss.detach().item()
                interaction_batches += 1
            if training and utility_targets is not None:
                target = torch.from_numpy(utility_targets[indices.numpy()]).float().to(device).unsqueeze(-1)
                utility_loss = F.binary_cross_entropy(raw_gate.float(), target)
                loss = loss + args.utility_aux_weight * utility_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_se += torch.sum((prediction - true) ** 2).item()
            total_ae += torch.sum(torch.abs(prediction - true)).item()
            total_count += true.numel()
    if rank_batches:
        print(f"semantic_rank_loss={total_rank / rank_batches:.6f}")
    if interaction_batches:
        print(f"interaction_loss={total_interaction / interaction_batches:.6f}")
    return total_se / total_count, total_ae / total_count


@torch.no_grad()
def evaluate(
    model,
    dataset,
    text_pooler,
    args,
    device,
    mode: str,
    drop_last: bool = False,
    text_permutation: np.ndarray | None = None,
    zero_text: bool = False,
    text_control: str | None = None,
    empirical_token_shuffle: bool = False,
    text_noise_scale: float = 0.0,
    control_seed: int = 0,
    foreign_text_pooler=None,
    foreign_row_map: np.ndarray | None = None,
):
    model.eval()
    loader = make_loader(dataset, args, shuffle=False, order_seed=args.order_seed, drop_last=drop_last)
    records = []
    control_rng = np.random.default_rng(control_seed)
    for batch in loader:
        x, true, x_mark, y_mark, dec_inp, text_tokens, text_mask, prior, indices = prepare_batch(
            batch,
            dataset,
            text_pooler,
            args,
            device,
            text_permutation=text_permutation,
            zero_text=zero_text,
            text_control=text_control,
            empirical_token_rng=control_rng
            if empirical_token_shuffle or text_noise_scale > 0
            else None,
            text_noise_scale=text_noise_scale,
            foreign_text_pooler=foreign_text_pooler,
            foreign_row_map=foreign_row_map,
        )
        prediction, gate, raw_gate, text_pred = model(
            x, x_mark, dec_inp, y_mark, text_tokens, text_mask, prior, mode
        )
        records.append(
            {
                "indices": indices.numpy(),
                "prediction": prediction.cpu().numpy(),
                "true": true.cpu().numpy(),
                "gate": gate.cpu().numpy(),
                "raw_gate": raw_gate.cpu().numpy(),
                "text_pred": text_pred.cpu().numpy(),
            }
        )
        if model.last_source_weights is not None:
            records[-1]["source_weights"] = model.last_source_weights.cpu().numpy()
        if model.last_expert_outputs is not None:
            records[-1]["expert_outputs"] = model.last_expert_outputs.cpu().numpy()
            records[-1]["expert_availability"] = (
                model.last_expert_availability.cpu().numpy()
            )
    return {key: np.concatenate([record[key] for record in records]) for key in records[0]}


def metrics(evaluation) -> Tuple[float, float]:
    error = evaluation["prediction"] - evaluation["true"]
    return float(np.mean(error**2)), float(np.mean(np.abs(error)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def reliability_diagnostics(evaluation) -> dict | None:
    required = {"source_weights", "expert_outputs", "expert_availability"}
    if not required.issubset(evaluation):
        return None
    weights = evaluation["source_weights"].astype(np.float64)
    experts = evaluation["expert_outputs"].astype(np.float64)
    true = evaluation["true"].astype(np.float64)
    availability = evaluation["expert_availability"].astype(bool)
    squared_error = (experts - true) ** 2
    expanded_available = np.broadcast_to(availability[:, None, :], weights.shape)
    masked_error = np.where(expanded_available, squared_error, np.inf)
    oracle_expert = np.argmin(masked_error, axis=2)
    routed_expert = np.argmax(weights, axis=2)
    agreement = float(np.mean(routed_expert == oracle_expert))
    oracle_mse = float(np.mean(np.min(masked_error, axis=2)))
    fused_mse = float(np.mean((evaluation["prediction"] - true) ** 2))
    valid_weights = weights[expanded_available]
    valid_utility = -squared_error[expanded_available]
    weight_rank = _rankdata(valid_weights)
    utility_rank = _rankdata(valid_utility)
    if weight_rank.std() == 0 or utility_rank.std() == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(weight_rank, utility_rank)[0, 1])
    oracle_indicator = np.zeros_like(weights, dtype=np.float64)
    np.put_along_axis(oracle_indicator, oracle_expert[:, :, None], 1.0, axis=2)
    valid_oracle = oracle_indicator[expanded_available]
    bin_edges = np.linspace(0.0, 1.0, 6)
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (valid_weights >= left) & (
            valid_weights <= right if right == 1.0 else valid_weights < right
        )
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(
                float(np.mean(valid_weights[in_bin]))
                - float(np.mean(valid_oracle[in_bin]))
            )
    return {
        "router_oracle_agreement": agreement,
        "oracle_mse": oracle_mse,
        "fused_mse": fused_mse,
        "oracle_regret": fused_mse - oracle_mse,
        "weight_utility_spearman": spearman,
        "oracle_selection_ece": ece,
        "numerical_weight": float(np.mean(weights[:, :, 0])),
        "prior_weight": float(np.mean(weights[:, :, 1])),
        "report_weight": float(np.mean(weights[:, :, 2])),
        "search_weight": float(np.mean(weights[:, :, 3])),
        "text_weight": float(np.mean(weights[:, :, 2:])),
    }


def calibrate_gate_prior(model, val_dataset, text_pooler, args, device) -> Tuple[float, float]:
    endogenous = evaluate(
        model, val_dataset, text_pooler, args, device, "off", drop_last=False
    )
    multimodal = evaluate(
        model, val_dataset, text_pooler, args, device, "fixed", drop_last=False
    )
    best_gate, best_mse = 0.0, math.inf
    grid = np.arange(0.0, 1.0 + args.gate_prior_grid_step / 2, args.gate_prior_grid_step)
    for gate in grid:
        prediction = (1.0 - gate) * endogenous["prediction"] + gate * multimodal["text_pred"]
        mse = float(np.mean((prediction - endogenous["true"]) ** 2))
        if mse < best_mse:
            best_gate, best_mse = float(gate), mse
    return best_gate, best_mse


def train_model(
    args,
    seed: int,
    mode: str,
    initial_state,
    train_dataset,
    val_dataset,
    text_pooler,
    device,
    utility_targets=None,
    gate_prior: float | None = None,
):
    seed_everything(seed + 71_003)
    model = NativeUtilityFusion(args, text_pooler.dim).to(device)
    model.load_state_dict(initial_state)
    if gate_prior is not None:
        model.initialize_gate_prior(gate_prior)
    optimizer = build_optimizer(model, args, mode)
    epochs = args.base_epochs if mode in BASE_MODES or mode == "cross_attention" else args.continue_epochs
    best_val = math.inf
    best_state = None
    bad_epochs = 0
    for epoch_index in range(epochs):
        train_mse, _ = run_epoch(
            model,
            train_dataset,
            text_pooler,
            args,
            device,
            mode,
            epoch_index,
            optimizer,
            utility_targets,
        )
        val_mse, _ = run_epoch(
            model, val_dataset, text_pooler, args, device, mode, epoch_index, optimizer=None
        )
        print(
            f"seed={seed} mode={mode} epoch={epoch_index + 1:02d} "
            f"train_mse={train_mse:.6f} val_mse={val_mse:.6f}"
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_val


def make_utility_targets(args, off_eval, fixed_eval):
    off_se = (off_eval["prediction"] - off_eval["true"])[:, :, 0] ** 2
    fixed_se = (fixed_eval["prediction"] - fixed_eval["true"])[:, :, 0] ** 2
    utility = off_se - fixed_se
    target = 1.0 / (1.0 + np.exp(-np.clip(utility / args.utility_temp, -20.0, 20.0)))
    # Training evaluation is not shuffled, so indices map directly to dataset rows.
    full = np.full((args.train_windows, args.pred_len), 0.5, dtype=np.float32)
    full[off_eval["indices"]] = np.clip(target, 0.02, 0.98).astype(np.float32)
    print(
        f"utility: helpful={(utility > 0).mean():.2%} "
        f"target_mean={target.mean():.4f} mean_delta={utility.mean():.6f}"
    )
    return full


def save_samples(out_dir: Path, seed: int, evaluations, utility_target=None):
    reference = next(iter(evaluations.values()))
    rows = []
    for position, sample_index in enumerate(reference["indices"]):
        row = {"seed": seed, "sample_index": int(sample_index)}
        for mode, evaluation in evaluations.items():
            error = evaluation["prediction"][position] - evaluation["true"][position]
            row[f"{mode}_mse"] = float(np.mean(error**2))
            row[f"{mode}_mae"] = float(np.mean(np.abs(error)))
            row[f"{mode}_gate"] = float(np.mean(evaluation["gate"][position]))
            if "source_weights" in evaluation:
                weights = evaluation["source_weights"][position]
                if weights.shape[1] == 5:
                    labels = ("numerical", "prior", "report_unique", "search_unique", "synergy")
                elif weights.shape[1] == 4:
                    labels = ("numerical", "prior", "report", "search")
                else:
                    labels = ("numerical", "report", "search")
                for expert_index, label in enumerate(labels):
                    row[f"{mode}_{label}_weight"] = float(
                        np.mean(weights[:, expert_index])
                    )
            for horizon in range(error.shape[0]):
                row[f"{mode}_mse_h{horizon + 1:02d}"] = float(np.mean(error[horizon] ** 2))
                row[f"{mode}_gate_h{horizon + 1:02d}"] = float(
                    np.mean(evaluation["gate"][position, horizon])
                )
        if utility_target is not None and sample_index < len(utility_target):
            row["utility_target_mean"] = float(np.mean(utility_target[sample_index]))
        rows.append(row)
    with (out_dir / f"samples_seed{seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.text_encoder == "bert_tokens":
        text_pooler = FrozenBertTokenEncoder(args.bert_model, device)
    else:
        text_pooler = PrecomputedTextEncoder(
            args.text_embedding_path, args.text_mask_path, device, args
        )
    foreign_text_pooler = None
    if args.foreign_text_embedding_path or args.foreign_text_mask_path:
        if not (args.foreign_text_embedding_path and args.foreign_text_mask_path):
            raise ValueError("Both foreign text embedding and mask paths are required")
        if args.text_encoder != "precomputed":
            raise ValueError("Cross-domain text control requires precomputed embeddings")
        foreign_text_pooler = PrecomputedTextEncoder(
            args.foreign_text_embedding_path,
            args.foreign_text_mask_path,
            device,
            args,
        )
        if foreign_text_pooler.dim != text_pooler.dim:
            raise ValueError("Target and foreign text embedding dimensions must match")
    train_dataset = make_dataset(args, "train")
    val_dataset = make_dataset(args, "val")
    test_dataset = make_dataset(args, "test")
    args.train_windows = len(train_dataset)
    print(f"windows train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    requested_modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(requested_modes) - BASE_MODES - CONTINUATION_MODES
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    rows = []
    calibration_rows = []
    reliability_rows = []
    router_parameter_rows = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        seed_everything(seed)
        template = NativeUtilityFusion(args, text_pooler.dim)
        shared_initial_state = deepcopy(template.state_dict())
        trained = {}
        evaluations = {}

        for mode in ("off", "fixed"):
            model, best_val = train_model(
                args,
                seed,
                mode,
                shared_initial_state,
                train_dataset,
                val_dataset,
                text_pooler,
                device,
            )
            trained[mode] = model
            torch.save(
                {"state_dict": model.state_dict(), "seed": seed, "mode": mode, "config": vars(args)},
                out_dir / f"checkpoint_seed{seed}_{mode}.pt",
            )
            evaluation = evaluate(model, test_dataset, text_pooler, args, device, mode)
            evaluations[mode] = evaluation
            mse, mae = metrics(evaluation)
            rows.append({"seed": seed, "mode": mode, "mse": mse, "mae": mae, "best_val": best_val})
            print(f"TEST seed={seed} mode={mode} mse={mse:.6f} mae={mae:.6f}")

        off_train = evaluate(trained["off"], train_dataset, text_pooler, args, device, "off", drop_last=False)
        fixed_train = evaluate(trained["fixed"], train_dataset, text_pooler, args, device, "fixed", drop_last=False)
        utility_target = make_utility_targets(args, off_train, fixed_train)
        fixed_state = deepcopy(trained["fixed"].state_dict())
        adaptive_state = deepcopy(fixed_state)
        off_state = trained["off"].state_dict()
        for key in adaptive_state:
            if key.startswith("backbone."):
                adaptive_state[key] = off_state[key].detach().cpu().clone()
        calibrated_prior = None
        if args.calibrate_gate_prior:
            calibration_model = NativeUtilityFusion(args, text_pooler.dim).to(device)
            calibration_model.load_state_dict(adaptive_state)
            calibrated_prior, calibrated_val = calibrate_gate_prior(
                calibration_model, val_dataset, text_pooler, args, device
            )
            original_fixed_gate = calibration_model.fixed_gate
            calibration_model.fixed_gate = calibrated_prior
            calibrated_evaluation = evaluate(
                calibration_model, test_dataset, text_pooler, args, device, "fixed"
            )
            calibration_model.fixed_gate = original_fixed_gate
            evaluations["fixed_calibrated"] = calibrated_evaluation
            calibrated_mse, calibrated_mae = metrics(calibrated_evaluation)
            rows.append(
                {
                    "seed": seed,
                    "mode": "fixed_calibrated",
                    "mse": calibrated_mse,
                    "mae": calibrated_mae,
                    "best_val": calibrated_val,
                }
            )
            calibration_rows.append(
                {
                    "seed": seed,
                    "gate_prior": calibrated_prior,
                    "val_mse": calibrated_val,
                    "test_mse": calibrated_mse,
                    "test_mae": calibrated_mae,
                }
            )
            print(
                f"CALIBRATION seed={seed} gate_prior={calibrated_prior:.3f} "
                f"val_mse={calibrated_val:.6f} test_mse={calibrated_mse:.6f}"
            )

        four_expert_rng_state = None
        four_expert_cuda_rng_state = None
        for mode in requested_modes:
            if mode in BASE_MODES:
                continue
            if mode in FOUR_EXPERT_ROUTER_MODES:
                if four_expert_rng_state is None:
                    four_expert_rng_state = torch.get_rng_state()
                    if torch.cuda.is_available():
                        four_expert_cuda_rng_state = torch.cuda.get_rng_state_all()
                else:
                    torch.set_rng_state(four_expert_rng_state)
                    if four_expert_cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(four_expert_cuda_rng_state)
            mode_target = utility_target if mode in {"utility", "utility_shuffle"} else None
            if mode == "utility_shuffle":
                mode_target = utility_target.copy()
                np.random.default_rng(seed + 100_003).shuffle(mode_target, axis=0)
            if mode == "cross_attention":
                initial_state = shared_initial_state
            elif mode == "off_continue":
                initial_state = deepcopy(trained["off"].state_dict())
            elif mode in {
                "source_factorized",
                "source_router_trend",
                "prior_router_horizon_only",
                "prior_router_sample_only",
                "prior_router_full_mlp",
                "prior_source_router_no_context",
                "prior_source_router_trend",
                "prior_interaction_router",
                "prior_source_attention_router",
            } and args.calibrate_gate_prior:
                initial_state = adaptive_state
            else:
                initial_state = fixed_state
            model, best_val = train_model(
                args,
                seed,
                mode,
                initial_state,
                train_dataset,
                val_dataset,
                text_pooler,
                device,
                mode_target,
                calibrated_prior
                if mode in {
                    "source_factorized",
                    "source_router_trend",
                    "prior_router_horizon_only",
                    "prior_router_sample_only",
                    "prior_router_full_mlp",
                    "prior_source_router_no_context",
                    "prior_source_router_trend",
                    "prior_interaction_router",
                    "prior_source_attention_router",
                }
                else None,
            )
            evaluation = evaluate(model, test_dataset, text_pooler, args, device, mode)
            evaluations[mode] = evaluation
            torch.save(
                {"state_dict": model.state_dict(), "seed": seed, "mode": mode, "config": vars(args)},
                out_dir / f"checkpoint_seed{seed}_{mode}.pt",
            )
            mse, mae = metrics(evaluation)
            rows.append({"seed": seed, "mode": mode, "mse": mse, "mae": mae, "best_val": best_val})
            print(f"TEST seed={seed} mode={mode} mse={mse:.6f} mae={mae:.6f}")
            if mode in FOUR_EXPERT_ROUTER_MODES:
                optimizer_probe = build_optimizer(model, args, mode)
                router_parameter_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "router_parameters": sum(
                            parameter.numel()
                            for parameter in optimizer_probe.param_groups[0]["params"]
                        ),
                    }
                )

            if (not args.skip_test_controls) and mode in {
                "gate_no_utility",
                "source_factorized",
                "source_router_trend",
                "prior_router_horizon_only",
                "prior_router_sample_only",
                "prior_router_full_mlp",
                "prior_source_router_no_context",
                "prior_source_router_trend",
                "prior_interaction_router",
                "prior_source_attention_router",
                "cross_attention",
            }:
                semantic_controls = {}
                if not args.source_controls_only:
                    permutation = np.random.default_rng(seed + 200_003).permutation(len(test_dataset))
                    semantic_controls.update({
                        f"{mode}_text_shuffle_test": evaluate(
                            model,
                            test_dataset,
                            text_pooler,
                            args,
                            device,
                            mode,
                            text_permutation=permutation,
                        ),
                        f"{mode}_text_zero_test": evaluate(
                            model,
                            test_dataset,
                            text_pooler,
                            args,
                            device,
                            mode,
                            zero_text=True,
                        ),
                    })
                if mode == "prior_source_router_trend" and not args.source_controls_only:
                    period = args.negative_period
                    if period <= 0:
                        period = 52 if args.freq.lower().startswith("w") else 12
                    seasonal_permutation = seasonal_hard_negative_permutation(
                        len(test_dataset), period, seed + 300_003
                    )
                    semantic_controls.update(
                        {
                            f"{mode}_seasonal_shuffle_test": evaluate(
                                model,
                                test_dataset,
                                text_pooler,
                                args,
                                device,
                                mode,
                                text_permutation=seasonal_permutation,
                            ),
                            f"{mode}_empirical_token_test": evaluate(
                                model,
                                test_dataset,
                                text_pooler,
                                args,
                                device,
                                mode,
                                empirical_token_shuffle=True,
                                control_seed=seed + 400_003,
                            ),
                        }
                    )
                    for noise_scale in (0.5, 1.0, 2.0):
                        label = str(noise_scale).replace(".", "p")
                        semantic_controls[f"{mode}_noise_{label}_test"] = evaluate(
                            model,
                            test_dataset,
                            text_pooler,
                            args,
                            device,
                            mode,
                            text_noise_scale=noise_scale,
                            control_seed=seed + 500_003,
                        )
                    if foreign_text_pooler is not None:
                        foreign_count = len(foreign_text_pooler.embedding)
                        foreign_test_count = max(1, int(foreign_count * args.test_ratio))
                        foreign_start = foreign_count - foreign_test_count
                        foreign_row_map = np.random.default_rng(seed + 600_003).integers(
                            foreign_start,
                            foreign_count,
                            size=len(test_dataset),
                        )
                        semantic_controls[f"{mode}_cross_domain_text_test"] = evaluate(
                            model,
                            test_dataset,
                            text_pooler,
                            args,
                            device,
                            mode,
                            foreign_text_pooler=foreign_text_pooler,
                            foreign_row_map=foreign_row_map,
                        )
                if mode in {
                    "source_factorized",
                    "source_router_trend",
                    "prior_router_horizon_only",
                    "prior_router_sample_only",
                    "prior_router_full_mlp",
                    "prior_source_router_no_context",
                    "prior_source_router_trend",
                    "prior_interaction_router",
                    "prior_source_attention_router",
                }:
                    text_controls = ["report_only", "search_only"]
                    if not args.source_controls_only:
                        text_controls.append("source_swap")
                    if mode in FOUR_EXPERT_ROUTER_MODES | {
                        "prior_interaction_router",
                        "prior_source_attention_router",
                    }:
                        text_controls.append("no_text")
                    for text_control in text_controls:
                        semantic_controls[f"{mode}_{text_control}_test"] = evaluate(
                            model,
                            test_dataset,
                            text_pooler,
                            args,
                            device,
                            mode,
                            text_control=text_control,
                        )
                for control_name, control_eval in semantic_controls.items():
                    evaluations[control_name] = control_eval
                    control_mse, control_mae = metrics(control_eval)
                    rows.append(
                        {
                            "seed": seed,
                            "mode": control_name,
                            "mse": control_mse,
                            "mae": control_mae,
                            "best_val": best_val,
                        }
                    )
                    print(
                        f"TEST seed={seed} mode={control_name} "
                        f"mse={control_mse:.6f} mae={control_mae:.6f}"
                    )

        for evaluation_mode, evaluation in evaluations.items():
            diagnostic = reliability_diagnostics(evaluation)
            if diagnostic is not None:
                reliability_rows.append(
                    {
                        "seed": seed,
                        "mode": evaluation_mode,
                        "pred_len": args.pred_len,
                        **diagnostic,
                    }
                )
        save_samples(out_dir, seed, evaluations, utility_target)

    with (out_dir / "summary_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if calibration_rows:
        with (out_dir / "calibration_by_seed.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=calibration_rows[0].keys())
            writer.writeheader()
            writer.writerows(calibration_rows)
    if reliability_rows:
        with (out_dir / "reliability_diagnostics_by_seed.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=reliability_rows[0].keys())
            writer.writeheader()
            writer.writerows(reliability_rows)
    if router_parameter_rows:
        with (out_dir / "router_parameters_by_seed.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=router_parameter_rows[0].keys())
            writer.writeheader()
            writer.writerows(router_parameter_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./data/Public_Health")
    parser.add_argument("--data_path", default="US_FLURATIO_Week.csv")
    parser.add_argument("--freq", default="w")
    parser.add_argument(
        "--out_dir",
        default=str(UTILITY_ROOT / "outputs" / "HealthUS_h24"),
    )
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument(
        "--modes",
        default="off,fixed,off_continue,prior_source_router_trend",
    )
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--text_len", type=int, default=4)
    parser.add_argument("--bert_model", default="google-bert/bert-base-uncased")
    parser.add_argument("--text_encoder", choices=("bert_tokens", "precomputed"), default="precomputed")
    parser.add_argument(
        "--text_embedding_path",
        default=str(
            UTILITY_ROOT
            / "data"
            / "embeddings"
            / "HealthUS"
            / "text_token_embeddings.npy"
        ),
    )
    parser.add_argument(
        "--text_mask_path",
        default=str(
            UTILITY_ROOT
            / "data"
            / "embeddings"
            / "HealthUS"
            / "text_token_mask.npy"
        ),
    )
    parser.add_argument("--foreign_text_embedding_path", default="")
    parser.add_argument("--foreign_text_mask_path", default="")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_epochs", type=int, default=10)
    parser.add_argument("--continue_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--text_learning_rate", type=float, default=1e-2)
    parser.add_argument("--gate_learning_rate", type=float, default=1e-3)
    parser.add_argument("--continue_lr_ratio", type=float, default=0.1)
    parser.add_argument("--utility_temp", type=float, default=0.1)
    parser.add_argument("--utility_aux_weight", type=float, default=0.1)
    parser.add_argument("--fixed_gate", type=float, default=0.7)
    parser.add_argument("--high_fixed_gate", type=float, default=0.7)
    parser.add_argument("--min_gate", type=float, default=0.0)
    parser.add_argument("--max_gate", type=float, default=1.0)
    parser.add_argument("--calibrate_gate_prior", action="store_true")
    parser.add_argument("--gate_prior_grid_step", type=float, default=0.05)
    parser.add_argument("--gate_context_dim", type=int, default=64)
    parser.add_argument("--source_context_dim", type=int, default=32)
    parser.add_argument("--source_tokens", type=int, default=4)
    parser.add_argument("--factor_rank", type=int, default=8)
    parser.add_argument("--trend_kernel", type=int, default=5)
    parser.add_argument("--source_attention_dim", type=int, default=64)
    parser.add_argument("--source_attention_heads", type=int, default=4)
    parser.add_argument("--semantic_rank_weight", type=float, default=0.0)
    parser.add_argument("--semantic_rank_margin", type=float, default=0.02)
    parser.add_argument("--semantic_help_temp", type=float, default=0.1)
    parser.add_argument("--semantic_help_floor", type=float, default=0.1)
    parser.add_argument("--negative_period", type=int, default=0)
    parser.add_argument("--interaction_loss_weight", type=float, default=0.0)
    parser.add_argument("--interaction_margin", type=float, default=0.02)
    parser.add_argument("--interaction_negative_weight", type=float, default=0.2)
    parser.add_argument("--gate_hidden", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--backbone_model",
        choices=("PatchTST", "DLinear", "TimesNet", "iTransformer", "TimeXer", "Autoformer"),
        default="PatchTST",
    )
    parser.add_argument("--features", default="S")
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--skip_test_controls", action="store_true")
    parser.add_argument(
        "--source_controls_only",
        action="store_true",
        help="Evaluate only report-only, search-only, and no-text source-decomposition controls.",
    )
    parser.add_argument("--order_seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
