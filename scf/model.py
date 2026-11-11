"""SCF: source-specific forecasts and sample–horizon factorized routing."""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Autoformer import Model as Autoformer
from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
from models.TimesNet import Model as TimesNet
from models.TimeXer import Model as TimeXer
from models.iTransformer import Model as iTransformer

FOUR_EXPERT_ROUTER_MODES = {
    "prior_router_horizon_only",
    "prior_router_sample_only",
    "prior_router_full_mlp",
    "prior_source_router_no_context",
    "prior_source_router_trend",
}

class NativeUtilityFusion(nn.Module):
    def __init__(self, args, text_dim: int):
        super().__init__()
        config = SimpleNamespace(
            task_name="long_term_forecast",
            features=args.features,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=1,
            top_k=args.top_k,
            num_kernels=args.num_kernels,
            output_attention=False,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            activation="gelu",
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            enc_in=1,
            dec_in=1,
            c_out=1,
            embed=args.embed,
            freq=args.freq,
            moving_avg=args.moving_avg,
            patch_len=args.patch_len,
            use_norm=bool(args.use_norm),
        )
        if args.backbone_model == "PatchTST":
            self.backbone = PatchTST(config).float()
        elif args.backbone_model == "DLinear":
            self.backbone = DLinear(config).float()
        elif args.backbone_model == "TimesNet":
            self.backbone = TimesNet(config).float()
        elif args.backbone_model == "iTransformer":
            self.backbone = iTransformer(config).float()
        elif args.backbone_model == "TimeXer":
            self.backbone = TimeXer(config).float()
        elif args.backbone_model == "Autoformer":
            self.backbone = Autoformer(config).float()
        else:
            raise ValueError(f"Unsupported backbone_model: {args.backbone_model}")
        self.pred_len = args.pred_len
        self.fixed_gate = float(args.fixed_gate)
        self.high_fixed_gate = float(args.high_fixed_gate)
        self.min_gate = float(args.min_gate)
        self.max_gate = float(args.max_gate)
        hidden = max(text_dim // 8, 64)
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, args.pred_len),
        )
        self.text_context = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, args.gate_context_dim),
            nn.GELU(),
        )
        gate_input_dim = args.gate_context_dim + 5 + 3 * args.pred_len
        self.gate_net = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, args.pred_len),
        )
        self.sample_gate_net = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, 1),
        )
        self.source_tokens = int(args.source_tokens)
        self.report_context = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, args.source_context_dim),
            nn.GELU(),
        )
        self.search_context = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, args.source_context_dim),
            nn.GELU(),
        )
        source_input_dim = 2 * args.source_context_dim + 5 + 5 + 3 * args.pred_len
        self.source_reliability_net = nn.Sequential(
            nn.LayerNorm(source_input_dim),
            nn.Linear(source_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, args.factor_rank),
        )
        self.source_horizon_factors = nn.Parameter(
            torch.empty(args.pred_len, args.factor_rank)
        )
        self.source_horizon_logits = nn.Parameter(
            torch.empty(args.pred_len)
        )
        self.factor_rank = int(args.factor_rank)
        router_input_dim = 2 * args.source_context_dim + 5 + 5 + 5 * args.pred_len
        self.router_reliability_net = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, 2 * args.factor_rank),
        )
        self.router_horizon_factors = nn.Parameter(
            torch.empty(2, args.pred_len, args.factor_rank)
        )
        self.router_horizon_logits = nn.Parameter(torch.empty(args.pred_len, 3))
        prior_router_input_dim = 2 * args.source_context_dim + 5 + 5 + 7 * args.pred_len
        self.source_context_dim = int(args.source_context_dim)
        self.prior_router_reliability_net = nn.Sequential(
            nn.LayerNorm(prior_router_input_dim),
            nn.Linear(prior_router_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, 3 * args.factor_rank),
        )
        self.prior_router_horizon_factors = nn.Parameter(
            torch.empty(3, args.pred_len, args.factor_rank)
        )
        self.prior_router_horizon_logits = nn.Parameter(torch.empty(args.pred_len, 4))
        # Comparator-only modules must not perturb the RNG stream of the
        # published RIFT architecture.  Restoring the state keeps every
        # pre-existing module bitwise aligned for a fixed seed.
        comparator_rng_state = torch.get_rng_state()
        self.prior_router_global_logits = nn.Parameter(torch.empty(4))
        self.prior_router_sample_net = nn.Sequential(
            nn.LayerNorm(prior_router_input_dim),
            nn.Linear(prior_router_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, 3),
        )
        # Choose the generic gate width so its trainable router budget is as
        # close as possible to the low-rank router at this horizon.
        low_rank_budget = (
            args.gate_hidden * (prior_router_input_dim + 1 + 3 * args.factor_rank)
            + 3 * args.factor_rank
            + 3 * args.pred_len * args.factor_rank
            + 4 * args.pred_len
        )
        full_fixed_budget = 7 * args.pred_len
        full_hidden_denominator = prior_router_input_dim + 1 + 3 * args.pred_len
        self.prior_router_full_hidden = max(
            1,
            round((low_rank_budget - full_fixed_budget) / full_hidden_denominator),
        )
        self.prior_router_full_net = nn.Sequential(
            nn.LayerNorm(prior_router_input_dim),
            nn.Linear(prior_router_input_dim, self.prior_router_full_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.prior_router_full_hidden, 3 * args.pred_len),
        )
        torch.set_rng_state(comparator_rng_state)
        self.interaction_router_reliability_net = nn.Sequential(
            nn.LayerNorm(prior_router_input_dim),
            nn.Linear(prior_router_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, 4 * args.factor_rank),
        )
        self.interaction_router_horizon_factors = nn.Parameter(
            torch.empty(4, args.pred_len, args.factor_rank)
        )
        self.interaction_router_horizon_logits = nn.Parameter(torch.empty(args.pred_len, 5))
        self.synergy_delta_net = nn.Sequential(
            nn.LayerNorm(prior_router_input_dim),
            nn.Linear(prior_router_input_dim, args.gate_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gate_hidden, args.pred_len),
        )
        self.attention_text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, args.source_attention_dim, bias=False),
        )
        self.attention_query_projection = nn.Linear(2, args.source_attention_dim)
        self.attention_horizon_queries = nn.Parameter(
            torch.empty(args.pred_len, args.source_attention_dim)
        )
        self.attention_source_positions = nn.Parameter(
            torch.empty(args.source_tokens, args.source_attention_dim)
        )
        self.report_source_attention = nn.MultiheadAttention(
            args.source_attention_dim,
            args.source_attention_heads,
            dropout=args.dropout,
            batch_first=True,
        )
        self.search_source_attention = nn.MultiheadAttention(
            args.source_attention_dim,
            args.source_attention_heads,
            dropout=args.dropout,
            batch_first=True,
        )
        self.report_attention_head = nn.Sequential(
            nn.LayerNorm(args.source_attention_dim),
            nn.Linear(args.source_attention_dim, 1),
        )
        self.search_attention_head = nn.Sequential(
            nn.LayerNorm(args.source_attention_dim),
            nn.Linear(args.source_attention_dim, 1),
        )
        self.trend_kernel = int(args.trend_kernel)
        if self.trend_kernel < 1 or self.trend_kernel % 2 == 0:
            raise ValueError("--trend_kernel must be a positive odd integer")
        self.last_source_weights = None
        self.last_source_predictions = None
        self.last_source_availability = None
        self.last_numerical_prediction = None
        self.last_interaction_expert_outputs = None
        self.last_expert_outputs = None
        self.last_expert_availability = None
        self.cross_text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, args.d_model, bias=False)
        )
        self.decomposition_projection = nn.Sequential(
            nn.LayerNorm(5), nn.Linear(5, args.d_model), nn.GELU()
        )
        self.patch_cross_attention = nn.MultiheadAttention(
            args.d_model, args.n_heads, dropout=args.dropout, batch_first=True
        )
        self.patch_injection_gate = nn.Sequential(
            nn.LayerNorm(3 * args.d_model),
            nn.Linear(3 * args.d_model, args.d_model),
            nn.GELU(),
            nn.Linear(args.d_model, 1),
        )
        nn.init.zeros_(self.patch_injection_gate[-1].weight)
        nn.init.constant_(self.patch_injection_gate[-1].bias, -8.0)
        initial_raw_gate = (self.fixed_gate - self.min_gate) / max(self.max_gate - self.min_gate, 1e-6)
        initial_raw_gate = min(max(initial_raw_gate, 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, math.log(initial_raw_gate / (1.0 - initial_raw_gate)))
        initial_logit = math.log(initial_raw_gate / (1.0 - initial_raw_gate))
        self.global_gate_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.horizon_gate_logits = nn.Parameter(
            torch.full((args.pred_len,), initial_logit, dtype=torch.float32)
        )
        nn.init.zeros_(self.sample_gate_net[-1].weight)
        nn.init.constant_(self.sample_gate_net[-1].bias, initial_logit)
        nn.init.normal_(self.source_reliability_net[-1].weight, std=0.01)
        nn.init.zeros_(self.source_reliability_net[-1].bias)
        nn.init.normal_(self.source_horizon_factors, std=0.02)
        nn.init.constant_(self.source_horizon_logits, initial_logit)
        nn.init.normal_(self.router_reliability_net[-1].weight, std=0.01)
        nn.init.zeros_(self.router_reliability_net[-1].bias)
        nn.init.normal_(self.router_horizon_factors, std=0.02)
        self._set_router_prior(self.fixed_gate)
        nn.init.normal_(self.prior_router_reliability_net[-1].weight, std=0.01)
        nn.init.zeros_(self.prior_router_reliability_net[-1].bias)
        nn.init.normal_(self.prior_router_horizon_factors, std=0.02)
        self._set_prior_source_router_prior(self.fixed_gate)
        comparator_rng_state = torch.get_rng_state()
        nn.init.normal_(self.prior_router_sample_net[-1].weight, std=0.01)
        nn.init.zeros_(self.prior_router_sample_net[-1].bias)
        nn.init.normal_(self.prior_router_full_net[-1].weight, std=0.01)
        nn.init.zeros_(self.prior_router_full_net[-1].bias)
        torch.set_rng_state(comparator_rng_state)
        nn.init.normal_(self.interaction_router_reliability_net[-1].weight, std=0.01)
        nn.init.zeros_(self.interaction_router_reliability_net[-1].bias)
        nn.init.normal_(self.interaction_router_horizon_factors, std=0.02)
        self._set_interaction_router_prior(self.fixed_gate)
        nn.init.zeros_(self.synergy_delta_net[-1].weight)
        nn.init.zeros_(self.synergy_delta_net[-1].bias)
        nn.init.normal_(self.attention_horizon_queries, std=0.02)
        nn.init.normal_(self.attention_source_positions, std=0.02)
        nn.init.normal_(self.report_attention_head[-1].weight, std=0.01)
        nn.init.zeros_(self.report_attention_head[-1].bias)
        nn.init.normal_(self.search_attention_head[-1].weight, std=0.01)
        nn.init.zeros_(self.search_attention_head[-1].bias)

    def _set_router_prior(self, gate_prior: float) -> None:
        text_prior = min(max(float(gate_prior), 0.02), 0.98)
        probabilities = torch.tensor(
            [1.0 - text_prior, text_prior / 2.0, text_prior / 2.0],
            dtype=self.router_horizon_logits.dtype,
            device=self.router_horizon_logits.device,
        )
        with torch.no_grad():
            self.router_horizon_logits.copy_(
                probabilities.log()[None, :].expand_as(self.router_horizon_logits)
            )

    def _set_prior_source_router_prior(self, gate_prior: float) -> None:
        prior_weight = min(max(float(gate_prior), 0.02), 0.96)
        semantic_weight = 0.01
        numerical_weight = 1.0 - prior_weight - 2.0 * semantic_weight
        probabilities = torch.tensor(
            [numerical_weight, prior_weight, semantic_weight, semantic_weight],
            dtype=self.prior_router_horizon_logits.dtype,
            device=self.prior_router_horizon_logits.device,
        )
        with torch.no_grad():
            self.prior_router_horizon_logits.copy_(
                probabilities.log()[None, :].expand_as(self.prior_router_horizon_logits)
            )
            self.prior_router_global_logits.copy_(probabilities.log())

    def _set_interaction_router_prior(self, gate_prior: float) -> None:
        prior_weight = min(max(float(gate_prior), 0.02), 0.94)
        unique_weight = 0.01
        synergy_weight = 0.01
        numerical_weight = 1.0 - prior_weight - 2.0 * unique_weight - synergy_weight
        probabilities = torch.tensor(
            [numerical_weight, prior_weight, unique_weight, unique_weight, synergy_weight],
            dtype=self.interaction_router_horizon_logits.dtype,
            device=self.interaction_router_horizon_logits.device,
        )
        with torch.no_grad():
            self.interaction_router_horizon_logits.copy_(
                probabilities.log()[None, :].expand_as(self.interaction_router_horizon_logits)
            )

    def initialize_gate_prior(self, gate_prior: float) -> None:
        raw = (gate_prior - self.min_gate) / max(self.max_gate - self.min_gate, 1e-6)
        # Keep a small gradient reserve when validation selects an endpoint.
        raw = min(max(raw, 0.02), 0.98)
        logit = math.log(raw / (1.0 - raw))
        with torch.no_grad():
            self.global_gate_logit.fill_(logit)
            self.horizon_gate_logits.fill_(logit)
            self.source_horizon_logits.fill_(logit)
            self.gate_net[-1].bias.fill_(logit)
            self.sample_gate_net[-1].bias.fill_(logit)
        self._set_router_prior(gate_prior)
        self._set_prior_source_router_prior(gate_prior)
        self._set_interaction_router_prior(gate_prior)

    @staticmethod
    def _normalize_horizon(value: torch.Tensor) -> torch.Tensor:
        centered = value - value.mean(1, keepdim=True).detach()
        scale = torch.sqrt(torch.var(value, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return centered / scale

    @staticmethod
    def _series_features(x: torch.Tensor) -> torch.Tensor:
        series = x[:, :, 0]
        t = torch.linspace(-1.0, 1.0, series.size(1), device=x.device, dtype=x.dtype)
        mean = series.mean(1, keepdim=True)
        std = series.std(1, keepdim=True, unbiased=False)
        minimum = series.min(1, keepdim=True).values
        maximum = series.max(1, keepdim=True).values
        trend = ((series - mean) * t[None, :]).mean(1, keepdim=True)
        return torch.cat([mean, std, minimum, maximum, trend], dim=1)

    @staticmethod
    def _decomposition_features(x: torch.Tensor) -> torch.Tensor:
        series = x[:, :, 0]
        trend = F.avg_pool1d(
            F.pad(series[:, None, :], (2, 2), mode="replicate"), kernel_size=5, stride=1
        )[:, 0, :]
        residual = series - trend
        return torch.stack(
            [
                trend[:, -1] - trend[:, 0],
                trend.std(1, unbiased=False),
                residual.std(1, unbiased=False),
                residual.abs().mean(1),
                residual[:, -1],
            ],
            dim=1,
        )

    @staticmethod
    def _masked_source_stats(tokens: torch.Tensor, mask: torch.Tensor):
        weights = mask[:, :, None]
        count = mask.sum(1, keepdim=True)
        pooled = (tokens * weights).sum(1) / count.clamp_min(1.0)
        squared_distance = (tokens - pooled[:, None, :]).square().mean(-1)
        dispersion = (squared_distance * mask).sum(1, keepdim=True) / count.clamp_min(1.0)
        available = (count > 0).to(tokens.dtype)
        pooled = pooled * available
        return pooled, dispersion, count, available

    def _source_gate_features(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        x: torch.Tensor,
        endo: torch.Tensor,
        text_pred: torch.Tensor,
    ) -> torch.Tensor:
        split = min(self.source_tokens, text_tokens.size(1))
        report_tokens, report_mask = text_tokens[:, :split], text_mask[:, :split]
        search_tokens, search_mask = text_tokens[:, split:], text_mask[:, split:]
        report, report_dispersion, report_count, report_available = self._masked_source_stats(
            report_tokens, report_mask
        )
        if search_tokens.size(1) == 0:
            search = torch.zeros_like(report)
            search_dispersion = report_dispersion.new_zeros(report_dispersion.shape)
            search_count = report_count.new_zeros(report_count.shape)
            search_available = report_available.new_zeros(report_available.shape)
        else:
            search, search_dispersion, search_count, search_available = self._masked_source_stats(
                search_tokens, search_mask
            )
        agreement = F.cosine_similarity(report, search, dim=1).unsqueeze(1)
        agreement = agreement * report_available * search_available
        source_quality = torch.cat(
            [
                report_count / max(split, 1),
                search_count / max(text_tokens.size(1) - split, 1),
                torch.log1p(report_dispersion),
                torch.log1p(search_dispersion),
                agreement,
            ],
            dim=1,
        )
        disagreement = torch.cat(
            [endo[:, :, 0], text_pred[:, :, 0], torch.abs(endo[:, :, 0] - text_pred[:, :, 0])],
            dim=1,
        )
        return torch.cat(
            [
                self.report_context(report),
                self.search_context(search),
                source_quality,
                self._series_features(x),
                disagreement,
            ],
            dim=1,
        )

    def _source_experts(self, text_tokens: torch.Tensor, text_mask: torch.Tensor, prior_y):
        split = min(self.source_tokens, text_tokens.size(1))
        token_delta = self.text_branch(text_tokens)

        def pool(start: int, end: int):
            mask = text_mask[:, start:end]
            delta = token_delta[:, start:end]
            pooled = (delta * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
            prediction = self._normalize_horizon(pooled.unsqueeze(-1)) + prior_y
            return prediction, mask.any(1)

        report_pred, report_available = pool(0, split)
        search_pred, search_available = pool(split, text_tokens.size(1))
        return report_pred, search_pred, report_available, search_available

    def _attention_source_experts(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        endo: torch.Tensor,
        prior_y: torch.Tensor,
    ):
        split = min(self.source_tokens, text_tokens.size(1))
        query_values = torch.stack([endo[:, :, 0], prior_y[:, :, 0]], dim=2)
        query = self.attention_query_projection(query_values)
        query = query + self.attention_horizon_queries[None, :, :]

        def attend(tokens, mask, attention, head):
            if tokens.size(1) == 0:
                return prior_y, torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
            available = mask.any(1)
            safe_mask = mask.bool()
            if (~available).any():
                safe_mask = safe_mask.clone()
                safe_mask[~available, 0] = True
            if tokens.size(1) > self.attention_source_positions.size(0):
                raise ValueError("Source token count exceeds --source_tokens for attention")
            keys = self.attention_text_projection(tokens)
            keys = keys + self.attention_source_positions[None, : tokens.size(1), :]
            context, _ = attention(
                query,
                keys,
                keys,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            delta = head(context)
            prediction = prior_y + self._smooth_horizon(delta)
            return prediction, available

        report_pred, report_available = attend(
            text_tokens[:, :split],
            text_mask[:, :split],
            self.report_source_attention,
            self.report_attention_head,
        )
        search_pred, search_available = attend(
            text_tokens[:, split:],
            text_mask[:, split:],
            self.search_source_attention,
            self.search_attention_head,
        )
        return report_pred, search_pred, report_available, search_available

    def _router_features(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        x: torch.Tensor,
        endo: torch.Tensor,
        report_pred: torch.Tensor,
        search_pred: torch.Tensor,
    ) -> torch.Tensor:
        split = min(self.source_tokens, text_tokens.size(1))
        report, report_dispersion, report_count, report_available = self._masked_source_stats(
            text_tokens[:, :split], text_mask[:, :split]
        )
        search_tokens, search_mask = text_tokens[:, split:], text_mask[:, split:]
        if search_tokens.size(1):
            search, search_dispersion, search_count, search_available = self._masked_source_stats(
                search_tokens, search_mask
            )
        else:
            search = torch.zeros_like(report)
            search_dispersion = report_dispersion.new_zeros(report_dispersion.shape)
            search_count = report_count.new_zeros(report_count.shape)
            search_available = report_available.new_zeros(report_available.shape)
        agreement = F.cosine_similarity(report, search, dim=1).unsqueeze(1)
        agreement = agreement * report_available * search_available
        source_quality = torch.cat(
            [
                report_count / max(split, 1),
                search_count / max(text_tokens.size(1) - split, 1),
                torch.log1p(report_dispersion),
                torch.log1p(search_dispersion),
                agreement,
            ],
            dim=1,
        )
        disagreement = torch.cat(
            [
                endo[:, :, 0],
                report_pred[:, :, 0],
                search_pred[:, :, 0],
                torch.abs(endo[:, :, 0] - report_pred[:, :, 0]),
                torch.abs(endo[:, :, 0] - search_pred[:, :, 0]),
            ],
            dim=1,
        )
        return torch.cat(
            [
                self.report_context(report),
                self.search_context(search),
                source_quality,
                self._series_features(x),
                disagreement,
            ],
            dim=1,
        )

    def _prior_router_features(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        x: torch.Tensor,
        endo: torch.Tensor,
        prior_y: torch.Tensor,
        report_pred: torch.Tensor,
        search_pred: torch.Tensor,
    ) -> torch.Tensor:
        source_features = self._router_features(
            text_tokens, text_mask, x, endo, report_pred, search_pred
        )
        prior_features = torch.cat(
            [prior_y[:, :, 0], torch.abs(endo[:, :, 0] - prior_y[:, :, 0])],
            dim=1,
        )
        return torch.cat([source_features, prior_features], dim=1)

    def _smooth_horizon(self, value: torch.Tensor) -> torch.Tensor:
        if self.trend_kernel == 1:
            return value
        pad = self.trend_kernel // 2
        flattened = value.transpose(1, 2)
        smoothed = F.avg_pool1d(
            F.pad(flattened, (pad, pad), mode="replicate"),
            kernel_size=self.trend_kernel,
            stride=1,
        )
        return smoothed.transpose(1, 2)

    def _numerical_tokens(self, x: torch.Tensor):
        if not isinstance(self.backbone, PatchTST):
            raise RuntimeError("_numerical_tokens is only available for the PatchTST backbone")
        means = x.mean(1, keepdim=True).detach()
        normalized = x - means
        stdev = torch.sqrt(torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = normalized / stdev
        embedded, n_vars = self.backbone.patch_embedding(normalized.permute(0, 2, 1))
        encoded, _ = self.backbone.encoder(embedded)
        tokens = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])[:, 0]
        return tokens, means, stdev

    def _decode_tokens(self, tokens: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor):
        head_input = tokens[:, None, :, :].permute(0, 1, 3, 2)
        prediction = self.backbone.head(head_input).permute(0, 2, 1)
        prediction = prediction * stdev[:, 0, :].unsqueeze(1)
        return prediction + means[:, 0, :].unsqueeze(1)

    def forward(self, x, x_mark, dec_inp, y_mark, text_tokens, text_mask, prior_y, mode: str):
        self.last_source_weights = None
        self.last_source_predictions = None
        self.last_source_availability = None
        self.last_numerical_prediction = None
        self.last_expert_outputs = None
        self.last_expert_availability = None
        if mode == "cross_attention":
            numerical_tokens, means, stdev = self._numerical_tokens(x)
            text_context = self.cross_text_projection(text_tokens)
            attention_mask = text_mask.bool()
            empty_text = ~attention_mask.any(1)
            if empty_text.any():
                attention_mask = attention_mask.clone()
                attention_mask[empty_text, 0] = True
            decomposition = self.decomposition_projection(self._decomposition_features(x))
            query = numerical_tokens + decomposition[:, None, :]
            cross_context, _ = self.patch_cross_attention(
                query,
                text_context,
                text_context,
                key_padding_mask=~attention_mask,
                need_weights=False,
            )
            decomposition_tokens = decomposition[:, None, :].expand_as(numerical_tokens)
            patch_gate = torch.sigmoid(
                self.patch_injection_gate(
                    torch.cat([numerical_tokens, cross_context, decomposition_tokens], dim=-1)
                )
            )
            prediction = self._decode_tokens(
                numerical_tokens + patch_gate * cross_context, means, stdev
            )
            token_delta = self.text_branch(text_tokens)
            text_delta = (
                (token_delta * text_mask[:, :, None]).sum(1)
                / text_mask.sum(1, keepdim=True).clamp_min(1.0)
            ).unsqueeze(-1)
            text_pred = self._normalize_horizon(text_delta) + prior_y
            prediction = (1.0 - self.fixed_gate) * prediction + self.fixed_gate * text_pred
            horizon_gate = patch_gate.mean(1, keepdim=True).expand(-1, self.pred_len, -1)
            return prediction, horizon_gate, horizon_gate, text_pred
        endo = self.backbone(x, x_mark, dec_inp, y_mark)
        if mode in {"off", "off_continue"}:
            zero_gate = torch.zeros_like(endo)
            return endo, zero_gate, zero_gate, endo

        token_delta = self.text_branch(text_tokens)
        text_delta = (
            (token_delta * text_mask[:, :, None]).sum(1)
            / text_mask.sum(1, keepdim=True).clamp_min(1.0)
        ).unsqueeze(-1)
        text_pred = self._normalize_horizon(text_delta) + prior_y
        if mode in {"fixed", "fixed_continue", "fixed_high_continue"}:
            fixed_value = self.high_fixed_gate if mode == "fixed_high_continue" else self.fixed_gate
            gate = torch.full_like(endo, fixed_value)
            raw_gate = torch.full_like(endo, fixed_value)
        elif mode == "global_learned":
            raw_gate = torch.sigmoid(self.global_gate_logit).view(1, 1, 1).expand_as(endo)
            gate = self.min_gate + (self.max_gate - self.min_gate) * raw_gate
        elif mode == "horizon_only":
            raw_gate = torch.sigmoid(self.horizon_gate_logits).view(1, self.pred_len, 1)
            raw_gate = raw_gate.expand_as(endo)
            gate = self.min_gate + (self.max_gate - self.min_gate) * raw_gate
        elif mode == "source_factorized":
            source_features = self._source_gate_features(
                text_tokens, text_mask, x, endo, text_pred
            )
            sample_factors = self.source_reliability_net(source_features)
            interaction = sample_factors @ self.source_horizon_factors.transpose(0, 1)
            logits = self.source_horizon_logits[None, :] + interaction / math.sqrt(
                self.factor_rank
            )
            raw_gate = torch.sigmoid(logits).unsqueeze(-1)
            gate = self.min_gate + (self.max_gate - self.min_gate) * raw_gate
        elif mode == "source_router_trend":
            report_pred, search_pred, report_available, search_available = self._source_experts(
                text_tokens, text_mask, prior_y
            )
            router_features = self._router_features(
                text_tokens, text_mask, x, endo, report_pred, search_pred
            )
            sample_factors = self.router_reliability_net(router_features).reshape(
                -1, 2, self.factor_rank
            )
            interaction = torch.einsum(
                "ber,ehr->beh", sample_factors, self.router_horizon_factors
            ) / math.sqrt(self.factor_rank)
            base_logits = self.router_horizon_logits[None, :, :].expand(
                endo.size(0), -1, -1
            )
            logits = torch.cat(
                [base_logits[:, :, :1], base_logits[:, :, 1:] + interaction.permute(0, 2, 1)],
                dim=2,
            )
            availability = torch.stack(
                [
                    torch.ones_like(report_available),
                    report_available,
                    search_available,
                ],
                dim=1,
            )
            logits = logits.masked_fill(~availability[:, None, :], -1e4)
            source_weights = torch.softmax(logits, dim=2)
            report_correction = self._smooth_horizon(report_pred - endo)
            search_correction = self._smooth_horizon(search_pred - endo)
            prediction = (
                endo
                + source_weights[:, :, 1:2] * report_correction
                + source_weights[:, :, 2:3] * search_correction
            )
            raw_gate = source_weights[:, :, 1:].sum(2, keepdim=True)
            gate = raw_gate
            text_weight = raw_gate.clamp_min(1e-6)
            text_pred = (
                source_weights[:, :, 1:2] * report_pred
                + source_weights[:, :, 2:3] * search_pred
            ) / text_weight
            self.last_source_weights = source_weights
            self.last_source_predictions = (report_pred, search_pred)
            self.last_source_availability = (report_available, search_available)
            self.last_numerical_prediction = endo
            return prediction, gate, raw_gate, text_pred
        elif mode in FOUR_EXPERT_ROUTER_MODES:
            report_pred, search_pred, report_available, search_available = self._source_experts(
                text_tokens, text_mask, prior_y
            )
            router_features = self._prior_router_features(
                text_tokens, text_mask, x, endo, prior_y, report_pred, search_pred
            )
            if mode == "prior_source_router_no_context":
                router_features = router_features.clone()
                router_features[:, : 2 * self.source_context_dim] = 0.0
            if mode == "prior_router_horizon_only":
                logits = self.prior_router_horizon_logits[None, :, :].expand(
                    endo.size(0), -1, -1
                )
            elif mode == "prior_router_sample_only":
                sample_offsets = self.prior_router_sample_net(router_features)
                global_logits = self.prior_router_global_logits[None, :].expand(
                    endo.size(0), -1
                )
                sample_logits = torch.cat(
                    [global_logits[:, :1], global_logits[:, 1:] + sample_offsets], dim=1
                )
                logits = sample_logits[:, None, :].expand(-1, self.pred_len, -1)
            elif mode == "prior_router_full_mlp":
                sample_horizon_offsets = self.prior_router_full_net(router_features).reshape(
                    -1, self.pred_len, 3
                )
                base_logits = self.prior_router_horizon_logits[None, :, :].expand(
                    endo.size(0), -1, -1
                )
                logits = torch.cat(
                    [
                        base_logits[:, :, :1],
                        base_logits[:, :, 1:] + sample_horizon_offsets,
                    ],
                    dim=2,
                )
            else:
                sample_factors = self.prior_router_reliability_net(router_features).reshape(
                    -1, 3, self.factor_rank
                )
                interaction = torch.einsum(
                    "ber,ehr->beh", sample_factors, self.prior_router_horizon_factors
                ) / math.sqrt(self.factor_rank)
                base_logits = self.prior_router_horizon_logits[None, :, :].expand(
                    endo.size(0), -1, -1
                )
                logits = torch.cat(
                    [
                        base_logits[:, :, :1],
                        base_logits[:, :, 1:] + interaction.permute(0, 2, 1),
                    ],
                    dim=2,
                )
            availability = torch.stack(
                [
                    torch.ones_like(report_available),
                    torch.ones_like(report_available),
                    report_available,
                    search_available,
                ],
                dim=1,
            )
            logits = logits.masked_fill(~availability[:, None, :], -1e4)
            expert_weights = torch.softmax(logits, dim=2)
            report_expert = prior_y + self._smooth_horizon(report_pred - prior_y)
            search_expert = prior_y + self._smooth_horizon(search_pred - prior_y)
            expert_outputs = torch.cat(
                [endo, prior_y, report_expert, search_expert], dim=2
            )
            prediction = (
                expert_weights[:, :, 0:1] * endo
                + expert_weights[:, :, 1:2] * prior_y
                + expert_weights[:, :, 2:3] * report_expert
                + expert_weights[:, :, 3:4] * search_expert
            )
            raw_gate = expert_weights[:, :, 1:].sum(2, keepdim=True)
            gate = raw_gate
            semantic_weight = expert_weights[:, :, 2:].sum(2, keepdim=True).clamp_min(1e-6)
            text_pred = (
                expert_weights[:, :, 2:3] * report_expert
                + expert_weights[:, :, 3:4] * search_expert
            ) / semantic_weight
            self.last_source_weights = expert_weights
            self.last_source_predictions = (report_pred, search_pred)
            self.last_source_availability = (report_available, search_available)
            self.last_numerical_prediction = endo
            self.last_expert_outputs = expert_outputs
            self.last_expert_availability = availability
            return prediction, gate, raw_gate, text_pred
        elif mode == "prior_interaction_router":
            report_pred, search_pred, report_available, search_available = self._source_experts(
                text_tokens, text_mask, prior_y
            )
            router_features = self._prior_router_features(
                text_tokens, text_mask, x, endo, prior_y, report_pred, search_pred
            )
            sample_factors = self.interaction_router_reliability_net(router_features).reshape(
                -1, 4, self.factor_rank
            )
            interaction = torch.einsum(
                "ber,ehr->beh", sample_factors, self.interaction_router_horizon_factors
            ) / math.sqrt(self.factor_rank)
            base_logits = self.interaction_router_horizon_logits[None, :, :].expand(
                endo.size(0), -1, -1
            )
            logits = torch.cat(
                [base_logits[:, :, :1], base_logits[:, :, 1:] + interaction.permute(0, 2, 1)],
                dim=2,
            )
            both_available = report_available & search_available
            availability = torch.stack(
                [
                    torch.ones_like(report_available),
                    torch.ones_like(report_available),
                    report_available,
                    search_available,
                    both_available,
                ],
                dim=1,
            )
            expert_weights = torch.softmax(
                logits.masked_fill(~availability[:, None, :], -1e4), dim=2
            )
            report_unique = prior_y + self._smooth_horizon(report_pred - prior_y)
            search_unique = prior_y + self._smooth_horizon(search_pred - prior_y)
            synergy_delta = self.synergy_delta_net(router_features).unsqueeze(-1)
            synergy_base = 0.5 * (report_unique + search_unique)
            synergy_expert = synergy_base + self._smooth_horizon(synergy_delta)
            expert_outputs = torch.stack(
                [endo, prior_y, report_unique, search_unique, synergy_expert],
                dim=2,
            )
            prediction = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=2)
            raw_gate = expert_weights[:, :, 1:].sum(2, keepdim=True)
            semantic_weight = expert_weights[:, :, 2:].sum(2, keepdim=True).clamp_min(1e-6)
            text_pred = (
                expert_weights[:, :, 2:3] * report_unique
                + expert_weights[:, :, 3:4] * search_unique
                + expert_weights[:, :, 4:5] * synergy_expert
            ) / semantic_weight
            self.last_source_weights = expert_weights
            self.last_source_predictions = (report_pred, search_pred)
            self.last_source_availability = (report_available, search_available)
            self.last_numerical_prediction = endo
            self.last_interaction_expert_outputs = expert_outputs
            return prediction, raw_gate, raw_gate, text_pred
        elif mode == "prior_source_attention_router":
            report_pred, search_pred, report_available, search_available = (
                self._attention_source_experts(text_tokens, text_mask, endo, prior_y)
            )
            router_features = self._prior_router_features(
                text_tokens, text_mask, x, endo, prior_y, report_pred, search_pred
            )
            sample_factors = self.prior_router_reliability_net(router_features).reshape(
                -1, 3, self.factor_rank
            )
            interaction = torch.einsum(
                "ber,ehr->beh", sample_factors, self.prior_router_horizon_factors
            ) / math.sqrt(self.factor_rank)
            base_logits = self.prior_router_horizon_logits[None, :, :].expand(
                endo.size(0), -1, -1
            )
            logits = torch.cat(
                [base_logits[:, :, :1], base_logits[:, :, 1:] + interaction.permute(0, 2, 1)],
                dim=2,
            )
            availability = torch.stack(
                [
                    torch.ones_like(report_available),
                    torch.ones_like(report_available),
                    report_available,
                    search_available,
                ],
                dim=1,
            )
            expert_weights = torch.softmax(
                logits.masked_fill(~availability[:, None, :], -1e4), dim=2
            )
            prediction = (
                expert_weights[:, :, 0:1] * endo
                + expert_weights[:, :, 1:2] * prior_y
                + expert_weights[:, :, 2:3] * report_pred
                + expert_weights[:, :, 3:4] * search_pred
            )
            raw_gate = expert_weights[:, :, 1:].sum(2, keepdim=True)
            semantic_weight = expert_weights[:, :, 2:].sum(2, keepdim=True).clamp_min(1e-6)
            text_pred = (
                expert_weights[:, :, 2:3] * report_pred
                + expert_weights[:, :, 3:4] * search_pred
            ) / semantic_weight
            self.last_source_weights = expert_weights
            self.last_source_predictions = (report_pred, search_pred)
            self.last_source_availability = (report_available, search_available)
            self.last_numerical_prediction = endo
            return prediction, raw_gate, raw_gate, text_pred
        else:
            pooled_text = (
                (text_tokens * text_mask[:, :, None]).sum(1)
                / text_mask.sum(1, keepdim=True).clamp_min(1.0)
            )
            context = self.text_context(pooled_text)
            disagreement = torch.cat(
                [endo[:, :, 0], text_pred[:, :, 0], torch.abs(endo[:, :, 0] - text_pred[:, :, 0])],
                dim=1,
            )
            gate_features = torch.cat([context, self._series_features(x), disagreement], dim=1)
            if mode == "sample_only":
                raw_gate = torch.sigmoid(self.sample_gate_net(gate_features)).view(-1, 1, 1)
                raw_gate = raw_gate.expand(-1, self.pred_len, -1)
            else:
                raw_gate = torch.sigmoid(self.gate_net(gate_features)).unsqueeze(-1)
            gate = self.min_gate + (self.max_gate - self.min_gate) * raw_gate
        prediction = (1.0 - gate) * endo + gate * text_pred
        return prediction, gate, raw_gate, text_pred

# Public name; preserves the original parameter names and initialization.
SCF = NativeUtilityFusion
