import torch
from torch import nn

from model.base_pfn import BasePFN, AttentionAVGPool

from .mup_imp.mup_custom import MupMultiheadAttention
from mup.init import xavier_uniform_ as mup_xavier_uniform_

import torch
import torch.nn as nn
from .base_pfn import BasePFN, AttentionAVGPool
from functools import partial
from model.mup_imp.mup_custom import MuReadoutModified

from torch.optim.swa_utils import AveragedModel


class MobileNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        **kwargs
    ):
        super(MobileNetBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                **kwargs,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                **kwargs,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.out_channels = out_channels

    def forward(self, x):
        bs, ds, sl, _ = x.shape
        projected = self.proj(x)
        x = x.transpose(-1, -2)
        x = x.flatten(0, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(-1, -2)
        x = x.reshape(bs, ds, sl, self.out_channels)
        
        # The residual connection is only valid if the number of channels is the same.
        if projected.shape == x.shape:
            return x + projected
        else:
            return x


class FFBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super(FFBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
        )
        if d_in != d_out:
            self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        residual = x
        x = self.ff(x)
        if x.shape[-1] != residual.shape[-1]:
            residual = self.proj(residual)
        return x + residual


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout, use_mup_parametrization=True):
        super(SelfAttentionBlock, self).__init__()

        attn_cls = (
            MupMultiheadAttention if use_mup_parametrization else nn.MultiheadAttention
        )

        self.attn = attn_cls(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        n_dims = len(x.shape)

        if n_dims > 3:
            a, b, c, d = x.shape
            x = x.flatten(0, 1)

        residual = x
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.ff(x)
        x = x + residual

        if n_dims > 3:
            x = x.view(a, b, c, d)

        return x


class Convttention(nn.Module):
    """
    Implements the Lat-PFN (JEPA-style) Embedder via Dilated Mobilenets and Self-Attention.
    """

    def __init__(self, d_in, d_out, base=2, depth=8, use_mup_parametrization=True):
        super().__init__()

        self.mobilenet = nn.Sequential(
            MobileNetBlock(d_in + 1, d_out, 3, 1, 1, 1, padding_mode="replicate"),
            *[
                MobileNetBlock(d_out, d_out, 3, 1, base**i, base**i)
                for i in range(1, depth + 1)
            ],
        )

        self.mlp = nn.Sequential(
            FFBlock(d_out, d_out, 0.1),
            SelfAttentionBlock(
                d_out, 4, 0.1, use_mup_parametrization=use_mup_parametrization
            ),
            nn.Linear(d_out, d_out),
        )
        self.d_model = d_out

    def forward(self, history_T, history_V, prompt_T=None):
        for_sort = torch.cat(
            [history_T] + ([prompt_T] if prompt_T is not None else []), dim=-2
        ).argsort(dim=-2)
        history = torch.cat([history_T, history_V, torch.zeros_like(history_T)], dim=-1)
        if prompt_T is not None:
            prompt = torch.cat(
                [prompt_T, torch.zeros_like(prompt_T), torch.ones_like(prompt_T)],
                dim=-1,
            )
            history = torch.cat([history, prompt], dim=-2)
        output = self.mlp(
            self.mobilenet(history.gather(dim=-2, index=for_sort.repeat(1, 1, 1, 3)))
        ).gather(dim=-2, index=for_sort.argsort(dim=-2).repeat(1, 1, 1, self.d_model))
        if prompt_T is not None:
            prompt_dim = prompt_T.shape[-2]
            return (
                output[..., :-prompt_dim, :],
                output[..., -prompt_dim:, :],
            )
        else:
            return output, None


class ScheduledEma(float):
    """
    Class used as a wrapper for the EMA decay constant, so that we can update by pointer.
    """

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value.float()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __truediv__(self, other):
        return self.value / other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rmul__(self, other):
        return other * self.value

    def __rtruediv__(self, other):
        return other / self.value


class LaTPFNV4(nn.Module):
    def __init__(
        self,
        d_model=128,
        d_ff=256,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        n_outputs=1,
        use_mup_parametrization: bool = True,
        n_domain_params=10,
        device="cuda",
        train_noise=0.02,
        masking_type="independent",
        ema_decay=0.999,
        ema_warmup_iterations=250 * 50,
        *,
        shape,
        **kwargs
    ):
        super().__init__()

        print("Ignoring kwargs:", self, kwargs)
        print("d_model", d_model)
        print("using mup parametrization", use_mup_parametrization)

        self.n_outputs = n_outputs
        self.train_noise = train_noise

        # --- FIX Part 1: ADD the embedding layer ---
        self.value_embedder = nn.Linear(shape.n_features, 1)

        # --- FIX Part 2: CHANGE d_in for Convttention to 1 ---
        self.TS_encoder = Convttention(
            1, d_model, base=2, depth=8, use_mup_parametrization=use_mup_parametrization
        )

        self.ts_ema_constant = ScheduledEma(
            value=torch.scalar_tensor(ema_decay, dtype=torch.float64)
        )
        self.ema_decay = ema_decay
        self.ema_warmup_iterations = ema_warmup_iterations
        self.TS_ema = AveragedModel(
            self.TS_encoder,
            device=device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.ts_ema_constant
            ),
        )
        self.proj = nn.Sequential(
            *[FFBlock(d_model, d_model, dropout) for _ in range(1)]
        )
        self.avg_pool = AttentionAVGPool(
            d_model, nhead, dropout, use_mup_parametrization=use_mup_parametrization
        )
        self.pfn = BasePFN(
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_mup_parametrization=use_mup_parametrization,
            masking_type=masking_type,
        )
        last_layer_clss = (
            partial(MuReadoutModified, output_mult=2)
            if use_mup_parametrization
            else nn.Linear
        )
        self.head_raw = nn.Sequential(
            *[FFBlock(d_model, d_model, dropout) for _ in range(num_layers)],
            last_layer_clss(d_model, n_outputs, bias=False),
        )
        self.head_di = nn.Sequential(
            *[FFBlock(d_model, d_model, dropout) for _ in range(2)]
        )
        self.last_layer_di = last_layer_clss(d_model, n_domain_params)

    def update_emas(self):
        self.TS_ema.update_parameters(self.TS_encoder)
        self.ts_ema_constant._value = torch.clip(
            self.ts_ema_constant._value
            + ((1 - self.ema_decay) / self.ema_warmup_iterations),
            0.995,
            1.0,
        )

    def create_embeddings(
        self, T: torch.Tensor, V: torch.Tensor, supress_warnings: bool = False
    ) -> dict[str, torch.Tensor]:
        # ... (assertions) ...
        with torch.no_grad():
            returnables = {}
            
            # --- The Fix ---
            # Apply the value embedding before concatenation
            V = self.value_embedder(V)
            full_input = torch.cat([T, V], dim=-1)

            ema_output = self.TS_ema(full_input)

            returnables["per_timestep_embedding"] = ema_output
            # ... (rest of the function)

    def forward(
        self,
        T_context_history,
        T_context_prompt,
        V_context_history,
        V_context_prompt,
        T_heldout_history,
        T_heldout_prompt,
        V_heldout_history,
        V_heldout_prompt=None,
        predict_all_heads: bool = False,
        backcast: bool = False,
        **kwargs
    ):
        # --- FIX Part 3: Apply the embedding layer to all V tensors FIRST ---
        V_context_history = self.value_embedder(V_context_history)
        V_context_prompt = self.value_embedder(V_context_prompt)
        V_heldout_history = self.value_embedder(V_heldout_history)
        if V_heldout_prompt is not None:
            V_heldout_prompt = self.value_embedder(V_heldout_prompt)

        # --- The original logic now works because V tensors are the correct shape ---
        embedding_context, _ = self.TS_encoder(
            torch.cat([T_context_history, T_context_prompt], dim=-2),
            torch.cat([V_context_history, V_context_prompt], dim=-2),
        )
        mean_context = self.avg_pool(self.proj(embedding_context))
        embedding_heldout_history, prompt = self.TS_encoder(
            T_heldout_history, V_heldout_history, T_heldout_prompt
        )
        pred = self.pfn(mean_context, prompt)
        noise_fn = torch.randn_like if self.training else torch.zeros_like
        noise = noise_fn(pred.detach()) * self.train_noise
        prediction_raw = self.head_raw(pred.detach() + noise)
        returnables = dict(forecast=prediction_raw)

        if predict_all_heads:
            with torch.no_grad():
                ema = self.TS_ema(
                    torch.cat([T_heldout_history, T_heldout_prompt], dim=-2),
                    torch.cat([V_heldout_history, V_heldout_prompt], dim=-2),
                )[0]
                returnables["latent_target"] = ema[
                    :, :, -V_heldout_prompt.shape[-2] :, :
                ]
                returnables["latent_full"] = ema
                returnables["latent_history"] = embedding_heldout_history
                returnables["bypass"] = prompt
            returnables["latent_prediction"] = pred
            domain_identification_prediction_context = self.last_layer_di(
                self.head_di(mean_context)
            )
            heldout = torch.cat([embedding_heldout_history, pred], dim=-2)
            mean_heldout = self.avg_pool(self.proj(heldout))
            returnables["avg"] = torch.cat([mean_context, mean_heldout], dim=-2)
            domain_identification_prediction_heldout = self.last_layer_di(
                self.head_di(mean_heldout)
            )
            returnables["domain_identification_prediction_context"] = (
                domain_identification_prediction_context
            )
            returnables["domain_identification_prediction_heldout"] = (
                domain_identification_prediction_heldout
            )
        if backcast:
            with torch.no_grad():
                returnables["backcast"] = self.head_raw(embedding_heldout_history)
        return returnables
