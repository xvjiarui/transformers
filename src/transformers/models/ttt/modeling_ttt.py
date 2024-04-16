"""PyTorch TTT model."""

from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import ModelOutput, logging
from .configuration_ttt import TttConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TttConfig"


class TttCache:
    def __init__(self, config, batch_size, dtype=torch.float32, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.inner_chunk_size = config.inner_net_chunk_size

        self.params_dic = defaultdict(dict)
        self.param_names = ["W1", "b1"]

    def update(self, py_tree, layer_idx, seq_len):
        # print('update', seq_len, self.inner_chunk_size, self.seqlen_offset)
        if seq_len % self.inner_chunk_size == 0:
            for name in self.param_names:
                self.params_dic[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                self.params_dic[f"{name}_grad"][layer_idx].zero_()
            # print('update seq_len % self.inner_chunk_size == 0')
        elif seq_len < self.inner_chunk_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.inner_chunk_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.inner_chunk_size == 0:
                for name in self.param_names:
                    self.params_dic[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                    self.params_dic[f"{name}_grad"][layer_idx].zero_()
                # print('update seq_len + self.self.seqlen_offset % self.inner_chunk_size == 0')
            else:
                for name in self.param_names:
                    self.params_dic[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")
    # for vmap
    def to_dic(self, layer_idx):
        return {name: self.params_dic[name][layer_idx] for name in self.params_dic}


class TttRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        TttRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(TttRMSNorm)


class TttMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


# Function to unpack tensors along the first dimension
def unpack_tensors(tensor_dict):
    # Determine the number of items to unpack (length of first dimension)
    num_items = next(iter(tensor_dict.values())).shape[0]

    # Initialize a list to hold the unpacked dictionaries
    unpacked_list = []

    for i in range(num_items):
        # Create a new dictionary for each item, slicing each tensor along the first dimension
        item_dict = {key: tensor[i].clone() for key, tensor in tensor_dict.items()}
        unpacked_list.append(item_dict)

    return unpacked_list


def scan(f, init, xs, length=None):
    """Minic jax.lax.scan function."""
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    if isinstance(xs, dict):
        xs = unpack_tensors(xs)
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)


class TttBaseModule(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.width = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.inner_chunk_size = config.inner_net_chunk_size

        token_idx = 1.0 / torch.arange(1, self.inner_chunk_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)

        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        self.decoder_ln_fn = partial(F.layer_norm, normalized_shape=[self.head_dim], eps=1e-6)
        # prepending head dim
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ln_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ln_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

        self.gate_ilr_fn = F.linear
        # prepending head dim
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.linear_weight = nn.Parameter(
            torch.stack([torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)], dim=0)
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        # init bias to 0 following original JAX impl.
        self.linear_bias = nn.Parameter(
            torch.stack([torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)], dim=0)
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _split_chunks(self, hidden_states, inner_chunk_size=None):
        B, N, num_head, head_dim = hidden_states.shape
        # @xinhao: 2 means two chunks as a group to use gradient checkpointing
        # T=2048, optimal ckpt num = sqrt(T) ~= 45
        # Since CS=16, when 4 chunks are grouped, ckpt num = 2048 / 64 = 32, which is closest to 45
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size
        hidden_states = hidden_states.reshape(B, -1, inner_chunk_size, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )  # [B,nh,n_chunk,K,f]
        return hidden_states

    def get_inner_loop_inputs(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
    ):
        batch = hidden_states
        B, L, C = batch.shape
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        if cache_params is not None:
            inner_chunk_step_offset = cache_params.seqlen_offset % self.inner_chunk_size
            # print('inner_chunk_step_offset', inner_chunk_step_offset)
        else:
            inner_chunk_step_offset = 0

        n_chunk = L // inner_chunk_size
        # [B ,n_chunk, inner_chunk_size, C]
        X = batch.reshape(B, n_chunk, inner_chunk_size, self.width)

        XC, XB, XA = self.q_proj(batch), self.k_proj(batch), self.v_proj(batch)

        XC = self._split_heads(XC)
        XB = self._split_heads(XB)
        XA = self._split_heads(XA)  # [B,nh,n_chunk / g, g * K,f]

        XC = self._split_chunks(XC, inner_chunk_size)
        XB = self._split_chunks(XB, inner_chunk_size)
        XA = self._split_chunks(XA, inner_chunk_size)

        # [B, num_heads, n_chunk, inner_chunk_size, 1]
        ilr_gated = torch.vmap(self.gate_ilr_fn, in_dims=(None, 0, 0), out_dims=1)(
            X, self.linear_weight, self.linear_bias
        )
        ilr_gated = F.sigmoid(ilr_gated)

        # [B, L]
        token_idx = self.token_idx[inner_chunk_step_offset : inner_chunk_step_offset + inner_chunk_size]

        coeff = (self.config.inner_net_lr * token_idx).reshape(1, 1, 1, -1, 1) * ilr_gated / self.head_dim

        return XC, XB, XA, coeff

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
        inner_chunk_size: Optional[int] = None,
        last_chunk_params_dic: Optional[Dict[str, torch.Tensor]] = None,
        return_params: Optional[bool] = False,
    ):
        XC, XB, XA, coeff = self.get_inner_loop_inputs(
            hidden_states, position_ids=position_ids, cache_params=cache_params, inner_chunk_size=inner_chunk_size
        )

        XCW_batch, batch_params_dic = self.process_inner_loop(
            XC,
            XB,
            XA,
            coeff,
            inner_chunk_size=inner_chunk_size,
            last_chunk_params_dic=last_chunk_params_dic,
            cache_params=cache_params,
        )
        z_batch = self.project_inner_loop_outputs(XCW_batch)

        if return_params:
            return z_batch, batch_params_dic
        else:
            return z_batch

    def project_inner_loop_outputs(self, XCW_batch):
        """
        Inputs
            XCW_batch: [B,N,F]
        Outputs
            z_batch: [B,N,F]
        """
        z_batch = self.o_proj(XCW_batch)
        return z_batch

    def process_inner_loop(self, XC, XB, XA, coeff, inner_chunk_size, last_chunk_params_dic, cache_params=None):
        """
        Inputs:
            XA, XB, XC: [B, n_chunk, chunk_size, F] or [B, n_chunk // 4, 4 * chunk_size, F]
            coeff: [B, n_chunk, chunk_size, 1] or [B,nh, n_chunk / 4, 4 * K, 1]
        Outputs:
            [B,N,F]
        """
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
    ):
        L = hidden_states.shape[1]
        reminder_len = L % self.inner_chunk_size
        num_chunks = L // self.inner_chunk_size
        output_hidden_states = []
        last_chunk_params_dic = None
        if num_chunks > 0:
            chunk_hidden_states, last_chunk_params_dic = self.forward_chunk(
                hidden_states[:, : num_chunks * self.inner_chunk_size],
                position_ids=position_ids[:, : num_chunks * self.inner_chunk_size]
                if position_ids is not None
                else None,
                cache_params=cache_params,
                return_params=True,
            )
            output_hidden_states.append(chunk_hidden_states)
        if reminder_len > 0:
            output_hidden_states.append(
                self.forward_chunk(
                    hidden_states[:, -reminder_len:],
                    position_ids=position_ids[:, -reminder_len:] if position_ids is not None else None,
                    cache_params=cache_params,
                    inner_chunk_size=reminder_len,
                    last_chunk_params_dic=last_chunk_params_dic,
                )
            )

        output_hidden_states = torch.cat(output_hidden_states, dim=1)

        return output_hidden_states


class TTTLayerNormModule(torch.nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, input, label):
        D = input.shape[-1]
        mu = input.mean(dim=1, keepdim=True)
        var = input.var(dim=1, keepdim=True, unbiased=False)

        std = torch.sqrt(var + self.eps)
        x_hat = (input - mu) / std
        y = self.gamma * x_hat + self.beta

        grad_output = y - label
        grad_x_hat = grad_output * self.gamma
        z = (
            (1.0 / D)
            * (
                D * grad_x_hat
                - grad_x_hat.sum(dim=1, keepdim=True)
                - x_hat * (grad_x_hat * x_hat).sum(dim=1, keepdim=True)
            )
            / std
        )

        return z

def decoder_ln_bwd(input, label, gamma, beta, eps=1e-6):
    D = input.shape[-1]
    mu = input.mean(dim=1, keepdim=True)
    var = input.var(dim=1, keepdim=True, unbiased=False)

    std = torch.sqrt(var + eps)
    x_hat = (input - mu) / std
    y = gamma * x_hat + beta

    grad_output = y - label
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=1, keepdim=True)
        )
        / std
    )

    return z
    

class TttM1Module(TttBaseModule):
    def __init__(self, config: TttConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process_inner_loop(self, XC, XB, XA, coeff, inner_chunk_size, last_chunk_params_dic, cache_params):
        if inner_chunk_size is None:
            inner_chunk_size = self.inner_chunk_size

        B = XA.shape[0]  # [B,nh,n_chunk / 4, 4 * K,f]
        L = XA.shape[2] * XA.shape[3]

        @torch.vmap
        def update_embed(XA, XB, XC, coeff, last_params_dic=None):
            if last_params_dic is not None:
                params_dic = last_params_dic
            else:
                params_dic = {
                    "W1_states": self.W1,
                    "b1_states": self.b1,
                }
                if cache_params is not None:
                    params_dic.upadte(
                        {
                            "W1_grad": torch.zeros_like(self.W1),
                            "b1_grad": torch.zeros_like(self.b1),
                        }
                    )

            @torch.vmap
            def parallelize_over_heads(XA, XB, XC, coeff, init_params_dic, ln_weight, ln_bias):
                def compute_chunk(params_dic, inputs):
                    # W_init_chunk: [C, C]
                    W1_init = params_dic["W1_states"]
                    b1_init = params_dic["b1_states"]

                    # [inner_chunk_size, C]
                    XA_chunk = inputs["XA"]
                    # [inner_chunk_size, C]
                    XB_chunk = inputs["XB"]
                    # [inner_chunk_size, C]
                    XC_chunk = inputs["XC"]
                    # [inner_chunk_size, 1]
                    coeff_chunk = inputs["coeff"]

                    X1 = XB_chunk
                    Z1 = X1 @ W1_init + b1_init

                    if self.config.use_vjp:
                        DLN_out, LN_vjp = torch.func.vjp(
                            lambda z: self.decoder_ln_fn(z, weight=ln_weight, bias=ln_bias), Z1
                        )
                        grad_l_wrt_DLN_out = DLN_out - XA_chunk  # [K,f]
                        grad_l_wrt_Z1 = LN_vjp(grad_l_wrt_DLN_out)[0]  # [K,f]
                    else:
                        grad_l_wrt_Z1 = decoder_ln_bwd(Z1, XA_chunk, ln_weight, ln_bias)

                    # NOTE: fractional forward, caching the gradients, and cumsum from cache
                    if cache_params is not None and inner_chunk_size % self.inner_chunk_size != 0:
                        # [K, C, C]
                        grad_W1 = (
                            torch.cumsum(torch.einsum("bi,bj->bij", XB_chunk, grad_l_wrt_Z1), dim=0)
                            + params_dic["W1_grad"]
                        )
                        # [K, C]
                        grad_b1 = torch.cumsum(grad_l_wrt_Z1, dim=0) + params_dic["b1_grad"]
                        b1_bar = b1_init - (coeff_chunk * grad_b1)  # [K, C]
                        # [K, C]
                        Z1_bar = (XC_chunk.unsqueeze(1) @ (W1_init - coeff_chunk.unsqueeze(-1) * grad_W1)).squeeze(
                            1
                        ) + b1_bar
                        XCW_chunk = Z1_bar

                        W1_last = W1_init - (coeff_chunk[-1] * grad_W1[-1])
                        b1_last = b1_bar[-1:]

                        last_param_dic = {
                            "W1_states": W1_last,
                            "b1_states": b1_last,
                            "W1_grad": grad_W1[-1],
                            "b1_grad": grad_b1[-1:],
                        }
                    else:

                        Attn1 = torch.tril(XC_chunk @ XB_chunk.transpose(1, 0))
                        b1_bar = b1_init - coeff_chunk * torch.cumsum(grad_l_wrt_Z1, dim=0)  # [K,f]
                        Z1_bar = XC_chunk @ W1_init - (coeff_chunk * Attn1) @ grad_l_wrt_Z1 + b1_bar
                        XCW_chunk = Z1_bar

                        W1_last = W1_init - (coeff_chunk[-1] * X1).transpose(1, 0) @ grad_l_wrt_Z1
                        b1_last = b1_bar[-1:]

                        last_param_dic = {
                            "W1_states": W1_last,
                            "b1_states": b1_last,
                            "W1_grad": torch.zeros_like(W1_init),
                            "b1_grad": torch.zeros_like(b1_init),
                        }

                    return last_param_dic, XCW_chunk

                inputs = {"XA": XA, "XB": XB, "XC": XC, "coeff": coeff}
                # [n_chunk, inner_chunk_size, head_dim]
                output_params_dic, XCW = scan(compute_chunk, init_params_dic, inputs)
                return XCW.reshape(-1, self.head_dim), output_params_dic

            # [num_heads, L, C]
            return parallelize_over_heads(XA, XB, XC, coeff, params_dic, self.ln_weight, self.ln_bias)

        # in this case, we are decoding
        if last_chunk_params_dic is None and cache_params is not None:
            last_chunk_params_dic = cache_params.to_dic(self.layer_idx)

        if last_chunk_params_dic is not None:
            XCW_batch, batch_params_dic = update_embed(XA, XB, XC, coeff, last_chunk_params_dic)
        else:
            XCW_batch, batch_params_dic = update_embed(XA, XB, XC, coeff)

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dic, self.layer_idx, L)

        # [B, L, C]
        XCW_batch = XCW_batch.permute(0, 2, 1, 3).reshape(B, L, -1)

        return XCW_batch, batch_params_dic


class TttDecoderLayer(nn.Module):
    def __init__(self, config: TttConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # TODO(jiarui): rename `self_attn` to ttt related?
        # self.self_attn = TttModule(config=config, layer_idx=layer_idx)
        self.self_attn = TttM1Module(config=config, layer_idx=layer_idx)

        self.mlp = TttMLP(config)
        self.input_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TttCache] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # TTT
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TttPreTrainedModel(PreTrainedModel):
    config_class = TttConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TttDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class TttOutput(ModelOutput):
    """
    Class for the TTT model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`TttCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[TttCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TttCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`TttCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[TttCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TttModel(TttPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TttDecoderLayer`]

    Args:
        config: TttConfig
    """

    def __init__(self, config: TttConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TttDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TttRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TttCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_params is None and use_cache:
            cache_params = self.create_cache(inputs_embeds.size(0), inputs_embeds.device, inputs_embeds.dtype)

        seqlen_offset = 0
        if cache_params is not None:
            seqlen_offset = cache_params.seqlen_offset
        position_ids = torch.arange(
            seqlen_offset, seqlen_offset+ inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)

        # embed positions
        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_params,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_params=cache_params,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return TttOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )

    def create_cache(self, batch_size, device, dtype) -> TttCache:
        logger.info(f"Creating cache of size: {batch_size}")
        print("create_cache")
        cache = TttCache(self.config, batch_size, dtype=dtype, device=device)
        for layer_idx in range(self.config.num_hidden_layers):
            for name in cache.param_names:
                weight = getattr(self.layers[layer_idx].self_attn, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim())
                cache.params_dic[f"{name}_states"][layer_idx] = tiled_weight
                cache.params_dic[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

        return cache


class TttForCausalLM(TttPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = TttModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, cache_params: Optional[TttCache] = None, inputs_embeds=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, -1].unsqueeze(-1) if attention_mask is not None else None

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TttCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        *,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert not output_attentions, "output_attentions is not available in TttForCausalLM"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TttCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )
