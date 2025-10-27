# deltanet_attention.py
"""
Mathematically Correct DeltaNet Attention Implementation
Extracted from Qwen3-Next (transformers/models/qwen3_next/modeling_qwen3_next.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """L2 normalization - Line 432-435 from original"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class RMSNormGated(nn.Module):
    """Gated RMS Normalization - Lines 68-83 from original"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chunk-based Gated Delta Rule - Lines 438-515 from original
    Used during training for efficiency

    Args:
        query: [batch, seq_len, num_heads, head_dim]
        key: [batch, seq_len, num_heads, head_dim]
        value: [batch, seq_len, num_heads, head_dim]
        g: [batch, seq_len, num_heads] - decay factor
        beta: [batch, seq_len, num_heads] - learning rate

    Returns:
        output: [batch, seq_len, num_heads, head_dim]
        final_state: [batch, num_heads, head_dim, head_dim] or None
    """
    initial_dtype = query.dtype

    # L2 normalize queries and keys
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # Transpose to [batch, num_heads, seq_len, head_dim]
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Pad to chunk size
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    # Scale query
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    # Beta-weighted values and keys
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # Causal mask for intra-chunk attention
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Cumulative decay within chunks
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta  # Corrected values
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))  # Corrected keys with decay

    # Initialize recurrent state
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # Process each chunk
    num_chunks = total_sequence_length // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]

        # Intra-chunk attention with decay
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)

        # Predict values from recurrent state
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state

        # Compute delta (prediction error)
        v_new = v_i - v_prime

        # Output: attention from state + intra-chunk attention
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new

        # Update recurrent state
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    # Reshape and crop to original length
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Recurrent Gated Delta Rule - Lines 518-557 from original
    Used during inference (autoregressive generation)

    Processes tokens one-by-one with O(1) memory per token
    """
    initial_dtype = query.dtype

    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    # Process each token sequentially
    for i in range(sequence_length):
        q_t = query[:, :, i]  # [batch, heads, k_dim]
        k_t = key[:, :, i]  # [batch, heads, k_dim]
        v_t = value[:, :, i]  # [batch, heads, v_dim]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [batch, heads, 1, 1]
        beta_t = beta[:, :, i].unsqueeze(-1)  # [batch, heads, 1]

        # Decay recurrent state
        last_recurrent_state = last_recurrent_state * g_t

        # Predict value from state: v̂ = S @ k
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)

        # Compute delta (prediction error)
        delta = (v_t - kv_mem) * beta_t

        # Update state with delta: S += k^T @ delta
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)

        # Output: o = S @ q
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class DeltaNetAttention(nn.Module):
    """
    Complete DeltaNet Attention Module
    Based on Qwen3NextGatedDeltaNet (Lines 560-771)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        head_dim: int = 64,
        conv_kernel_size: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.eps = eps

        # Projection dimensions (matching lines 577-592)
        # Q, K, V, Z (gate) - all same size for simplicity
        projection_size_qkvz = self.total_dim * 4  # q, k, v, z
        projection_size_ba = num_heads * 2  # beta, alpha

        self.in_proj_qkvz = nn.Linear(hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(hidden_size, projection_size_ba, bias=False)

        # Causal convolution (lines 579-586)
        conv_dim = self.total_dim * 3  # q, k, v (not z)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=conv_dim,  # Depthwise convolution
            padding=conv_kernel_size - 1,
        )

        # Time step parameters (lines 596-599)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated normalization (lines 602-611)
        self.norm = RMSNormGated(head_dim, eps=eps)

        # Output projection
        self.out_proj = nn.Linear(self.total_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_recurrent: bool = False,
        initial_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            use_recurrent: If True, use recurrent mode (for inference)
            initial_state: Previous recurrent state if available
            return_state: Whether to return final state

        Returns:
            output: [batch, seq_len, hidden_size]
            final_state: [batch, num_heads, head_dim, head_dim] or None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V, Z, Beta, Alpha (lines 680-682)
        projected_qkvz = self.in_proj_qkvz(hidden_states)  # [B, S, total_dim * 4]
        projected_ba = self.in_proj_ba(hidden_states)  # [B, S, num_heads * 2]

        # Split projections (lines 627-654 simplified)
        q, k, v, z = projected_qkvz.chunk(4, dim=-1)
        b, a = projected_ba.chunk(2, dim=-1)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        z = z.view(batch_size, seq_len, self.num_heads, self.head_dim)
        b = b.view(batch_size, seq_len, self.num_heads)
        a = a.view(batch_size, seq_len, self.num_heads)

        # Flatten for convolution (lines 683-686)
        q_flat = q.reshape(batch_size, seq_len, self.total_dim)
        k_flat = k.reshape(batch_size, seq_len, self.total_dim)
        v_flat = v.reshape(batch_size, seq_len, self.total_dim)

        mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # [B, S, total_dim * 3]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, total_dim * 3, S]

        # Apply causal convolution (lines 702-711)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S, total_dim * 3]

        # Split after convolution (lines 714-725)
        q_conv, k_conv, v_conv = mixed_qkv.chunk(3, dim=-1)
        q = q_conv.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_conv.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v_conv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Compute beta and gamma (lines 727-729)
        beta = b.sigmoid()  # [B, S, H] ∈ (0, 1)
        g = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias
        )  # [B, S, H]

        # Apply gated delta rule
        if use_recurrent:
            core_attn_out, final_state = recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=return_state,
                use_qk_l2norm=True,
            )
        else:
            core_attn_out, final_state = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                chunk_size=64,
                initial_state=initial_state,
                output_final_state=return_state,
                use_qk_l2norm=True,
            )

        # Gated normalization (lines 762-768)
        batch_s, seq_s, heads_s, dim_s = core_attn_out.shape
        core_attn_out_flat = core_attn_out.reshape(-1, self.head_dim)
        z_flat = z.reshape(-1, self.head_dim)

        core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
        core_attn_out = core_attn_out_flat.reshape(batch_s, seq_s, heads_s, dim_s)
        core_attn_out = core_attn_out.reshape(batch_s, seq_s, self.total_dim)

        # Output projection (line 770)
        output = self.out_proj(core_attn_out)

        return output, final_state


class StandardSoftmaxAttention(nn.Module):
    """Standard O(N²) Softmax Attention for comparison"""

    def __init__(self, hidden_size: int, num_heads: int = 16, head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.scaling = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, self.total_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.total_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.total_dim, bias=False)
        self.o_proj = nn.Linear(self.total_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Softmax and attend
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.total_dim)
        output = self.o_proj(attn_output)

        return output


if __name__ == "__main__":
    # Quick test
    batch_size = 2
    seq_len = 128
    hidden_size = 512

    x = torch.randn(batch_size, seq_len, hidden_size)

    print("Testing DeltaNet Attention...")
    deltanet = DeltaNetAttention(hidden_size, num_heads=8, head_dim=64)
    out_delta, state = deltanet(x, return_state=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_delta.shape}")
    print(f"State shape: {state.shape if state is not None else None}")

    print("\nTesting Standard Attention...")
    standard = StandardSoftmaxAttention(hidden_size, num_heads=8, head_dim=64)
    out_standard = standard(x)
    print(f"Output shape: {out_standard.shape}")

    print("\n✓ Both implementations work!")
