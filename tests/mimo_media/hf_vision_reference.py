# The HF reference vision tower, verbatim from XiaomiMiMo/MiMo-V2.6-Flash-RL modeling_mimo_v2.py
# (lines 585-911, "Vision encoder"), with only its imports supplied. Used by
# vision_parity.py (torch, CPU) to check rapid_mlx.mlx_lm_models.mimo_v2_media.VisionModel.
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {"silu": nn.SiLU()}

def _rotate_half_vision(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half_vision(q) * sin)
    k_embed = (k * cos) + (_rotate_half_vision(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class MiMoVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class MiMoVisionPatchEmbed(nn.Module):
    def __init__(
        self, patch_size: int = 16, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1280
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class MiMoVisionSwiGLUMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=True)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=True)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=True)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoVisionAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_sinks: bool = False,
        window_size: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.window_size = window_size

        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv = nn.Linear(dim, qkv_dim, bias=True)
        self.proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=True)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads)) if use_sinks else None

    def _build_window_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.window_size <= 0:
            return None
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        mask = mask.masked_fill((row_idx - col_idx).abs() > self.window_size, float("-inf"))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        full_attn: bool = False,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[:, :q_dim].view(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].view(seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim :].view(seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        q_chunks = torch.split(q, lengths.tolist(), dim=0)
        k_chunks = torch.split(k, lengths.tolist(), dim=0)
        v_chunks = torch.split(v, lengths.tolist(), dim=0)

        outputs = []
        for q_c, k_c, v_c in zip(q_chunks, k_chunks, v_chunks):
            q_c = q_c.unsqueeze(0).transpose(1, 2)
            k_c = k_c.unsqueeze(0).transpose(1, 2)
            v_c = v_c.unsqueeze(0).transpose(1, 2)

            if self.num_kv_groups > 1:
                k_c = k_c.repeat_interleave(self.num_kv_groups, dim=1)
                v_c = v_c.repeat_interleave(self.num_kv_groups, dim=1)

            attn_mask = None
            if not full_attn:
                attn_mask = self._build_window_mask(q_c.shape[2], q_c.device, q_c.dtype)

            if self.sinks is not None:
                sink_bias = torch.zeros(
                    1, self.num_heads, q_c.shape[2], k_c.shape[2], device=q_c.device, dtype=q_c.dtype
                )
                sink_bias[..., 0] = self.sinks.view(1, self.num_heads, 1)
                attn_mask = sink_bias if attn_mask is None else attn_mask + sink_bias

            attn_out = F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=attn_mask, scale=self.scaling)
            outputs.append(attn_out.squeeze(0).transpose(0, 1))

        attn_output = torch.cat(outputs, dim=0)
        attn_output = attn_output.reshape(seq_len, -1)
        return self.proj(attn_output)


class MiMoVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        use_sinks: bool = False,
        window_size: int = -1,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=rms_norm_eps)
        self.attn = MiMoVisionAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            use_sinks=use_sinks, window_size=window_size,
        )
        self.mlp = MiMoVisionSwiGLUMLP(dim=dim, intermediate_dim=intermediate_dim, hidden_act=hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        full_attn: bool = False,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings, full_attn=full_attn,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MiMoVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class MiMoVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        depth = config.depth
        num_heads = config.num_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "qk_channels", 64)
        spatial_merge_size = getattr(config, "spatial_merge_size", 2)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.fullatt_block_indexes = getattr(config, "fullatt_block_indexes", [])
        use_sink = getattr(config, "use_sink", False)
        visual_token_window_size = getattr(config, "visual_token_window_size", -1)
        self.vit_window_attn_types = getattr(config, "vit_window_attn_types", None) or [-1] * depth

        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size

        self.patch_embed = MiMoVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=getattr(config, "in_channels", None) or getattr(config, "in_chans", 3),
            embed_dim=hidden_size,
        )

        self.rotary_pos_emb = MiMoVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    dim=hidden_size,
                    intermediate_dim=config.intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=rms_norm_eps,
                    use_sinks=use_sink and (i not in self.fullatt_block_indexes),
                    window_size=visual_token_window_size,
                )
                for i in range(depth)
            ]
        )

        self.merger = MiMoVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        tensor = tensor[index]
        return tensor.flatten(0, 1)

    def get_window_index_1d(self, grid_thw: torch.Tensor, col: bool = True) -> torch.Tensor:
        window_index = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            index_new = index.transpose(1, 2).reshape(-1) if col else index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        return torch.cat(window_index, dim=0)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = pixel_values.to(device=self.patch_embed.proj.weight.device, dtype=self.dtype)
        x = self.patch_embed(x)

        rotary_emb = self.rot_pos_emb(grid_thw)
        rotary_emb = rotary_emb.to(device=x.device)
        emb = torch.cat((rotary_emb, rotary_emb), dim=-1)

        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(device=x.device)
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col).to(device=x.device)

        row_based_embeddings = (emb.cos(), emb.sin())
        col_emb = self.apply_index(emb, window_index_1d_col)
        col_based_embeddings = (col_emb.cos(), col_emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device=x.device)

        for i, blk in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[i]

            if window_attn_type == 1 and (i == 0 or self.vit_window_attn_types[i - 1] != 1):
                x = self.apply_index(x, window_index_1d_col)

            if i > 0 and window_attn_type != 1 and self.vit_window_attn_types[i - 1] == 1:
                x = self.apply_index(x, reverse_window_index_1d_col)

            position_embeddings = col_based_embeddings if window_attn_type == 1 else row_based_embeddings
            full_attn = i in self.fullatt_block_indexes
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, full_attn=full_attn)

        return self.merger(x)
