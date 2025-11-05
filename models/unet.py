from typing import Optional, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# You must provide these from your project:
# conv_nd, normalization, ResBlock, AttentionBlock, TimestepEmbedSequential, Downsample, Upsample, zero_module
from .layers import *  # noqa


# ------------------------------------------------------------
# 0) Fourier / Continuous embeddings
# ------------------------------------------------------------
class FourierEmbedding(nn.Module):
    """Sin/Cos Fourier features for a scalar (e.g., tau in [0,1] or any real)."""
    def __init__(self, out_dim: int, max_freq: float = 1000.0):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even (sin/cos pairs)."
        half = out_dim // 2
        self.register_buffer(
            "freqs",
            th.exp(th.linspace(0.0, 1.0, steps=half) * th.log(th.tensor(max_freq))),
            persistent=False,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.dim() == 1:
            x = x[:, None]
        angles = x * self.freqs[None, :]
        return th.cat([th.sin(angles), th.cos(angles)], dim=-1)


class ContinuousTimeEmbedding(nn.Module):
    """
    Continuous scalar t -> Fourier features with optional normalization & learnable affine.
    """
    def __init__(self, out_dim: int, t_range: Optional[Tuple[float, float]] = None,
                 max_freq: float = 1000.0, learnable_affine: bool = True):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        self.t_range = t_range
        self.learnable_affine = learnable_affine
        self.fourier = FourierEmbedding(out_dim=out_dim, max_freq=max_freq)
        if learnable_affine:
            self.scale = nn.Parameter(th.ones(1))
            self.shift = nn.Parameter(th.zeros(1))
        else:
            self.register_buffer("scale", th.ones(1), persistent=False)
            self.register_buffer("shift", th.zeros(1), persistent=False)

    def forward(self, t: th.Tensor) -> th.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        t = t.to(dtype=th.float32)
        if self.t_range is not None:
            t_min, t_max = self.t_range
            denom = max(float(t_max - t_min), 1e-8)
            t = (t - float(t_min)) / denom
        t = self.scale * t + self.shift
        return self.fourier(t)


class FourierTimePE(nn.Module):
    """Sin/Cos PE for integer indices (not used below, kept for completeness)."""
    def __init__(self, d: int, max_freq: float = 300., min_freq: float = 1.0):
        super().__init__()
        assert d % 2 == 0
        h = d // 2
        freqs = th.exp(th.linspace(th.log(th.tensor(min_freq)),
                                   th.log(th.tensor(max_freq)), steps=h))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: th.Tensor):
        if t.dim() == 1:
            t = t[None, :]
        angles = 2 * th.pi * t[..., None] * self.freqs
        return th.cat([th.sin(angles), th.cos(angles)], dim=-1)


# ------------------------------------------------------------
# 1) Simple MLP
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, act=nn.GELU):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# 2) Observation encoders
# ------------------------------------------------------------
class ConvTrunk(nn.Module):
    """Small, stable CNN trunk for image inputs: 4~8x downsample with GN+GELU."""
    def __init__(self, in_ch: int, width: int = 64, depth: int = 3):
        super().__init__()
        ch = in_ch
        blocks = []
        for i in range(depth):
            stride = 2 if i > 0 else 1
            blocks += [
                nn.Conv2d(ch, width, 3, stride=stride, padding=1),
                nn.GroupNorm(8, width),
                nn.GELU(),
            ]
            ch = width
        self.net = nn.Sequential(*blocks)
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class InstantObsEncoder(nn.Module):
    """
    E_inst: z_t -> c_t
      - Vector: LayerNorm -> MLP -> LayerNorm
      - Image:  ConvTrunk(GN) -> GAP -> LayerNorm -> 2-layer head -> LayerNorm
      - z_t=None: learnable NULL vector (expanded to batch)
    """
    def __init__(self, in_dim: Union[int, Tuple[int,int,int]], out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self._is_vector = isinstance(in_dim, int)

        self.out_norm = nn.LayerNorm(out_dim)
        self.null_vec = nn.Parameter(th.zeros(1, out_dim))

        if self._is_vector:
            d_in = int(in_dim)
            self.in_norm_vec = nn.LayerNorm(d_in)
            self.vec_mlp = MLP(d_in, max(64, out_dim), out_dim, n_layers=3)
        else:
            C, H, W = in_dim  # type: ignore
            self.backbone = ConvTrunk(C, width=64, depth=3)
            D = self.backbone.out_ch
            self.g_norm = nn.LayerNorm(D)
            self.proj_vec = nn.Sequential(
                nn.Linear(D, max(64, out_dim)), nn.SiLU(),
                nn.Linear(max(64, out_dim), out_dim)
            )
            if isinstance(self.proj_vec[-1], nn.Linear):
                nn.init.zeros_(self.proj_vec[-1].weight)
                nn.init.zeros_(self.proj_vec[-1].bias)

    def forward(self, z_t: Optional[th.Tensor], batch_size: Optional[int] = None) -> th.Tensor:
        if z_t is None:
            B = 1 if batch_size is None else int(batch_size)
            c = self.null_vec.expand(B, -1)
            return self.out_norm(c)

        if self._is_vector:
            x = self.in_norm_vec(z_t)
            c = self.vec_mlp(x)
            return self.out_norm(c)

        f = self.backbone(z_t)                                  # [B, D, h, w]
        g = F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1) # [B, D]
        g = self.g_norm(g)
        c = self.proj_vec(g)                                    # [B, out_dim]
        return self.out_norm(c)


class SeqObsEncoder(nn.Module):
    """
    E_seq: encodes z_{1:S} with ABSOLUTE + RELATIVE time (within each sequence).
      - Vector: [B,S,D_in]
      - Image:  [B,S,C,H,W] -> per-frame CNN (ConvTrunk) -> [B,S,d_model]
    Returns:
      - tokens   [B,S',d_model]     (S'==S, or S'+1 if null-token appended for all-PAD)
      - g_s      [B,d_model]        (masked mean or CLS)
      - pad_mask [B,S']             (True=PAD)
    Notes:
      - REL time: Δt_{b,s} = t_ctx[b,s] - t_ref[b], where t_ref[b] = last valid time in the sequence
      - ABS & REL embeddings are concatenated: [token || e_abs || e_rel] -> Linear -> TransformerEncoder
      - z_seq=None and learnable_null=True: returns one learnable null token & null summary per batch.
    """
    def __init__(self, in_spec: Union[int, Tuple[int,int,int]], d_model: int,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.0,
                 pe_dim: int = 64,                       # abs/rel embedding dim each
                 pool: str = "mean",                      # 'mean' | 'cls'
                 t_abs_range: Optional[Tuple[float, float]] = None,
                 t_rel_range: Optional[Tuple[float, float]] = None,
                 learnable_null: bool = True):
        super().__init__()
        self.pool = pool.lower()
        assert self.pool in {"mean", "cls"}
        self.is_image = not isinstance(in_spec, int)
        self.learnable_null = bool(learnable_null)

        # --- token path ---
        if self.is_image:
            C, H, W = in_spec  # type: ignore
            self.frame_backbone = ConvTrunk(C, width=64, depth=3)
            D = self.frame_backbone.out_ch
            self.frame_proj = nn.Linear(D, d_model)
            token_dim = d_model
        else:
            self.frame_backbone = None
            self.frame_proj = None
            token_dim = int(in_spec)

        # --- ABS/REL time embeddings (continuous) ---
        self.time_abs_emb = ContinuousTimeEmbedding(out_dim=pe_dim, t_range=t_abs_range)
        self.time_rel_emb = ContinuousTimeEmbedding(out_dim=pe_dim, t_range=t_rel_range)

        # concat -> project to d_model
        in_proj_dim = token_dim + 2 * pe_dim
        self.input_proj = nn.Linear(in_proj_dim, d_model)
        nn.init.zeros_(self.input_proj.bias)

        # --- Transformer encoder (pre-LN) ---
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # --- CLS pooling option ---
        if self.pool == "cls":
            self.cls = nn.Parameter(th.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)

        # --- learnable null for missing/all-PAD ---
        if self.learnable_null:
            self.null_token = nn.Parameter(th.zeros(1, 1, d_model))
            self.null_summary = nn.Parameter(th.zeros(1, d_model))

    def _encode_frames(self, z_seq_img: th.Tensor) -> th.Tensor:
        B, S, C, H, W = z_seq_img.shape
        x = z_seq_img.view(B * S, C, H, W)
        f = self.frame_backbone(x)                      # [B*S, D, h, w]
        g = F.adaptive_avg_pool2d(f, 1).flatten(1)      # [B*S, D]
        v = self.frame_proj(g)                          # [B*S, d_model]
        return v.view(B, S, -1)                         # [B, S, d_model]

    def forward(self,
                z_seq: Optional[th.Tensor],
                seq_mask: Optional[th.Tensor] = None,   # [B,S?], True=PAD
                seq_lens: Optional[th.Tensor] = None,   # [B], optional
                t_ctx: Optional[th.Tensor] = None,      # [B,S] absolute times
                batch_size_hint: Optional[int] = None,  # used when z_seq is None
                ):
        # --- Case: no sequence provided ---
        if z_seq is None:
            if not self.learnable_null:
                return None, None, None
            if batch_size_hint is not None:
                B = int(batch_size_hint)
            elif t_ctx is not None:
                B = t_ctx.size(0)
            elif seq_mask is not None:
                B = seq_mask.size(0)
            else:
                B = 1
            tokens_out = self.null_token.expand(B, 1, -1).contiguous()
            g_s = self.null_summary.expand(B, -1).contiguous()
            pad_out = th.zeros(B, 1, dtype=th.bool, device=tokens_out.device)
            return tokens_out, g_s, pad_out

        if t_ctx is None:
            raise ValueError("SeqObsEncoder requires t_ctx[B,S] (absolute times).")

        device = z_seq.device

        # 1) tokenize
        if z_seq.dim() == 5:
            tokens = self._encode_frames(z_seq)        # [B,S,d_model]
        elif z_seq.dim() == 3:
            tokens = z_seq                             # [B,S,D_in]
        else:
            raise ValueError(f"Unsupported z_seq shape {tuple(z_seq.shape)}")

        B, S = tokens.shape[:2]

        # 2) pad mask
        if seq_mask is not None:
            if seq_mask.dtype != th.bool:
                seq_mask = (seq_mask != 0)
            pad_mask = seq_mask.to(device)
            if pad_mask.shape[1] != S:
                if pad_mask.shape[1] > S:
                    pad_mask = pad_mask[:, :S]
                else:
                    pad_mask = F.pad(pad_mask, (0, S - pad_mask.shape[1]), value=True)
        elif seq_lens is not None:
            arange = th.arange(S, device=device)[None, :].expand(B, S)
            pad_mask = arange >= seq_lens[:, None]
        else:
            pad_mask = th.zeros(B, S, dtype=th.bool, device=device)

        # 3) ABS/REL time embeddings
        e_abs = self.time_abs_emb(t_ctx.reshape(-1)).view(B, S, -1)  # [B,S,pe_dim]

        valid = (~pad_mask).long()
        lengths = valid.sum(dim=1).clamp_min(1)                      # [B]
        last_idx = (lengths - 1).unsqueeze(1)                        # [B,1]
        t_ref = th.gather(t_ctx, dim=1, index=last_idx).squeeze(1)   # [B]
        dt = (t_ctx - t_ref[:, None]).reshape(-1)                    # [B*S]
        e_rel = self.time_rel_emb(dt).view(B, S, -1)                 # [B,S,pe_dim]

        if pad_mask is not None:
            m = (~pad_mask)[..., None].to(e_abs.dtype)
            e_abs = e_abs * m
            e_rel = e_rel * m

        # 4) concat -> Linear
        x = th.cat([tokens, e_abs, e_rel], dim=-1)       # [B,S, token_dim + 2*pe_dim]
        x = self.input_proj(x)                           # [B,S,d_model]

        # 5) optional CLS
        if self.pool == "cls":
            cls = self.cls.expand(B, 1, -1)
            x = th.cat([cls, x], dim=1)                  # [B,S+1,d_model]
            pad_mask = th.cat([th.zeros(B, 1, dtype=th.bool, device=device), pad_mask], dim=1)

        # 6) Transformer encoder
        y = self.encoder(x, src_key_padding_mask=pad_mask)  # [B,S(d)+,d_model]
        y = self.norm(y)

        # 7) pooling
        if self.pool == "cls":
            g_s = y[:, 0, :]
            tokens_out = y[:, 1:, :]
            pad_out = pad_mask[:, 1:]
        else:
            valid_f = (~pad_mask).float()
            denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1e-8)
            g_s = (y * valid_f[..., None]).sum(dim=1) / denom
            tokens_out = y
            pad_out = pad_mask

        # 8) protect all-PAD with learnable null (so cross-attn always has something)
        if self.learnable_null:
            all_pad = pad_out.all(dim=1)  # [B]
            if all_pad.any():
                dummy_tok = self.null_token.expand(B, 1, -1).to(tokens_out.dtype).to(tokens_out.device)
                tokens_out = th.cat([tokens_out, dummy_tok], dim=1)  # [B,S+1,d_model]
                extra_pad = th.ones(B, 1, dtype=th.bool, device=pad_out.device)
                pad_out = th.cat([pad_out, extra_pad], dim=1)        # [B,S+1]
                pad_out[all_pad, -1] = False
                g_s = th.where(
                    all_pad[:, None],
                    self.null_summary.expand(B, -1).to(g_s.dtype).to(g_s.device),
                    g_s
                )

        return tokens_out, g_s, pad_out


# ------------------------------------------------------------
# 3) Cross-Attention (image Q) <- (tokens K,V)
# ------------------------------------------------------------
class CrossAttention2D(nn.Module):
    """
    Image feature map attends to tokens.
    Q: flattened image features (B, HW, C)
    K,V: tokens (B, S, D) projected to C
    """
    def __init__(self, in_channels: int, context_dim: int, n_heads: int = 8):
        super().__init__()
        assert in_channels % n_heads == 0 and (in_channels // n_heads) > 0, \
            "in_channels must be divisible by n_heads and head_dim>0"
        self.n_heads = n_heads
        self.head_dim = in_channels // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = normalization(in_channels)
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(context_dim, in_channels)
        self.v_proj = nn.Linear(context_dim, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self._context = None  # [B,S,D]
        self._pad_mask = None # [B,S] True=PAD

    def set_context(self, tokens: Optional[th.Tensor], pad_mask: Optional[th.Tensor]):
        self._context = tokens
        self._pad_mask = pad_mask

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, C, H, W = x.shape
        if self._context is None:
            return x
        tokens = self._context
        S = tokens.shape[1]
        if S == 0:
            return x

        h = self.norm(x)
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]

        q = self.q_proj(h)                               # [B, HW, C]
        k = self.k_proj(tokens)                          # [B, S, C]
        v = self.v_proj(tokens)                          # [B, S, C]

        # split heads
        q = q.view(B, H * W, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, HW, d]
        k = k.view(B, S,   self.n_heads, self.head_dim).permute(0, 2, 1, 3)    # [B, nh, S, d]
        v = v.view(B, S,   self.n_heads, self.head_dim).permute(0, 2, 1, 3)    # [B, nh, S, d]

        logits = th.matmul(q, k.transpose(-2, -1)) * self.scale                # [B, nh, HW, S]

        if self._pad_mask is not None:
            mask_bool = self._pad_mask[:, None, None, :].bool()                # [B,1,1,S]
            logits = logits.masked_fill(mask_bool, -1e9)

        # stabilize softmax
        maxv = logits.max(dim=-1, keepdim=True).values
        logits = logits - maxv
        attn = F.softmax(logits, dim=-1)

        if self._pad_mask is not None:
            valid = (~self._pad_mask)[:, None, None, :].to(attn.dtype)
            attn = attn * valid  # fully-masked rows -> zeros

        out = th.matmul(attn, v)                                              # [B, nh, HW, d]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)          # [B, HW, C]
        out = self.out_proj(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


# ------------------------------------------------------------
# 4) Conditional UNet with AdaGN (FiLM) + optional Cross-Attention
# ------------------------------------------------------------
class ConditionalUNetModel(nn.Module):
    """
    UNet backbone with:
      - global time embeddings: tau (diffusion) + t (physical/continuous)
      - Instant obs encoder E_inst (z_t -> c_t, with learnable null)
      - Sequential obs encoder E_seq (z_{1:S} -> tokens, g_s)  [ABS+REL time + learnable null]
      - FiLM (AdaGN) conditioning via h_cond = fuse([gamma_tau||gamma_t], c_t, g_s)
      - Cross-attention from image features (Q) to sequence tokens (K,V)
    """
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions,
        dropout: float = 0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        # === global continuous time (t) ===
        t_range: Optional[Tuple[float, float]] = None,
        t_max_freq: float = 1000.0,
        t_learnable_affine: bool = True,
        # === conditioning: inst & seq ===
        inst_obs_dim: Optional[Union[int, Tuple[int,int,int]]] = None,    # z_t shape spec (int or (C,H,W))
        seq_obs_dim: Optional[Union[int, Tuple[int,int,int]]] = None,     # each token spec
        seq_model_dim: Optional[int] = None,                               # d_model for E_seq
        seq_nheads: int = 8,
        seq_layers: int = 2,
        add_inst_token: bool = True,                                       # prepend c_t as token to K/V
        cross_attention_resolutions: Tuple[int, ...] = (8, 4),
        cross_attn_heads: int = 8,
        # === Seq time embedding ranges (inside SeqObsEncoder) ===
        seq_pe_dim: int = 64,
        seq_pool: str = "mean",                                            # 'mean' | 'cls'
        seq_t_abs_range: Optional[Tuple[float, float]] = None,
        seq_t_rel_range: Optional[Tuple[float, float]] = None,
        seq_learnable_null: bool = True,
        # === output head ===
        zero_out: bool = True,
    ):
        super().__init__()
        if dims != 2:
            raise NotImplementedError("This ConditionalUNetModel implementation is for 2D.")

        # base hparams
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.cross_attention_resolutions = set(cross_attention_resolutions)

        self.time_embed_dim = model_channels * 4

        # global time embeddings (t, tau)
        self.t_embed = ContinuousTimeEmbedding(
            out_dim=model_channels, t_range=t_range,
            max_freq=t_max_freq, learnable_affine=t_learnable_affine,
        )
        self.t_proj = MLP(in_dim=model_channels, hidden_dim=self.time_embed_dim, out_dim=self.time_embed_dim)

        self.tau_fourier = FourierEmbedding(out_dim=model_channels)
        self.tau_proj = MLP(in_dim=model_channels, hidden_dim=self.time_embed_dim, out_dim=self.time_embed_dim)

        self.time_fuse = MLP(in_dim=2 * self.time_embed_dim, hidden_dim=self.time_embed_dim, out_dim=self.time_embed_dim)

        # observational encoders
        self.inst_encoder = None
        self.inst_proj = None
        if inst_obs_dim is not None:
            self.inst_encoder = InstantObsEncoder(inst_obs_dim, out_dim=2 * model_channels)
            self.inst_proj = nn.Linear(2 * model_channels, self.time_embed_dim)

        self.seq_encoder = None
        self.add_inst_token = add_inst_token
        if seq_model_dim is None:
            seq_model_dim = 2 * model_channels
        self.seq_model_dim = seq_model_dim
        if seq_obs_dim is not None:
            self.seq_encoder = SeqObsEncoder(
                in_spec=seq_obs_dim, d_model=seq_model_dim,
                nhead=seq_nheads, num_layers=seq_layers, dropout=dropout,
                pe_dim=seq_pe_dim, pool=seq_pool,
                t_abs_range=seq_t_abs_range, t_rel_range=seq_t_rel_range,
                learnable_null=seq_learnable_null,
            )
            self.seq_proj = nn.Linear(seq_model_dim, self.time_embed_dim)

        # FiLM fuse
        self.cond_fuse = MLP(in_dim=self.time_embed_dim * 3, hidden_dim=self.time_embed_dim, out_dim=self.time_embed_dim)

        # UNet backbone with optional cross-attn
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        self._cross_modules = []

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, self.time_embed_dim, dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=True,
                    )
                ]
                ch = int(mult * model_channels)

                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads, num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                if ds in self.cross_attention_resolutions and self.seq_encoder is not None:
                    ca = CrossAttention2D(in_channels=ch, context_dim=self.seq_model_dim, n_heads=cross_attn_heads)
                    layers.append(ca)
                    self._cross_modules.append(ca)

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                down_layer = (
                    ResBlock(
                        ch, self.time_embed_dim, dropout,
                        out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=True, down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                self.input_blocks.append(TimestepEmbedSequential(down_layer))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # middle
        mid_layers = [
            ResBlock(
                ch, self.time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=True,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint,
                num_heads=num_heads, num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
        ]
        if self.seq_encoder is not None:
            ca_mid = CrossAttention2D(in_channels=ch, context_dim=self.seq_model_dim, n_heads=cross_attn_heads)
            mid_layers.append(ca_mid)
            self._cross_modules.append(ca_mid)
        mid_layers.append(
            ResBlock(
                ch, self.time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=True,
            )
        )
        self.middle_block = TimestepEmbedSequential(*mid_layers)
        self._feature_size += ch

        # up
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, self.time_embed_dim, dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=True,
                    )
                ]
                ch = int(model_channels * mult)

                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint,
                            num_heads=num_heads, num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                if ds in self.cross_attention_resolutions and self.seq_encoder is not None:
                    ca = CrossAttention2D(in_channels=ch, context_dim=self.seq_model_dim, n_heads=cross_attn_heads)
                    layers.append(ca)
                    self._cross_modules.append(ca)

                if level and i == num_res_blocks:
                    out_ch = ch
                    up_layer = (
                        ResBlock(
                            ch, self.time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=True, up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    layers.append(up_layer)
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # output head
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
        last: nn.Conv2d = self.out[-1]
        if zero_out:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        else:
            nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    # ------------------------- helpers -------------------------
    def _build_time_cond(self, timesteps: th.Tensor, tau: Optional[th.Tensor]) -> th.Tensor:
        t_emb = self.t_proj(self.t_embed(timesteps))  # [B, time_embed_dim]
        if tau is None:
            tau = th.zeros_like(timesteps, dtype=th.float32)
        tau = tau.to(t_emb.dtype)
        tau_emb = self.tau_proj(self.tau_fourier(tau))  # [B, time_embed_dim]
        return self.time_fuse(th.cat([tau_emb, t_emb], dim=1))

    def _build_obs_cond(
        self,
        batch_size: int,
        z_t: Optional[th.Tensor],
        z_seq: Optional[th.Tensor],
        seq_mask: Optional[th.Tensor],
        seq_lens: Optional[th.Tensor],
        t_ctx: Optional[th.Tensor],
    ):
        # instant (use learnable NULL when z_t is None)
        if self.inst_encoder is not None:
            c_t = self.inst_encoder(z_t, batch_size=batch_size)  # [B, 2C]
            c_t_proj = self.inst_proj(c_t) if (self.inst_proj is not None) else None
        else:
            c_t, c_t_proj = None, None

        # sequential (call even if z_seq is None -> learnable null path if enabled)
        if self.seq_encoder is not None:
            tokens, g_s, pad_mask = self.seq_encoder(
                z_seq, seq_mask=seq_mask, seq_lens=seq_lens, t_ctx=t_ctx,
                batch_size_hint=batch_size
            )  # tokens:[B,S',d_seq], g_s:[B,d_seq], pad:[B,S']
            g_s_proj = self.seq_proj(g_s) if (g_s is not None) else None
        else:
            tokens, g_s_proj, pad_mask = None, None, None

        return c_t, c_t_proj, tokens, g_s_proj, pad_mask

    def _fuse_film(self, time_part: th.Tensor, c_t_proj: Optional[th.Tensor], g_s_proj: Optional[th.Tensor]) -> th.Tensor:
        zeros = th.zeros_like(time_part)
        if c_t_proj is None: c_t_proj = zeros
        if g_s_proj is None: g_s_proj = zeros
        return self.cond_fuse(th.cat([time_part, c_t_proj, g_s_proj], dim=1))

    def _prepare_tokens_for_cross(self, tokens: Optional[th.Tensor], c_t: Optional[th.Tensor], pad_mask: Optional[th.Tensor]):
        if tokens is None:
            return None, None

        B, S, D = tokens.shape
        dev, dtp = tokens.device, tokens.dtype

        if pad_mask is None:
            pad_mask = th.zeros(B, S, dtype=th.bool, device=dev)
        else:
            pad_mask = pad_mask.to(dev)
            if pad_mask.dtype != th.bool:
                pad_mask = (pad_mask != 0)
            if pad_mask.shape[1] != S:
                if pad_mask.shape[1] > S:
                    pad_mask = pad_mask[:, :S]
                else:
                    pad_mask = F.pad(pad_mask, (0, S - pad_mask.shape[1]), value=True)

        # optionally prepend instant token
        if self.add_inst_token:
            if c_t is None:
                c_tok = th.zeros(B, 1, D, device=dev, dtype=dtp)
            else:
                c_tok = c_t[:, None, :].to(device=dev, dtype=dtp)
            tokens = th.cat([c_tok, tokens], dim=1)
            first = th.zeros(B, 1, device=dev, dtype=th.bool)
            pad_mask = th.cat([first, pad_mask], dim=1)

        # ensure not all PAD
        all_pad = pad_mask.all(dim=1, keepdim=True)
        if all_pad.any():
            dummy = th.zeros(B, 1, D, device=dev, dtype=dtp)
            tokens = th.cat([tokens, dummy], dim=1)
            false_col = th.zeros(B, 1, device=dev, dtype=th.bool)
            pad_mask = th.cat([pad_mask, false_col], dim=1)

        # scrub non-finite
        nonfinite = ~th.isfinite(tokens)
        if nonfinite.any():
            tokens = tokens.masked_fill(nonfinite, 0.0)

        return tokens, pad_mask

    def _set_cross_context(self, tokens: Optional[th.Tensor], pad_mask: Optional[th.Tensor]):
        for m in self._cross_modules:
            m.set_context(tokens, pad_mask)

    # --------------------------- forward ---------------------------
    def forward(
        self,
        x: th.Tensor,                                  # [B, in_channels, H, W]
        timesteps: th.Tensor,                          # [B] physical time t
        tau: Optional[th.Tensor] = None,               # [B] diffusion time in [0,1]
        z_t: Optional[th.Tensor] = None,               # [B, ...] instant obs
        z_seq: Optional[th.Tensor] = None,             # [B, S, ...] sequential obs
        seq_mask: Optional[th.Tensor] = None,          # [B, S] True=PAD
        seq_lens: Optional[th.Tensor] = None,          # [B] optional
        t_ctx: Optional[th.Tensor] = None,             # [B, S] absolute times for tokens
    ) -> th.Tensor:
        B = x.shape[0]

        # time/obs cond
        time_part = self._build_time_cond(timesteps, tau)
        c_t, c_t_proj, tokens, g_s_proj, pad_mask = self._build_obs_cond(
            batch_size=B,
            z_t=z_t,
            z_seq=z_seq,
            seq_mask=seq_mask,
            seq_lens=seq_lens,
            t_ctx=t_ctx,
        )
        emb = self._fuse_film(time_part, c_t_proj, g_s_proj)

        # cross-attn context
        tokens, pad_mask = self._prepare_tokens_for_cross(tokens, c_t, pad_mask)
        self._set_cross_context(tokens, pad_mask)

        # UNet forward
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)