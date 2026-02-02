# models/tft.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class GLU(nn.Module):
    """
    Gated Linear Unit:
        y = (Wc x + bc) ⊙ σ(Wg x + bg)
    Optional residual+LayerNorm wrapper:
        y = LayerNorm( residual + Dropout(GLU(x)) )

    Args:
        input_dim:  size of last dim in x
        output_dim: size of output features
        dropout:    dropout applied to gated output
        use_residual: add residual path if input_dim == output_dim
        layernorm:  apply layernorm on (residual + gated_out)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_residual: bool = True,
        layernorm: bool = True,
    ):
        super().__init__()
        self.fc_content = nn.Linear(input_dim, output_dim)
        self.fc_gate    = nn.Linear(input_dim, output_dim)
        self.sigmoid    = nn.Sigmoid()
        self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_residual = use_residual and (input_dim == output_dim)
        self.ln = nn.LayerNorm(output_dim) if layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., input_dim)
        return: (..., output_dim)
        """
        content = self.fc_content(x)               # (..., output_dim)
        gate    = self.sigmoid(self.fc_gate(x))    # (..., output_dim)
        gated   = content * gate                   # (..., output_dim)
        gated   = self.dropout(gated)

        if self.use_residual:
            out = self.ln(gated + x)              # residual only if dims match
        else:
            out = self.ln(gated)
        return out
    
class GRN(nn.Module):
    """
    Gated Residual Network (GRN)

    Core flow (TFT-style):
      h1 = ELU( W_x x  +  W_c c  +  b1 )          # context c optional
      h2 = Dropout( W_h h1 + b2 )                  # project to output dim
      g  = GLU(h2)                                 # gated linear unit
      y  = LayerNorm( g + skip(x) )                # residual + norm

    Args:
        input_dim   : dimension of x
        hidden_dim  : internal hidden size
        output_dim  : output size
        context_dim : dimension of optional context c (set None if not used)
        dropout     : dropout prob on pre-GLU projection
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        # x and optional context go to hidden
        self.fc_x = nn.Linear(input_dim, hidden_dim)
        self.fc_c = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        self.elu  = nn.ELU()

        # hidden -> output_dim (pre-gate)
        self.fc_h = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # gate over output_dim
        self.glu  = GLU(output_dim, output_dim, dropout=dropout, use_residual=False, layernorm=False)

        # residual path (project if dims differ)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

        # final normalization
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:        (..., input_dim)
        context:  (..., context_dim) or None
        return:   (..., output_dim)
        """
        h = self.fc_x(x)
        if self.fc_c is not None:
            if context is None:
                raise ValueError("GRN expected a context tensor but got None.")
            h = h + self.fc_c(context)

        h = self.elu(h)
        h = self.drop(self.fc_h(h))          # (..., output_dim)

        g = self.glu(h)                      # gated output (..., output_dim)
        y = self.ln(g + self.skip(x))        # residual + norm
        return y

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN)

    For a set of K input variables (each with d_in features), the VSN:
      1) Transforms each variable i with a GRN_i(x_i, context) -> e_i in R^{d_model}
      2) Scores each variable with a gating GRN'_i(x_i, context) -> g_i in R
      3) Applies softmax over g_i to get weights w_i, then fuses:
            y = sum_i w_i * e_i

    Shapes
    ------
      x:        (..., K, d_in)
      context:  (..., d_ctx) or None     # optional static/temporal context
      mask:     (..., K) bool or 0/1     # optional, True keeps, False masks out
      return:
        y:        (..., d_model)         # fused representation
        weights:  (..., K)               # per-variable softmax weights
        embeds:   (..., K, d_model)      # per-variable transformed embeddings

    Args
    ----
      n_vars:       K (number of variables)
      input_dim:    d_in (per-variable input size)
      d_model:      output embedding size for each variable before fusion
      hidden_dim:   GRN hidden size
      context_dim:  optional context size (None to disable)
      dropout:      dropout prob inside GRNs (and GLUs within them)
    """
    def __init__(
        self,
        n_vars: int,
        input_dim: int,
        d_model: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.input_dim = input_dim
        self.d_model = d_model

        # Per-variable transform GRNs: x_i -> e_i (d_model)
        self.var_transform = nn.ModuleList([
            GRN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=d_model,
                context_dim=context_dim,
                dropout=dropout,
            )
            for _ in range(n_vars)
        ])

        # Per-variable gating GRNs: x_i -> scalar logit
        # We produce a 1-d output and later squeeze to (...,)
        self.var_gates = nn.ModuleList([
            GRN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                context_dim=context_dim,
                dropout=dropout,
            )
            for _ in range(n_vars)
        ])

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,                      # (..., K, d_in)
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,  # (..., K) True=keep, False=mask
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.shape[-2] == self.n_vars and x.shape[-1] == self.input_dim, \
            f"Expected x shape (..., {self.n_vars}, {self.input_dim}), got {tuple(x.shape)}"

        xs = [x[..., i, :] for i in range(self.n_vars)]

        embeds, logits = [], []
        for i in range(self.n_vars):
            ei = self.var_transform[i](xs[i], context)        # (..., d_model)
            gi = self.var_gates[i](xs[i], context)[..., 0]    # (...,)
            embeds.append(ei)
            logits.append(gi)

        embeds = torch.stack(embeds, dim=-2)   # (..., K, d_model)
        logits = torch.stack(logits, dim=-1)   # (..., K)

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            # very negative to zero-out after softmax (safer than finfo.min)
            neg_big = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits = torch.where(mask, logits, neg_big)

            # guard: if all masked at some position → fallback to uniform over all vars
            all_masked = ~mask.any(dim=-1, keepdim=True)      # (..., 1)
            if all_masked.any():
                # replace logits with zeros so softmax -> uniform
                logits = torch.where(all_masked, torch.zeros_like(logits), logits)

        weights = self.softmax(logits)          # (..., K)
        y = (weights.unsqueeze(-1) * embeds).sum(dim=-2)  # (..., d_model)
        return y, weights, embeds
    
class StaticContextEncoder(nn.Module):
    """
    Static context encoder for TFT.

    Inputs
    ------
    x_static : (B, K_s, d_in)    # K_s static variables, each embedded to d_in
    mask_s   : (B, K_s) or None  # optional availability mask for static vars

    Outputs
    -------
    s          : (B, d_model)    # fused static summary (via static VSN)
    w_static   : (B, K_s)        # static variable selection weights
    h0, c0     : (1, B, d_lstm)  # initial LSTM hidden/cell states
    ctx_seq    : (B, d_model)    # context for temporal variable GRNs/VSNs
    ctx_attn   : (B, d_model)    # context for attention blocks
    """

    def __init__(
        self,
        n_static_vars: int,
        input_dim: int,
        d_model: int,
        hidden_dim: int,
        d_lstm: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 1) Static variable selection (no external context yet)
        self.static_vsn = VariableSelectionNetwork(
            n_vars=n_static_vars,
            input_dim=input_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            context_dim=None,         # no context to begin with
            dropout=dropout,
        )

        # 2) Project selected static summary s → multiple contextual heads
        #    GRNs keep gating/residual behavior and stabilize with LayerNorm
        self.ctx_seq_grn   = GRN(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)
        self.ctx_attn_grn  = GRN(d_model, hidden_dim, d_model, context_dim=None, dropout=dropout)

        # 3) Initialize LSTM states from s (paper does this to inform sequence encoder/decoder)
        self.h0_proj = nn.Linear(d_model, d_lstm)
        self.c0_proj = nn.Linear(d_model, d_lstm)

    def forward(
        self,
        x_static: torch.Tensor,                 # (B, K_s, d_in)
        mask_s: Optional[torch.Tensor] = None,  # (B, K_s) optional
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Static variable selection → fused summary s and weights
        s, w_static, _ = self.static_vsn(x_static, context=None, mask=mask_s)  # s: (B, d_model)

        # Context vectors produced from s
        ctx_seq  = self.ctx_seq_grn(s)    # (B, d_model)  conditions temporal VSN/GRNs
        ctx_attn = self.ctx_attn_grn(s)   # (B, d_model)  conditions attention layers

        # LSTM initial states (shape (num_layers=1, B, d_lstm))
        h0 = self.h0_proj(s).unsqueeze(0)  # (1, B, d_lstm)
        c0 = self.c0_proj(s).unsqueeze(0)  # (1, B, d_lstm)

        return s, w_static, h0, c0, ctx_seq, ctx_attn

# ─────────────────────────────────────────────────────────
# LSTM encoder/decoder with static conditioning
# ─────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Stacked LSTM encoder over past window.
    Optionally conditions initial (h0, c0) on static context via linear maps.

    Inputs
    ------
      x_enc:  (B, T_enc, d_model)  encoder time-varying embeddings
      s_ctx:  (B, d_static) or None  static context embedding

    Returns
    -------
      H:      (B, T_enc, hidden_size)  all hidden states
      (hN,cN): final states tuples for the top LSTM layer, each (num_layers, B, hidden_size)
    """
    def __init__(
        self,
        d_model: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        static_dim: Optional[int] = None,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # project static -> initial states of top layer (h0, c0)
        self.static_dim = static_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if static_dim is not None:
            self.h0_proj = nn.Linear(static_dim, num_layers * hidden_size)
            self.c0_proj = nn.Linear(static_dim, num_layers * hidden_size)
        else:
            self.h0_proj = self.c0_proj = None

    def _init_states(self, B: int, s_ctx: Optional[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.static_dim is None or s_ctx is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            return h0, c0
        # shape (B, num_layers*hidden)
        h0_flat = self.h0_proj(s_ctx)  # (B, num_layers*hidden)
        c0_flat = self.c0_proj(s_ctx)  # (B, num_layers*hidden)
        # reshape to (num_layers, B, hidden)
        h0 = h0_flat.view(B, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()
        c0 = c0_flat.view(B, self.num_layers, self.hidden_size).transpose(0, 1).contiguous()
        return h0, c0

    def forward(self, x_enc: torch.Tensor, s_ctx: Optional[torch.Tensor] = None):
        B, T, _ = x_enc.shape
        device = x_enc.device
        h0, c0 = self._init_states(B, s_ctx, device)
        H, (hN, cN) = self.lstm(x_enc, (h0, c0))   # H: (B,T,hidden)
        return H, (hN, cN)


class LSTMDecoder(nn.Module):
    """
    Autoregressive LSTM decoder over future window.
    Consumes known-future embeddings and (optionally) previous target.
    Can be trained with teacher forcing.

    Inputs
    ------
      x_dec:     (B, T_dec, d_model)   known future embeddings per step
      hN,cN:     (num_layers,B,H)      initial states from encoder (or from static)
      y_prev:    (B, T_dec, d_y) or None  previous targets for teacher forcing
      tf_ratio:  float in [0,1]        prob of using ground truth at each step
      proj_in:   optional nn.Linear to combine [x_dec_t, y_{t-1}] -> decoder input dim

    Returns
    -------
      H_dec:   (B, T_dec, H)    decoder hidden states
      (hT,cT): final states
    """
    def __init__(
        self,
        d_model: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        y_dim: int = 1,
        use_prev_target: bool = True,
    ):
        super().__init__()
        self.use_prev_target = use_prev_target
        self.y_dim = y_dim  # <-- store this
        in_dim = d_model + (y_dim if use_prev_target else 0)

        self.input_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self,
        x_dec: torch.Tensor,                     # (B,T_dec,d_model)
        hN: torch.Tensor, cN: torch.Tensor,      # encoder final states
        y_prev: Optional[torch.Tensor] = None,   # (B,T_dec,y_dim)
        tf_ratio: float = 1.0,
        y_start: Optional[torch.Tensor] = None,  # (B,y_dim)
    ):
        B, T, _ = x_dec.shape
        device = x_dec.device
        H_list = []
        h_t, c_t = hN, cN

        # First previous target
        if self.use_prev_target:
            if y_prev is not None:
                prev_y_t = y_prev[:, 0, :]                          # (B, y_dim)
            elif y_start is not None:
                prev_y_t = y_start                                   # (B, y_dim)
            else:
                prev_y_t = torch.zeros(B, self.y_dim, device=device) # (B, y_dim)
        else:
            prev_y_t = None

        for t in range(T):
            x_t = x_dec[:, t, :]                          # (B, d_model)
            if self.use_prev_target:
                dec_in = torch.cat([x_t, prev_y_t], dim=-1)  # (B, d_model+y_dim)
                dec_in = self.input_proj(dec_in)
            else:
                dec_in = x_t

            dec_in = dec_in.unsqueeze(1)                  # (B,1,d_model)
            H_t, (h_t, c_t) = self.lstm(dec_in, (h_t, c_t))   # H_t: (B,1,H)
            H_list.append(H_t)

            if self.use_prev_target:
                if (y_prev is not None) and (torch.rand(1).item() < tf_ratio) and (t + 1 < T):
                    prev_y_t = y_prev[:, t+1, :]          # teacher forcing
                else:
                    prev_y_t = torch.zeros_like(prev_y_t)

        H_dec = torch.cat(H_list, dim=1)  # (B,T_dec,H)
        return H_dec, (h_t, c_t)



class MultiHeadAttentionWithContext(nn.Module):
    """
    Multi-head attention block with static/temporal context for TFT.

    - Uses a GRN to condition queries on a context vector.
    - Uses PyTorch's nn.MultiheadAttention under the hood.
    - Wraps attention output with GLU + residual + LayerNorm.

    Inputs
    ------
      x:              (B, T_q, d_model)   query sequence (e.g., decoder states)
      memory:         (B, T_kv, d_model)  key/value sequence (e.g., enc+dec), or None for self-attention
      context:        (B, d_ctx) or None  static/temporal context for query GRN
      attn_mask:      (T_q, T_kv) or None; True / -inf where attention is blocked
      key_padding_mask: (B, T_kv) or None; True where positions should be masked

    Returns
    -------
      out:            (B, T_q, d_model)   gated + residual + normalized output
      attn_weights:   (B, T_q, T_kv)      attention weights per head averaged
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
        grn_hidden_dim: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal

        # GRN to condition queries on context (if provided)
        if context_dim is not None:
            if grn_hidden_dim is None:
                grn_hidden_dim = d_model
            self.query_grn = GRN(
                input_dim=d_model,
                hidden_dim=grn_hidden_dim,
                output_dim=d_model,
                context_dim=context_dim,
                dropout=dropout,
            )
        else:
            self.query_grn = None

        # Core multi-head attention (batch_first=True => (B,T,C))
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Gated output + residual + LayerNorm
        self.glu = GLU(
            input_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
            use_residual=False,
            layernorm=False,
        )
        self.ln = nn.LayerNorm(d_model)

    def _causal_mask(self, T_q: int, T_k: int, device) -> torch.Tensor:
        """
        Build a standard causal mask (prevent attending to future steps).
        Shape: (T_q, T_k), True where attention should be blocked.
        """
        # positions (i,j) with j > i should be masked
        mask = torch.triu(torch.ones(T_q, T_k, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        x: torch.Tensor,                       # (B,T_q,d_model)
        memory: Optional[torch.Tensor] = None, # (B,T_kv,d_model) or None
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,       # (T_q,T_kv) or None
        key_padding_mask: Optional[torch.Tensor] = None # (B,T_kv) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = x.shape
        device = x.device

        # 1) Condition queries on context via GRN 
        if self.query_grn is not None:
            if context is None:
                raise ValueError("MultiHeadAttentionWithContext expected a context tensor but got None.")
            # query_grn applied per time step with same context
            q_in = self.query_grn(x, context=context)   # (B,T_q,d_model)
        else:
            q_in = x                                    # (B,T_q,d_model)

        # 2) Keys/values come from memory (cross-attn) or x (self-attn)
        if memory is None:
            k_in = q_in
            v_in = q_in
            T_k = T_q
        else:
            k_in = memory
            v_in = memory
            T_k = memory.size(1)

        # 3) Build causal mask if requested and not provided, for self-attention
        if self.causal and attn_mask is None and memory is None:
            # nn.MultiheadAttention expects bool or float mask of shape (T_q, T_k)
            attn_mask = self._causal_mask(T_q, T_k, device=device)  # True = block

        # 4) Multi-head attention
        # attn_weights: (B, T_q, T_k) when batch_first=True
        attn_out, attn_weights = self.mha(
            query=q_in,
            key=k_in,
            value=v_in,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attn_out = self.dropout(attn_out)        # (B,T_q,d_model)

        # 5) GLU + residual + LayerNorm
        gated = self.glu(attn_out)              # (B,T_q,d_model)
        out = self.ln(gated + x)                # residual uses original x

        return out, attn_weights


class QuantileOutputLayer(nn.Module):
    """
    Quantile output head for TFT.

    Maps hidden states h_t ∈ R^{d_model} to a set of quantile predictions
    for each target dimension.

    Inputs
    ------
      h: (B, T, d_model)    hidden states (e.g., from decoder-attention stack)

    Outputs
    -------
      y_hat: (B, T, D, Q)
        - D = target_dim (e.g., 1 for a single load series)
        - Q = number of quantiles (e.g., 3 for [0.1, 0.5, 0.9])
    """
    def __init__(
        self,
        d_model: int,
        target_dim: int,
        quantiles: List[float],
    ):
        super().__init__()
        self.d_model = d_model
        self.target_dim = target_dim
        self.quantiles = quantiles
        self.n_q = len(quantiles)

        # Linear map: d_model → (target_dim * n_q)
        self.proj = nn.Linear(d_model, target_dim * self.n_q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, d_model)
        returns: (B, T, target_dim, n_q)
        """
        B, T, _ = h.shape
        out = self.proj(h)                             # (B, T, D * Q)
        out = out.view(B, T, self.target_dim, self.n_q)
        return out


def quantile_loss(
    y_true: torch.Tensor,       # (B, T, D)
    y_pred: torch.Tensor,       # (B, T, D, Q)
    quantiles: List[float],
    mask: Optional[torch.Tensor] = None,  # (B, T, D) or None
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pinball loss for multiple quantiles.

    For each quantile q and error u = y_true - y_pred_q:
        L_q(u) = max( q * u, (q - 1) * u )

    Shapes
    ------
      y_true : (B, T, D)
      y_pred : (B, T, D, Q)
      mask   : optional 0/1 or bool mask (B, T, D). False/0 → ignore in loss.

    Returns
    -------
      scalar loss (if reduction="mean" or "sum") or per-element loss tensor.
    """
    assert y_pred.shape[:-1] == y_true.shape, "y_pred and y_true shapes must align on (B,T,D)"
    Q = y_pred.shape[-1]
    assert Q == len(quantiles), "Last dimension of y_pred must match len(quantiles)"

    # (1,1,1,Q)
    q_tensor = torch.tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device).view(
        1, 1, 1, Q
    )

    # u = y_true - y_pred_q
    #  y_true: (B,T,D) → (B,T,D,1)
    #  y_pred: (B,T,D,Q)
    u = y_true.unsqueeze(-1) - y_pred           # (B,T,D,Q)

    # pinball loss per quantile
    # if u >= 0 → q * u
    # if u <  0 → (q - 1) * u
    loss = torch.where(u >= 0, q_tensor * u, (q_tensor - 1.0) * u)  # (B,T,D,Q)

    # apply mask if provided
    if mask is not None:
        # broadcast mask to (B,T,D,Q)
        m = mask.to(dtype=loss.dtype, device=loss.device).unsqueeze(-1)
        loss = loss * m

        valid_count = m.sum()
        if reduction == "mean" and valid_count > 0:
            return loss.sum() / valid_count
        elif reduction == "mean" and valid_count == 0:
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        # no reduction
        return loss
    

class TFTNoStatic(nn.Module):
    """
    TFT-style model without static features.
    - Separate VSNs for encoder vs decoder inputs
    - LSTM encoder/decoder
    - Stepwise attention over [encoder history | decoder prefix]
    - Quantile outputs + autoregressive feedback using median quantile
    """

    def __init__(
        self,
        n_enc_vars: int,
        n_dec_vars: int,
        d_model: int = 64,
        hidden_grn: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        quantiles: List[float] = (0.1, 0.5, 0.9),
        y_dim: int = 1,
    ):
        super().__init__()
        self.enc_grn = GRN(d_model, hidden_grn, d_model, context_dim=None, dropout=dropout)
        self.dec_grn = GRN(d_model, hidden_grn, d_model, context_dim=None, dropout=dropout)
        self.n_enc_vars = n_enc_vars
        self.n_dec_vars = n_dec_vars
        self.d_model = d_model
        self.quantiles = list(quantiles)
        self.y_dim = y_dim

        # Encoder-side VSN (observed/past variables)
        self.vsn_enc = VariableSelectionNetwork(
            n_vars=n_enc_vars,
            input_dim=1,
            d_model=d_model,
            hidden_dim=hidden_grn,
            context_dim=None,
            dropout=dropout,
        )

        # Decoder-side VSN (known-future variables)
        self.vsn_dec = VariableSelectionNetwork(
            n_vars=n_dec_vars,
            input_dim=1,
            d_model=d_model,
            hidden_dim=hidden_grn,
            context_dim=None,
            dropout=dropout,
        )

        # LSTMs (use hidden_size=d_model so shapes line up)
        self.encoder = LSTMEncoder(
            d_model=d_model,
            hidden_size=d_model,
            num_layers=1,
            dropout=dropout,
            static_dim=None,
        )

        self.decoder = LSTMDecoder(
            d_model=d_model,
            hidden_size=d_model,
            num_layers=1,
            dropout=dropout,
            y_dim=y_dim,
            use_prev_target=True,
        )

        # Attention: query is 1-step decoder state, memory is [enc | dec_prefix]
        self.attn = MultiHeadAttentionWithContext(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            context_dim=None,
            causal=False,  # we control causality by only giving prefix memory
        )

        self.head = QuantileOutputLayer(
            d_model=d_model,
            target_dim=y_dim,
            quantiles=self.quantiles,
        )

        self.post_attn_grn = GRN(
            input_dim=d_model,
            hidden_dim=hidden_grn,
            output_dim=d_model,
            context_dim=None,
            dropout=dropout,
        )

    def _vsn_timewise(self, x: torch.Tensor, vsn: VariableSelectionNetwork, n_vars: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, K) numeric scalars per variable
        returns:
          z: (B, T, d_model)
          w: (B, T, K) variable selection weights
        """
        B, T, K = x.shape
        assert K == n_vars, f"Expected K={n_vars}, got {K}"

        x_flat = x.unsqueeze(-1).view(B * T, K, 1)         # (B*T, K, 1)
        z_flat, w_flat, _ = vsn(x_flat, context=None)      # z: (B*T,d_model), w: (B*T,K)
        z = z_flat.view(B, T, self.d_model)
        w = w_flat.view(B, T, K)
        return z, w

    def forward(
        self,
        x_enc: torch.Tensor,          # (B, T_enc, K_enc)
        x_dec: torch.Tensor,          # (B, T_dec, K_dec)
        y_dec: Optional[torch.Tensor] = None,  # (B, T_dec, y_dim) for teacher forcing
        tf_ratio: float = 1.0,
        free_run: bool = False,       # if True, always feed predictions back (eval)
    ):
        B, T_enc, _ = x_enc.shape
        _, T_dec, _ = x_dec.shape
        device = x_enc.device

        # 1) Variable selection
        z_enc, w_enc = self._vsn_timewise(x_enc, self.vsn_enc, self.n_enc_vars)  # (B,T_enc,d_model)
        z_dec, w_dec = self._vsn_timewise(x_dec, self.vsn_dec, self.n_dec_vars)  # (B,T_dec,d_model)

        # 2) Encode history
        H_enc, (hN, cN) = self.encoder(z_enc, s_ctx=None)
        H_enc = self.enc_grn(H_enc)  # (B,T_enc,d_model)


        # 3) Stepwise decode + attention + quantile output
        h_t, c_t = hN, cN
        dec_prefix = []      # list of (B,1,d_model)
        y_hat_steps = []     # list of (B,1,y_dim,Q)
        attn_steps = []      # list of (B,1,T_mem)

        # start prev_y
        if free_run or (y_dec is None):
            prev_y = torch.zeros(B, self.y_dim, device=device)
        else:
            prev_y = y_dec[:, 0, :]

        # median quantile index (used for feedback)
        q_median = 0.5
        if q_median in self.quantiles:
            q_med_idx = self.quantiles.index(q_median)
        else:
            # fallback: closest to 0.5
            q_med_idx = int(torch.argmin(torch.tensor([abs(q - 0.5) for q in self.quantiles])).item())

        for t in range(T_dec):
            x_t = z_dec[:, t, :]  # (B,d_model)

            # --- one LSTM decoder step (teacher forcing input handled via prev_y) ---
            dec_in = torch.cat([x_t, prev_y], dim=-1)            # (B, d_model+y_dim)
            dec_in = self.decoder.input_proj(dec_in).unsqueeze(1) # (B,1,d_model)
            H_t, (h_t, c_t) = self.decoder.lstm(dec_in, (h_t, c_t))
            H_t = self.dec_grn(H_t) # (B,1,d_model)
            dec_prefix.append(H_t)

            # memory is encoder history + decoder prefix so far
            H_dec_prefix = torch.cat(dec_prefix, dim=1)          # (B, t+1, d_model)
            memory = torch.cat([H_enc, H_dec_prefix], dim=1)     # (B, T_enc+t+1, d_model)

            # attention: query is current step only
            attn_out, attn_w = self.attn(
                x=H_t,        # (B,1,d_model)
                memory=memory # (B,T_mem,d_model)
            )                # attn_out: (B,1,d_model), attn_w: (B,1,T_mem)

            # post-attention gating + output head
            attn_out = self.post_attn_grn(attn_out)             # (B,1,d_model)
            y_hat_t = self.head(attn_out)                       # (B,1,y_dim,Q)

            y_hat_steps.append(y_hat_t)
            attn_steps.append(attn_w)

            # --- choose next prev_y (teacher forcing vs free-running) ---
            use_tf = (
                (not free_run)
                and (y_dec is not None)
                and (t + 1 < T_dec)
                and (tf_ratio > 0)
                and (torch.rand(1).item() < tf_ratio)
            )

            if use_tf:
                prev_y = y_dec[:, t+1, :]
            else:
                # feed back median quantile
                prev_y = y_hat_t[:, 0, :, q_med_idx].detach()   # (B,y_dim)

        y_hat = torch.cat(y_hat_steps, dim=1)   # (B,T_dec,y_dim,Q)
        attn_w = attn_steps  # list of (B,1,T_mem) tensors, varying T_mem

        return y_hat, {"w_enc": w_enc, "w_dec": w_dec, "attn": attn_steps}



