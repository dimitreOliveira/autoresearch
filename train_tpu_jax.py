"""
Autoresearch pretraining script. Single-TPU, single-file. (JAX version)
Usage: uv run train_tpu_jax.py
"""

import functools
import gc
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from prepare import (
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    get_token_bytes,
    make_dataloader,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)  # type: ignore
sys.stderr.reconfigure(line_buffering=True)  # type: ignore

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

jax.config.update("jax_default_matmul_precision", "high")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**19
BASE_LR = 0.004
WEIGHT_DECAY = 0.05
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.1

SEQUENCE_LEN = 2048
DEPTH = 8
DEVICE_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
DTYPE = jnp.bfloat16

# TPU hardware detection
PEAK_FLOPS = 197.0e12  # Default TPU v5e
try:
    tpu_env = os.environ.get("TPU_NAME", "")
    tpu_accel_type = os.environ.get("TPU_ACCELERATOR_TYPE", "")
    if "v6e" in tpu_accel_type or "v6e" in tpu_env:
        PEAK_FLOPS = 918.0e12
        print("Detected TPU v6e (Trillium)")
    elif "v5e" in tpu_accel_type or "v5e" in tpu_env:
        PEAK_FLOPS = 197.0e12
        print("Detected TPU v5e")
    elif "v2" in tpu_accel_type or "v2" in tpu_env:
        PEAK_FLOPS = 45.0e12
        print("Detected TPU v2")
    else:
        print(
            f"Warning: Unknown TPU type '{tpu_accel_type}', defaulting to TPU v5e peak flops."
        )
except Exception:
    pass

# ---------------------------------------------------------------------------
# GPT Model (Flax)
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_window_sizes(config):
    pattern = config.window_pattern.upper()
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": long_window, "S": short_window}
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = long_window
    return window_sizes


def rms_norm(x, eps=1e-5):
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps)


def init_wte(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype) * 1.0


def init_lm_head(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype) * 0.001


def init_linear(key, shape, dtype=jnp.float32):
    fan_in = shape[0]
    s = (3.0**0.5) * (fan_in**-0.5)
    return jax.random.uniform(key, shape, minval=-s, maxval=s, dtype=dtype)


def precompute_rotary_embeddings(seq_len, head_dim, base=10000.0):
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(DTYPE)
    sin = jnp.sin(freqs).astype(DTYPE)
    # Broadcast to (1, T, 1, head_dim/2)
    return cos[None, :, None, :], sin[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    # x: (B, T, H, head_dim)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


class CausalSelfAttention(nn.Module):
    config: GPTConfig
    layer_idx: int
    window_size: int

    @nn.compact
    def __call__(self, x, ve_in, cos_sin):
        B, T, C = x.shape
        head_dim = self.config.n_embd // self.config.n_head

        q = nn.Dense(
            self.config.n_head * head_dim,
            use_bias=False,
            kernel_init=init_linear,
            name="c_q",
        )(x)
        k = nn.Dense(
            self.config.n_kv_head * head_dim,
            use_bias=False,
            kernel_init=init_linear,
            name="c_k",
        )(x)
        v = nn.Dense(
            self.config.n_kv_head * head_dim,
            use_bias=False,
            kernel_init=init_linear,
            name="c_v",
        )(x)

        q = q.reshape((B, T, self.config.n_head, head_dim))
        k = k.reshape((B, T, self.config.n_kv_head, head_dim))
        v = v.reshape((B, T, self.config.n_kv_head, head_dim))

        if ve_in is not None:
            v_dtype = v.dtype
            ve = ve_in.reshape((B, T, self.config.n_kv_head, head_dim))
            ve_gate_channels = 32
            x_ve = x[..., :ve_gate_channels]
            ve_gate = nn.Dense(
                self.config.n_kv_head,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="ve_gate",
            )(x_ve)
            gate = 2.0 * nn.sigmoid(ve_gate)
            v = (v + gate[..., None] * ve).astype(v_dtype)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = rms_norm(q)
        k = rms_norm(k)

        scale = 1.0 / math.sqrt(head_dim)
        qt = jnp.transpose(q, (0, 2, 1, 3))
        kt = jnp.transpose(k, (0, 2, 1, 3))
        vt = jnp.transpose(v, (0, 2, 1, 3))

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", qt, kt) * scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        if (
            self.window_size is not None
            and self.window_size >= 0
            and self.window_size < T
        ):
            window_mask = jnp.triu(
                jnp.ones((T, T), dtype=jnp.bool_), k=-(self.window_size - 1)
            )
            mask = causal_mask & window_mask
        else:
            mask = causal_mask

        mask = mask[None, None, :, :]
        attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(DTYPE)

        y = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, vt)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape((B, T, -1))
        y = nn.Dense(
            self.config.n_embd,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="c_proj",
        )(y)
        return y


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            4 * self.config.n_embd, use_bias=False, kernel_init=init_linear, name="c_fc"
        )(x)
        x = jnp.square(nn.relu(x))
        x = nn.Dense(
            self.config.n_embd,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="c_proj",
        )(x)
        return x


class Block(nn.Module):
    config: GPTConfig
    layer_idx: int
    window_size: int

    @nn.compact
    def __call__(self, x, ve, cos_sin):
        attn_out = CausalSelfAttention(
            self.config, self.layer_idx, self.window_size, name="attn"
        )(rms_norm(x), ve, cos_sin)
        x = x + attn_out
        mlp_out = MLP(self.config, name="mlp")(rms_norm(x))
        x = x + mlp_out
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape
        window_sizes = compute_window_sizes(self.config)

        wte = nn.Embed(
            self.config.vocab_size,
            self.config.n_embd,
            embedding_init=init_wte,
            name="wte",
        )
        x = wte(idx)

        head_dim = self.config.n_embd // self.config.n_head
        # precompute only up to T
        cos, sin = precompute_rotary_embeddings(T, head_dim)

        x = rms_norm(x)
        x0 = x

        resid_lambdas = self.param(
            "resid_lambdas", nn.initializers.ones, (self.config.n_layer,), jnp.float32
        )
        x0_lambdas = self.param(
            "x0_lambdas",
            nn.initializers.constant(0.1),
            (self.config.n_layer,),
            jnp.float32,
        )

        for i in range(self.config.n_layer):
            x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            if has_ve(i, self.config.n_layer):
                ve = nn.Embed(
                    self.config.vocab_size,
                    self.config.n_kv_head * head_dim,
                    embedding_init=init_linear,
                    name=f"value_embeds_{i}",
                )(idx)
            else:
                ve = None

            x = nn.remat(Block)(self.config, i, window_sizes[i], name=f"h_{i}")(
                x, ve, (cos, sin)
            )

        x = rms_norm(x)

        lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=init_lm_head,
            name="lm_head",
        )
        logits = lm_head(x)

        softcap = 15.0
        logits = softcap * jnp.tanh(logits / softcap)

        return logits


def estimate_flops(model_config, num_params, wte_params, lm_head_params):
    nparams_exclude = wte_params + lm_head_params + model_config.n_layer * 2
    h = model_config.n_head
    q = model_config.n_embd // model_config.n_head
    t = model_config.sequence_len
    attn_flops = 0
    window_sizes = compute_window_sizes(model_config)
    for window in window_sizes:
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq
    return 6 * (num_params - nparams_exclude) + attn_flops


# ---------------------------------------------------------------------------
# Custom Optimizer (AdamW)
# ---------------------------------------------------------------------------

def init_adamw_state(params):
    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "exp_avg": jax.tree_util.tree_map(jnp.zeros_like, params),
        "exp_avg_sq": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

def adamw_step(params, grads, state, lr, wd, beta1=0.9, beta2=0.95, eps=1e-8):
    step = state["step"] + 1
    
    def get_p(p, g, m, v):
        is_2d = p.ndim >= 2
        actual_wd = wd if is_2d else 0.0
        p = p * (1.0 - lr * actual_wd)
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * jnp.square(g)
        m_hat = m_new / (1 - beta1 ** step)
        v_hat = v_new / (1 - beta2 ** step)
        return p - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    def get_m(p, g, m, v):
        return beta1 * m + (1 - beta1) * g

    def get_v(p, g, m, v):
        return beta2 * v + (1 - beta2) * jnp.square(g)

    new_params = jax.tree_util.tree_map(get_p, params, grads, state["exp_avg"], state["exp_avg_sq"])
    new_m = jax.tree_util.tree_map(get_m, params, grads, state["exp_avg"], state["exp_avg_sq"])
    new_v = jax.tree_util.tree_map(get_v, params, grads, state["exp_avg"], state["exp_avg_sq"])
    
    new_state = {
        "step": step,
        "exp_avg": new_m,
        "exp_avg_sq": new_v,
    }
    return new_params, new_state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def cross_entropy_loss(logits, targets):
    logits_flat = logits.reshape((-1, logits.shape[-1]))
    targets_flat = targets.reshape((-1,))
    mask = targets_flat != -1
    safe_targets = jnp.where(mask, targets_flat, 0)

    # log softmax and gather
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    picked = jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)

    loss = -picked * mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0), loss


def loss_fn(params, model, x, y):
    logits = model.apply({"params": params}, x)
    loss, raw_loss = cross_entropy_loss(logits, y)
    return loss, raw_loss


@functools.partial(jax.jit, static_argnames=("grad_accum_steps",))
def train_step_accum(
    params,
    state,
    x_batch,
    y_batch,
    lr,
    wd,
    grad_accum_steps,
):
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)

    def accum_body(carry, xy):
        acc_grads, lsum = carry
        x, y = xy
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model, x, y
        )
        acc_grads = jax.tree_util.tree_map(
            lambda a, b: a + b / grad_accum_steps, acc_grads, grads
        )
        return (acc_grads, lsum + loss), None

    (final_grads, total_loss), _ = jax.lax.scan(
        accum_body,
        (zero_grads, jnp.array(0.0)),
        (x_batch, y_batch),
        length=grad_accum_steps,
        unroll=1,
    )

    new_params, new_state = adamw_step(
        params, final_grads, state, lr, wd, beta1=ADAM_BETAS[0], beta2=ADAM_BETAS[1]
    )

    return new_params, new_state, total_loss / grad_accum_steps


# BPB Eval locally in JAX
@jax.jit
def eval_step(params, x, y):
    logits = model.apply({"params": params}, x)
    _, loss_flat = cross_entropy_loss(logits, y)
    return loss_flat


def evaluate_bpb_jax(params, tokenizer, batch_size, seq_len=MAX_SEQ_LEN):
    token_bytes = get_token_bytes(device="cpu").numpy()
    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val", device="cpu")
    steps = EVAL_TOKENS // (batch_size * seq_len)
    total_nats = jnp.array(0.0)
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        x_jax = jnp.asarray(x.numpy())
        y_jax = jnp.asarray(y.numpy())

        loss_flat = eval_step(params, x_jax, y_jax)

        y_flat = y.numpy().reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0

        # Accumulate on device to avoid synchronizing inside the loop
        mask_jax = jax.device_put(mask)
        total_nats += jnp.sum(loss_flat.reshape(-1) * mask_jax)
        total_bytes += int(nbytes.sum())

    return float(total_nats) / (math.log(2) * total_bytes)


if __name__ == "__main__":
    t_start = time.time()
    import numpy as np

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    def build_model_config(depth):
        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM
        return GPTConfig(
            sequence_len=SEQUENCE_LEN,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
        )

    config = build_model_config(DEPTH)
    print(f"Model config: {asdict(config)}")

    model = GPT(config)
    rng = jax.random.PRNGKey(42)
    dummy_idx = jnp.ones((1, SEQUENCE_LEN), dtype=jnp.int32)
    variables = model.init(rng, dummy_idx)
    params = variables["params"]

    opt_state = init_adamw_state(params)

    # param counts
    def count_params(tree):
        return sum(x.size for x in jax.tree_util.tree_leaves(tree))

    num_params = count_params(params)
    num_wte = count_params(params["wte"])
    num_lm_head = count_params(params["lm_head"])

    # Approximate estimated FLOPs internally
    num_flops_per_token = estimate_flops(config, num_params, num_wte, num_lm_head)
    print(f"Total params: {num_params:,}")
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * SEQUENCE_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    train_loader = make_dataloader(
        tokenizer, DEVICE_BATCH_SIZE, SEQUENCE_LEN, "train", device="cpu"
    )
    x_pt, y_pt, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    t0 = time.time()
    dt = 0

    # We buffer grad accum batches in NumPy, then send to JAX
    x_batch_np = np.zeros(
        (grad_accum_steps, DEVICE_BATCH_SIZE, SEQUENCE_LEN), dtype=np.int32
    )
    y_batch_np = np.zeros(
        (grad_accum_steps, DEVICE_BATCH_SIZE, SEQUENCE_LEN), dtype=np.int32
    )

    while True:
        for micro_step in range(grad_accum_steps):
            x_batch_np[micro_step] = x_pt.numpy()
            y_batch_np[micro_step] = y_pt.numpy()
            x_pt, y_pt, epoch = next(train_loader)

        x_jax = jax.device_put(x_batch_np)
        y_jax = jax.device_put(y_batch_np)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        lr = BASE_LR * lrm
        wd = get_weight_decay(progress)

        params, opt_state, train_loss = train_step_accum(
            params,
            opt_state,
            x_jax,
            y_jax,
            jnp.array(lr),
            jnp.array(wd),
            grad_accum_steps,
        )

        train_loss_f = float(train_loss)

        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            sys.exit(1)

        t1 = time.time()
        # Initial compilation obscures dt for step 0
        if step == 0:
            dt = 0
            t0 = t1
        else:
            dt = t1 - t0
            t0 = t1

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
        mfu = (
            (100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS)
            if dt > 0
            else 0
        )
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {lr:.4f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step > 10:
            total_training_time += dt

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    total_tokens = step * TOTAL_BATCH_SIZE

    # Final eval
    val_bpb = evaluate_bpb_jax(params, tokenizer, EVAL_BATCH_SIZE, seq_len=MAX_SEQ_LEN)

    t_end = time.time()
    steady_state_mfu = (
        100
        * num_flops_per_token
        * TOTAL_BATCH_SIZE
        * max(0, step - 10)
        / max(total_training_time, 1e-6)
        / PEAK_FLOPS
    )

    # Try getting simulated peak vram usage from memory allocator via JAX
    # It might not expose peak bytes easily, fallback to 0
    try:
        mem_info = jax.local_devices()[0].memory_stats()
        peak_vram_mb = mem_info.get("peak_bytes_in_use", 0) / (1024 * 1024)
    except Exception:
        peak_vram_mb = 0.0

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
