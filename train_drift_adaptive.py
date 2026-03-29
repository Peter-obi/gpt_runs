"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train_drift_adaptive.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_drift_adaptive.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_drift_adaptive.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_drift_adaptive.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import inspect
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

class Signum(torch.optim.Optimizer):
    """SignSGD with momentum (Signum) and optional decoupled weight decay."""

    def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if dampening < 0.0 or dampening > 1.0:
            raise ValueError(f"Invalid dampening: {dampening}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                # Keep decoupled decay behavior aligned with existing AdamW/SGD flows.
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                if momentum != 0.0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = grad.detach().clone()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(grad, alpha=(1.0 - dampening))
                    update = buf
                else:
                    update = grad

                p.add_(torch.sign(update), alpha=-lr)

        return loss

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration to orthogonalize G (returns G * (G^T G)^{-1/2})."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype
    X = G.to(torch.bfloat16) / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(orig_dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.
    2D weight matrices: Newton-Schulz orthogonalization applied to momentum.
    1D params (biases, layernorm): Signum-like sign update.
    Momentum buffer stored as 'momentum_buffer' for compatibility with noise injection.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                state = self.state[p]
                buf = state.get('momentum_buffer')
                if buf is None:
                    buf = g.detach().clone()
                    state['momentum_buffer'] = buf
                else:
                    buf.mul_(momentum).add_(g)
                update = (g + momentum * buf) if nesterov else buf
                if p.ndim == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    scale = max(1, p.size(0) / p.size(1)) ** 0.5
                    p.add_(update, alpha=-lr * scale)
                else:
                    p.add_(torch.sign(update), alpha=-lr)
        return loss

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# optimizer selection
optimizer_type = 'adamw' # 'adamw', 'sgd', 'signum', or 'muon'
sgd_momentum = 0.9
sgd_nesterov = False
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5
# optional post-step parameter noise
# Supported:
# - 'none'
# - 'drift-power-opposed'
# - 'drift-power-random-norm-matched'
# - 'drift-power-opposed-adaptive'
# - 'drift-power-opposed-layerwise-adaptive'
# - 'drift-power-opposed-layerwise-relative'
# - 'drift-inv-opposed-layerwise-relative'
# - 'drift-stale-gated-opposed-layerwise-relative'
# - 'drift-fresh-opposed-layerwise-relative'
noise_type = 'none'
noise_scale = 0.0
noise_power = 0.5
noise_start_iter = 0
noise_clip_rms_mult = 0.0
noise_eps = 1e-12
# inverse-weighting params (used when noise_type='drift-inv-opposed-layerwise-relative')
noise_inv_power = 0.3   # q: w_i = (|d_i| + eps)^{-q}
noise_inv_wmin = 0.01   # floor clip
noise_inv_wmax = 100.0  # cap clip
# stale-gated weighting params (used when noise_type='drift-stale-gated-opposed-layerwise-relative')
# r_i = |mu * m_i| / (|g_i| + eps), w_i = clip(r_i^a, w_min, w_max)
noise_stale_ratio_power = 0.5
noise_stale_wmin = 0.01
noise_stale_wmax = 100.0
# adaptive-lambda controller (used when noise_type='drift-power-opposed-adaptive')
noise_adaptive_target_ratio = 0.5  # target ||noise|| / ||optimizer_step||
noise_adaptive_k = 0.01
noise_adaptive_min_scale = 1e-5
noise_adaptive_max_scale = 5e-2
noise_adaptive_ema_beta = 0.0      # 0 = no EMA smoothing
noise_adaptive_start_iter = 0      # delay controller updates while still injecting noise
noise_adaptive_ratio_clip = 1e6    # clamp measured ratio before EMA/control

# layerwise controller cadence (used when noise_type='drift-power-opposed-layerwise-adaptive')
noise_layerwise_update_interval = 50
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0},
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

if optimizer_type == 'adamw':
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
    print(f"using optimizer=adamw fused={use_fused}")
elif optimizer_type == 'sgd':
    optimizer = torch.optim.SGD(
        optim_groups,
        lr=learning_rate,
        momentum=sgd_momentum,
        nesterov=sgd_nesterov,
    )
    print(f"using optimizer=sgd momentum={sgd_momentum} nesterov={sgd_nesterov}")
elif optimizer_type == 'signum':
    optimizer = Signum(
        optim_groups,
        lr=learning_rate,
        momentum=sgd_momentum,
        dampening=0.0,
        weight_decay=weight_decay,
    )
    print(f"using optimizer=signum momentum={sgd_momentum}")
elif optimizer_type == 'muon':
    optimizer = Muon(
        optim_groups,
        lr=learning_rate,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
        weight_decay=weight_decay,
    )
    print(f"using optimizer=muon lr={learning_rate} momentum={muon_momentum} nesterov={muon_nesterov} ns_steps={muon_ns_steps}")
else:
    raise ValueError(
        f"unknown optimizer_type={optimizer_type!r}, expected "
        "'adamw', 'sgd', 'signum', or 'muon'"
    )
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def quantiles_10_50_90(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        nan = float("nan")
        return nan, nan, nan
    vals.sort()
    n = len(vals)
    def q(frac):
        idx = int(round(frac * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return vals[idx]
    return q(0.1), q(0.5), q(0.9)

def inject_parameter_noise_opposed(
    model,
    optimizer,
    noise_type: str,
    noise_scale: float,
    noise_power: float,
    clip_rms_mult: float,
    eps: float,
    layerwise_scales=None,
    layerwise_ratio_ema=None,
    layerwise_target_ratio: float = 0.0,
    layerwise_k: float = 0.0,
    layerwise_min_scale: float = 0.0,
    layerwise_max_scale: float = 0.0,
    layerwise_ema_beta: float = 0.0,
    update_layerwise_controller: bool = False,
    layerwise_ratio_clip: float = 1e6,
):
    if noise_type == 'none' or noise_scale <= 0.0:
        return 0.0, 0.0
    if noise_type not in (
        'drift-power-opposed',
        'drift-power-random-norm-matched',
        'drift-power-opposed-adaptive',
        'drift-power-opposed-layerwise-adaptive',
        'drift-power-opposed-layerwise-relative',
        'drift-inv-opposed-layerwise-relative',
        'drift-stale-gated-opposed-layerwise-relative',
        'drift-fresh-opposed-layerwise-relative',
    ):
        raise ValueError(
            f"unknown noise_type={noise_type!r}, expected "
            "'none', 'drift-power-opposed', 'drift-power-random-norm-matched', "
            "'drift-power-opposed-adaptive', or "
            "'drift-power-opposed-layerwise-adaptive', or "
            "'drift-power-opposed-layerwise-relative', or "
            "'drift-inv-opposed-layerwise-relative', or "
            "'drift-stale-gated-opposed-layerwise-relative', or "
            "'drift-fresh-opposed-layerwise-relative'"
        )

    if noise_power < 0.0:
        raise ValueError(f"noise_power must be >= 0, got {noise_power}")
    if noise_stale_ratio_power < 0.0:
        raise ValueError(f"noise_stale_ratio_power must be >= 0, got {noise_stale_ratio_power}")
    if noise_stale_wmin <= 0.0 or noise_stale_wmax <= 0.0:
        raise ValueError("noise_stale_wmin and noise_stale_wmax must be > 0")
    if noise_stale_wmin > noise_stale_wmax:
        raise ValueError(
            f"noise_stale_wmin={noise_stale_wmin} cannot exceed noise_stale_wmax={noise_stale_wmax}"
        )

    param_to_momentum = {}
    param_to_lr = {}
    for group in optimizer.param_groups:
        mu = float(group.get('momentum', 0.0))
        lr = float(group.get('lr', 0.0))
        for p in group['params']:
            param_to_momentum[p] = mu
            param_to_lr[p] = lr

    perturbations = {}
    total_norm_sq = 0.0
    param_norm_sq = 0.0
    step_norm_sq = 0.0
    layerwise_mode = (noise_type == 'drift-power-opposed-layerwise-adaptive')
    layerwise_relative_mode = noise_type in (
        'drift-power-opposed-layerwise-relative',
        'drift-inv-opposed-layerwise-relative',
        'drift-stale-gated-opposed-layerwise-relative',
        'drift-fresh-opposed-layerwise-relative',
    )

    if layerwise_mode and (layerwise_scales is None or layerwise_ratio_ema is None):
        raise ValueError(
            "layerwise_scales and layerwise_ratio_ema must be provided for "
            "drift-power-opposed-layerwise-adaptive"
        )

    if layerwise_mode:
        layerwise_ema_beta = float(max(0.0, min(0.9999, layerwise_ema_beta)))
        layerwise_ratio_clip = float(layerwise_ratio_clip)
        if layerwise_ratio_clip <= 0.0:
            layerwise_ratio_clip = 1e-12

    for name, param in model.named_parameters():
        g = param.grad
        if g is None:
            continue

        drift = g.detach()
        state = optimizer.state.get(param, {})
        v = state.get('momentum_buffer')
        mu = float(param_to_momentum.get(param, 0.0))
        if v is not None:
            drift = mu * v + drift

        if not torch.isfinite(drift).all():
            continue

        mag = torch.pow(torch.abs(drift) + eps, noise_power)
        if clip_rms_mult > 0.0:
            rms = torch.sqrt(torch.mean(mag * mag) + eps)
            mag = torch.clamp(mag, max=clip_rms_mult * rms)

        if noise_type == 'drift-power-random-norm-matched':
            # Control: keep drift-derived magnitude profile but randomize direction.
            z_opp = torch.randn_like(param)
            opposed_like = -torch.sign(drift) * mag * torch.abs(z_opp)
            opp_norm = opposed_like.norm()

            z_rand = torch.randn_like(param)
            z_rand_norm = z_rand.norm()
            if opp_norm > 1e-12 and z_rand_norm > 1e-12:
                perturbation = z_rand * (opp_norm / z_rand_norm)
            else:
                perturbation = torch.zeros_like(param)
        elif noise_type == 'drift-inv-opposed-layerwise-relative':
            # Inverse drift weighting: louder for stuck params, quieter for active ones.
            # w_i = clip((|d_i| + eps)^{-q}, w_min, w_max)
            # perturbation = -sign(d) * w  (then renormalized layerwise)
            w = torch.pow(torch.abs(drift) + eps, -noise_inv_power)
            w = torch.clamp(w, min=noise_inv_wmin, max=noise_inv_wmax)
            perturbation = -torch.sign(drift) * w
        elif noise_type == 'drift-stale-gated-opposed-layerwise-relative':
            # Staleness-gated opposition:
            # r_i = |mu*m_i| / (|g_i| + eps), w_i = clip(r_i^a, w_min, w_max)
            # perturbation = -sign(mu*m_i) * w  (then renormalized layerwise)
            if v is None:
                perturbation = torch.zeros_like(param)
            else:
                stale = mu * v
                ratio = torch.abs(stale) / (torch.abs(g.detach()) + eps)
                w = torch.pow(ratio + eps, noise_stale_ratio_power)
                w = torch.clamp(w, min=noise_stale_wmin, max=noise_stale_wmax)
                perturbation = -torch.sign(stale) * w
        elif noise_type == 'drift-fresh-opposed-layerwise-relative':
            # Negative control: oppose fresh gradient direction only.
            perturbation = -torch.sign(g.detach())
        else:
            z = torch.randn_like(param)
            perturbation = -torch.sign(drift) * mag * torch.abs(z)
        if not torch.isfinite(perturbation).all():
            continue

        perturbations[param] = perturbation
        lr_param = float(param_to_lr.get(param, 0.0))
        drift_step_norm_sq = (drift * lr_param).pow(2).sum().item()
        step_norm_sq += drift_step_norm_sq

        if layerwise_relative_mode:
            step_norm_l = drift_step_norm_sq ** 0.5
            target_norm_l = float(noise_scale) * step_norm_l
            total_norm_sq += target_norm_l * target_norm_l

            current_norm_l = perturbation.norm().item()
            if current_norm_l > 1e-12 and target_norm_l > 0.0:
                alpha_l = target_norm_l / current_norm_l
                perturbation = perturbation * alpha_l
            else:
                perturbation = torch.zeros_like(param)
            perturbations[param] = perturbation
        elif layerwise_mode:
            scale_l = float(layerwise_scales.get(name, noise_scale))
            if not math.isfinite(scale_l):
                scale_l = float(noise_scale)
            scale_l = max(float(layerwise_min_scale), min(float(layerwise_max_scale), scale_l))
            layerwise_scales[name] = scale_l

            current_norm_l = perturbation.norm().item()
            param_norm_l = param.data.norm().item()
            target_norm_l = scale_l * param_norm_l
            total_norm_sq += target_norm_l * target_norm_l

            if current_norm_l > 1e-12 and target_norm_l > 0.0:
                alpha_l = target_norm_l / current_norm_l
                perturbation = perturbation * alpha_l
            else:
                perturbation = torch.zeros_like(param)
            perturbations[param] = perturbation

            if update_layerwise_controller and drift_step_norm_sq > 1e-24 and target_norm_l > 0.0:
                step_norm_l = drift_step_norm_sq ** 0.5
                ratio_l = target_norm_l / step_norm_l
                if math.isfinite(ratio_l):
                    ratio_l = max(0.0, min(layerwise_ratio_clip, ratio_l))
                else:
                    continue
                prev_ema = layerwise_ratio_ema.get(name)
                if prev_ema is None:
                    ema_l = ratio_l
                else:
                    ema_l = layerwise_ema_beta * prev_ema + (1.0 - layerwise_ema_beta) * ratio_l
                layerwise_ratio_ema[name] = ema_l
                err_l = float(layerwise_target_ratio) - ema_l
                next_scale_l = scale_l * math.exp(float(layerwise_k) * err_l)
                next_scale_l = max(
                    float(layerwise_min_scale),
                    min(float(layerwise_max_scale), next_scale_l),
                )
                if math.isfinite(next_scale_l):
                    layerwise_scales[name] = next_scale_l
        else:
            total_norm_sq += perturbation.pow(2).sum().item()
            param_norm_sq += param.data.pow(2).sum().item()

    current_norm = total_norm_sq ** 0.5
    if current_norm < 1e-12:
        return 0.0, step_norm_sq ** 0.5

    if layerwise_mode or layerwise_relative_mode:
        with torch.no_grad():
            for param, perturbation in perturbations.items():
                param.add_(perturbation)
        return total_norm_sq ** 0.5, step_norm_sq ** 0.5

    target_norm = noise_scale * (param_norm_sq ** 0.5)
    alpha = target_norm / current_norm
    with torch.no_grad():
        for param, perturbation in perturbations.items():
            param.add_(perturbation, alpha=alpha)
    return target_norm, step_norm_sq ** 0.5

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
adaptive_noise_type = (noise_type == 'drift-power-opposed-adaptive')
layerwise_adaptive_noise_type = (noise_type == 'drift-power-opposed-layerwise-adaptive')
layerwise_relative_noise_type = noise_type in (
    'drift-power-opposed-layerwise-relative',
    'drift-inv-opposed-layerwise-relative',
    'drift-stale-gated-opposed-layerwise-relative',
    'drift-fresh-opposed-layerwise-relative',
)
adaptive_noise_scale = float(noise_scale)
adaptive_ratio_ema = None
layerwise_noise_scales = {}
layerwise_ratio_emas = {}
if adaptive_noise_type:
    if noise_adaptive_min_scale <= 0.0 or noise_adaptive_max_scale <= 0.0:
        raise ValueError(
            "noise_adaptive_min_scale and noise_adaptive_max_scale must be > 0"
        )
    if noise_adaptive_min_scale > noise_adaptive_max_scale:
        raise ValueError(
            f"noise_adaptive_min_scale={noise_adaptive_min_scale} cannot exceed "
            f"noise_adaptive_max_scale={noise_adaptive_max_scale}"
        )
    adaptive_noise_scale = max(
        float(noise_adaptive_min_scale),
        min(float(noise_adaptive_max_scale), adaptive_noise_scale),
    )
    if noise_adaptive_start_iter < noise_start_iter:
        raise ValueError(
            f"noise_adaptive_start_iter={noise_adaptive_start_iter} must be >= "
            f"noise_start_iter={noise_start_iter}"
        )
    if noise_adaptive_ratio_clip <= 0.0:
        raise ValueError("noise_adaptive_ratio_clip must be > 0")
if layerwise_adaptive_noise_type:
    if noise_adaptive_min_scale <= 0.0 or noise_adaptive_max_scale <= 0.0:
        raise ValueError(
            "noise_adaptive_min_scale and noise_adaptive_max_scale must be > 0"
        )
    if noise_adaptive_min_scale > noise_adaptive_max_scale:
        raise ValueError(
            f"noise_adaptive_min_scale={noise_adaptive_min_scale} cannot exceed "
            f"noise_adaptive_max_scale={noise_adaptive_max_scale}"
        )
    if noise_layerwise_update_interval <= 0:
        raise ValueError("noise_layerwise_update_interval must be >= 1")
    if noise_adaptive_start_iter < noise_start_iter:
        raise ValueError(
            f"noise_adaptive_start_iter={noise_adaptive_start_iter} must be >= "
            f"noise_start_iter={noise_start_iter}"
        )
    if noise_adaptive_ratio_clip <= 0.0:
        raise ValueError("noise_adaptive_ratio_clip must be > 0")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wb_payload = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if adaptive_noise_type:
                wb_payload["noise/scale"] = adaptive_noise_scale
                if adaptive_ratio_ema is not None:
                    wb_payload["noise/ratio_ema"] = adaptive_ratio_ema
            wandb.log(wb_payload)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    noise_norm = 0.0
    noise_ratio = float("nan")
    effective_noise_scale = float(noise_scale)
    if iter_num >= noise_start_iter:
        update_layerwise_controller = False
        if adaptive_noise_type:
            effective_noise_scale = adaptive_noise_scale
        elif layerwise_adaptive_noise_type:
            if iter_num >= noise_adaptive_start_iter:
                update_layerwise_controller = (
                    ((iter_num - noise_adaptive_start_iter) % int(noise_layerwise_update_interval)) == 0
                )
        noise_norm, step_norm = inject_parameter_noise_opposed(
            raw_model,
            optimizer,
            noise_type=noise_type,
            noise_scale=effective_noise_scale,
            noise_power=noise_power,
            clip_rms_mult=noise_clip_rms_mult,
            eps=noise_eps,
            layerwise_scales=layerwise_noise_scales,
            layerwise_ratio_ema=layerwise_ratio_emas,
            layerwise_target_ratio=float(noise_adaptive_target_ratio),
            layerwise_k=float(noise_adaptive_k),
            layerwise_min_scale=float(noise_adaptive_min_scale),
            layerwise_max_scale=float(noise_adaptive_max_scale),
            layerwise_ema_beta=float(noise_adaptive_ema_beta),
            update_layerwise_controller=update_layerwise_controller,
            layerwise_ratio_clip=float(noise_adaptive_ratio_clip),
        )
        if step_norm > 1e-12 and noise_norm > 0.0:
            noise_ratio = noise_norm / step_norm
            if math.isfinite(noise_ratio):
                noise_ratio = max(0.0, min(float(noise_adaptive_ratio_clip), noise_ratio))
            else:
                noise_ratio = float("nan")
            if adaptive_noise_type:
                beta = float(noise_adaptive_ema_beta)
                beta = max(0.0, min(0.9999, beta))
                if adaptive_ratio_ema is None:
                    adaptive_ratio_ema = noise_ratio
                else:
                    adaptive_ratio_ema = beta * adaptive_ratio_ema + (1.0 - beta) * noise_ratio
                if iter_num >= noise_adaptive_start_iter:
                    ratio_for_ctrl = adaptive_ratio_ema
                    err = float(noise_adaptive_target_ratio) - ratio_for_ctrl
                    next_scale = effective_noise_scale * math.exp(float(noise_adaptive_k) * err)
                    next_scale = max(
                        float(noise_adaptive_min_scale),
                        min(float(noise_adaptive_max_scale), next_scale),
                    )
                    if math.isfinite(next_scale):
                        adaptive_noise_scale = next_scale
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        nscale_label = "nscale"
        layerwise_suffix = ""
        if layerwise_adaptive_noise_type:
            nscale_label = "base_nscale"
            s10, s50, s90 = quantiles_10_50_90(layerwise_noise_scales.values())
            r10, r50, r90 = quantiles_10_50_90(layerwise_ratio_emas.values())
            layerwise_suffix = (
                f", lscale[p10/p50/p90]={s10:.2e}/{s50:.2e}/{s90:.2e}, "
                f"lratio_ema[p10/p50/p90]={r10:.2f}/{r50:.2f}/{r90:.2f}, "
                f"nlayers={len(layerwise_noise_scales)}"
            )
        elif layerwise_relative_noise_type:
            nscale_label = "rel_lambda"
        print(
            f"iter {iter_num}: loss {lossf:.4f}, noise {noise_norm:.3e}, "
            f"{nscale_label} {effective_noise_scale:.3e}, ratio {noise_ratio:.3e}, "
            f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            f"{layerwise_suffix}"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
