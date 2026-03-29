# SGD+momentum baseline override for train.py
# Use with: python train.py config/train_gpt2.py config/train_gpt2_sgd_momentum_baseline.py
#
# Single-GPU RTX 6000 Blackwell (96GB) signal check.
# Effective batch: 64 * 1024 * 16 = 1,048,576 tokens/step  (~1024 sequences, Adam-dominates regime)
# Token budget:    ~5.2B tokens  (~2.5 hrs on RTX 6000 Blackwell)

optimizer_type = 'sgd'
out_dir = 'out-gpt2-sgdm-baseline'
wandb_run_name = 'gpt2-sgdm-baseline'
dataset = '/content'

# SGD+momentum knobs
learning_rate = 1e-2
min_lr = 1e-3  # learning_rate / 10
sgd_momentum = 0.9
sgd_nesterov = False
weight_decay = 0.0  # keep clean baseline; no AdamW-style decay

# RTX 6000 Blackwell: large micro-batch to use 96GB VRAM, fewer accum steps
batch_size = 64
gradient_accumulation_steps = 16  # 64 * 1024 * 16 = 1,048,576 tokens/step

# Token budget (~5.2B tokens = 5000 steps)
max_iters = 2500
lr_decay_iters = 2500

# Eval/log frequency
eval_interval = 250
eval_iters = 200
log_interval = 10
