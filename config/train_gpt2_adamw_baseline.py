# AdamW baseline override for train.py
# Use with: python train.py config/train_gpt2.py config/train_gpt2_adamw_baseline.py
#
# Single-GPU RTX 6000 Blackwell (96GB) signal check.
# Effective batch: 64 * 1024 * 16 = 1,048,576 tokens/step  (~1024 sequences, Adam-dominates regime)
# Token budget:    ~5.2B tokens  (~2.5 hrs on RTX 6000 Blackwell)

optimizer_type = 'adamw'
out_dir = 'out-gpt2-adamw-baseline'
wandb_run_name = 'gpt2-adamw-baseline'
dataset = '/content'

# Explicitly restate baseline optimizer knobs
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
min_lr = 6e-5  # learning_rate / 10

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
