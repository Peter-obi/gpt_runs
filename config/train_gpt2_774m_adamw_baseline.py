# AdamW baseline for GPT-2 Large (774M)
# Use with: python train.py config/train_gpt2.py config/train_gpt2_774m_adamw_baseline.py
#
# GPT-2 Large: n_layer=36, n_head=20, n_embd=1280 (~774M params)
# Effective batch: 16 * 1024 * 64 = 1,048,576 tokens/step (matches 124M runs)

optimizer_type = 'adamw'
out_dir = 'out-gpt2-774m-adamw-baseline'
wandb_run_name = 'gpt2-774m-adamw-baseline'
dataset = '/content'

# GPT-2 Large architecture
n_layer = 36
n_head = 20
n_embd = 1280

# AdamW hyperparams (scaled lr for larger model)
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
min_lr = 3e-5

# Smaller micro-batch to fit 774M in memory, same token budget via more accum steps
batch_size = 16
gradient_accumulation_steps = 64  # 16 * 1024 * 64 = 1,048,576 tokens/step

warmup_iters = 250
max_iters = 2500
lr_decay_iters = 2500

eval_interval = 250
eval_iters = 200
log_interval = 10
