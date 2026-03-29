# 774M Signum baseline — matched to SGD+noise runs for fair comparison
# Same warmup, max_iters, eval_interval as layerwise-relative runs.
# Baseline to beat: AdamW val@250=5.7453, val@500=4.7119
# SGD+noise best: val@250=5.9441 (lambda=82)
#
# Use with:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     config/train_gpt2_774m_signum_probe250.py

optimizer_type = 'signum'
out_dir = 'out-gpt2-774m-signum'
wandb_run_name = 'gpt2-774m-signum'
wandb_log = False
dataset = '/content'

# GPT-2 Large architecture (~774M)
n_layer = 36
n_head = 20
n_embd = 1280

learning_rate = 1e-3     # Signum: sign-normalized, moderate lr between 3e-4 (too slow) and 3e-3 (diverges)
min_lr = 1e-4
sgd_momentum = 0.9
weight_decay = 0.0

noise_type = 'none'
noise_scale = 0.0

batch_size = 16
gradient_accumulation_steps = 64  # 16 * 1024 * 64 = 1,048,576 tokens/step

warmup_iters = 250
max_iters = 2500
lr_decay_iters = 2500
eval_interval = 250
eval_iters = 200
log_interval = 10
