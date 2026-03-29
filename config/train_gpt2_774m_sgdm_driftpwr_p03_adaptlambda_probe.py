# 774M probe run to measure natural noise/step ratio
# This mirrors the 124M frozen probe setup for apples-to-apples comparison.
# Use with:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     config/train_gpt2_774m_sgdm_driftpwr_p03_adaptlambda_probe.py

optimizer_type = 'sgd'
out_dir = 'out-gpt2-774m-adaptlambda-probe'
wandb_run_name = 'gpt2-774m-adaptlambda-probe'
wandb_log = False
dataset = '/content'

# GPT-2 Large architecture (~774M)
n_layer = 36
n_head = 20
n_embd = 1280

learning_rate = 1e-2
min_lr = 1e-3
sgd_momentum = 0.9
sgd_nesterov = False
weight_decay = 0.0

# Adaptive noise (frozen probe): keep lambda fixed at ns=3e-3
noise_type = 'drift-power-opposed-adaptive'
noise_scale = 3e-3
noise_power = 0.3
noise_start_iter = 0
noise_clip_rms_mult = 0.0

noise_adaptive_target_ratio = 999.0   # unreachable — keeps lambda frozen
noise_adaptive_k = 0.0                # no adaptation
noise_adaptive_min_scale = 3e-3
noise_adaptive_max_scale = 3e-3
noise_adaptive_ema_beta = 0.9

batch_size = 16
gradient_accumulation_steps = 64  # 16 * 1024 * 64 = 1,048,576 tokens/step

warmup_iters = 250
max_iters = 2500
lr_decay_iters = 2500
eval_interval = 250
eval_iters = 200
log_interval = 10
