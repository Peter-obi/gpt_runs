# 774M gated-noise layerwise relative run
# Gate: inject noise only where |g_i| > tau * |d_{k,i}|, tau=1.0
# Targets parameters where current gradient > tau * accumulated drift.
# Silences parameters where momentum has overshot the gradient.
# Layerwise-relative: ||noise_l|| = lambda * ||step_l||
#
# Use with:
#   python experiments/gated_noise/train_drift_gated.py \
#     config/train_gpt2.py \
#     experiments/gated_noise/train_gpt2_774m_sgdm_driftgated_p03_layerwise_relative.py
#
# Baseline to beat (774M layerwise-relative lambda=82):
#   val@250 = 5.9441, val@500 = 5.2544

optimizer_type = 'sgd'
out_dir = 'out-gpt2-774m-sgdm-driftgated-p03-layerwise-relative'
wandb_run_name = 'gpt2-774m-sgdm-driftgated-p03-layerwise-relative'
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

noise_type = 'drift-gated-opposed-layerwise-relative'
noise_scale = 82.0       # lambda: ||noise_l|| = 82 * ||step_l||
noise_power = 0.3        # unused for gated type but kept for consistency
noise_gate_tau = 1.0     # gate: fire where |g_i| > 1.0 * |d_{k,i}|
noise_start_iter = 0
noise_clip_rms_mult = 0.0

batch_size = 16
gradient_accumulation_steps = 64  # 16 * 1024 * 64 = 1,048,576 tokens/step

warmup_iters = 250
max_iters = 2500
lr_decay_iters = 2500
eval_interval = 250
eval_iters = 200
log_interval = 10
