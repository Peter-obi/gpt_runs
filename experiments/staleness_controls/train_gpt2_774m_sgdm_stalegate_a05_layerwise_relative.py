# 774M staleness-gated opposed noise (layerwise-relative)
# r_i = |mu * m_i| / (|g_i| + eps)
# w_i = clip(r_i^a, w_min, w_max)
# perturbation_i ∝ -sign(mu * m_i) * w_i
# then per-layer renorm: ||noise_l|| = lambda * ||step_l||
#
# Run:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     experiments/staleness_controls/train_gpt2_774m_sgdm_stalegate_a05_layerwise_relative.py

optimizer_type = 'sgd'
out_dir = 'out-gpt2-774m-sgdm-stalegate-a05-layerwise-relative'
wandb_run_name = 'gpt2-774m-sgdm-stalegate-a05-layerwise-relative'
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

noise_type = 'drift-stale-gated-opposed-layerwise-relative'
noise_scale = 82.0
noise_power = 0.3  # unused by this mode
noise_stale_ratio_power = 0.5
noise_stale_wmin = 0.01
noise_stale_wmax = 100.0
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
