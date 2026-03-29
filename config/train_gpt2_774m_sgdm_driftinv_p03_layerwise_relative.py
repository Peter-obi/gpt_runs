# 774M inverse-drift layerwise relative-noise run
# Per-parameter weighting: w_i = clip((|d_i|+eps)^{-q}, w_min, w_max)
# Louder noise for stuck params, quieter for active ones.
# Layerwise-relative normalization: ||noise_l|| = lambda * ||step_l||
# Use with:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     config/train_gpt2_774m_sgdm_driftinv_p03_layerwise_relative.py

optimizer_type = 'sgd'
out_dir = 'out-gpt2-774m-sgdm-driftinv-p03-layerwise-relative'
wandb_run_name = 'gpt2-774m-sgdm-driftinv-p03-layerwise-relative'
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

noise_type = 'drift-inv-opposed-layerwise-relative'
noise_scale = 82.0        # lambda: ||noise_l|| = 82 * ||step_l||
noise_power = 0.3         # unused for inv type but kept for consistency
noise_inv_power = 0.3     # q: w_i = (|d_i| + eps)^{-q}
noise_inv_wmin = 0.01     # floor clip
noise_inv_wmax = 100.0    # cap clip
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
