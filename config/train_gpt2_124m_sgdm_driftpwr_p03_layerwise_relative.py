# 124M scale-free layerwise relative-noise run (SGD+momentum + drift-power-opposed, p=0.3)
# Uses per-layer target: ||noise_l|| = rel_lambda * ||sgd_step_l||
# Use with:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     config/train_gpt2_124m_sgdm_driftpwr_p03_layerwise_relative.py

optimizer_type = 'sgd'
out_dir = 'out-gpt2-124m-sgdm-driftpwr-p03-layerwise-relative'
wandb_run_name = 'gpt2-124m-sgdm-driftpwr-p03-layerwise-relative'
wandb_log = False
dataset = '/content'

# 124M architecture
n_layer = 12
n_head = 12
n_embd = 768

learning_rate = 1e-2
min_lr = 1e-3
sgd_momentum = 0.9
sgd_nesterov = False
weight_decay = 0.0

noise_type = 'drift-power-opposed-layerwise-relative'
noise_scale = 82.0
noise_power = 0.3
noise_start_iter = 0
noise_clip_rms_mult = 0.0

batch_size = 64
gradient_accumulation_steps = 16  # 64 * 1024 * 16 = 1,048,576 tokens/step

warmup_iters = 250
max_iters = 2500
lr_decay_iters = 2500

eval_interval = 250
eval_iters = 200
log_interval = 10
