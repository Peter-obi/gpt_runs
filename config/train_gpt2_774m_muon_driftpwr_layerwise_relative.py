# 774M Muon + drift-opposed noise (layerwise-relative lambda=82)
# Tests whether post-step noise adds anything on top of Muon's orthogonalized step.
# Baseline to beat: Muon alone (train_gpt2_774m_muon_baseline.py)
# AdamW: val@250=5.7453, val@500=4.7119
#
# Use with:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     config/train_gpt2_774m_muon_driftpwr_layerwise_relative.py

optimizer_type = 'muon'
out_dir = 'out-gpt2-774m-muon-driftpwr-layerwise-relative'
wandb_run_name = 'gpt2-774m-muon-driftpwr-layerwise-relative'
wandb_log = False
dataset = '/content'

# GPT-2 Large architecture (~774M)
n_layer = 36
n_head = 20
n_embd = 1280

learning_rate = 2e-2
min_lr = 2e-3
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5
weight_decay = 0.0

noise_type = 'drift-power-opposed-layerwise-relative'
noise_scale = 82.0       # lambda: ||noise_l|| = 82 * ||step_l||
noise_power = 0.3
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
