# 774M Muon baseline — no noise
# Muon: Newton-Schulz orthogonalized momentum for 2D params, sign update for 1D.
# Baseline to beat: AdamW val@250=5.7453, val@500=4.7119
# SGD+noise best: val@250=5.9202 (lambda=200), val@500=5.2544 (lambda=82)
#
# Use with:
#   python train_drift_adaptive.py \
#     config/train_gpt2.py \
#     config/train_gpt2_774m_muon_baseline.py

optimizer_type = 'muon'
out_dir = 'out-gpt2-774m-muon-baseline'
wandb_run_name = 'gpt2-774m-muon-baseline'
wandb_log = False
dataset = '/content'

# GPT-2 Large architecture (~774M)
n_layer = 36
n_head = 20
n_embd = 1280

learning_rate = 2e-2     # standard Muon lr
min_lr = 2e-3
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5
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
