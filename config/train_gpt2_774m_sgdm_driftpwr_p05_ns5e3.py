# SGD+momentum + drift-power-opposed (p=0.5) for GPT-2 Large (774M)
# Use with: python train.py config/train_gpt2.py config/train_gpt2_774m_sgdm_driftpwr_p05_ns5e3.py

optimizer_type = 'sgd'
out_dir = 'out-gpt2-774m-sgdm-driftpwr-p05-ns5e3'
wandb_run_name = 'gpt2-774m-sgdm-driftpwr-p05-ns5e3'
dataset = '/content'

# GPT-2 Large architecture
n_layer = 36
n_head = 20
n_embd = 1280

learning_rate = 1e-2
min_lr = 1e-3
sgd_momentum = 0.9
sgd_nesterov = False
weight_decay = 0.0

noise_type = 'drift-power-opposed'
noise_scale = 5e-3
noise_power = 0.5
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
