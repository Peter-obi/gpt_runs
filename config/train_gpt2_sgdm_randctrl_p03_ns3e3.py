# SGD+momentum + drift-power random norm-matched control noise
# Use with: python train.py config/train_gpt2.py config/train_gpt2_sgdm_randctrl_p03_ns3e3.py

optimizer_type = 'sgd'
out_dir = 'out-gpt2-sgdm-randctrl-p03-ns3e3'
wandb_run_name = 'gpt2-sgdm-randctrl-p03-ns3e3'
dataset = '/content'

learning_rate = 1e-2
min_lr = 1e-3
sgd_momentum = 0.9
sgd_nesterov = False
weight_decay = 0.0

noise_type = 'drift-power-random-norm-matched'
noise_scale = 3e-3
noise_power = 0.3
noise_start_iter = 0
noise_clip_rms_mult = 0.0

batch_size = 64
gradient_accumulation_steps = 16

max_iters = 2500
lr_decay_iters = 2500

eval_interval = 250
eval_iters = 200
log_interval = 10
