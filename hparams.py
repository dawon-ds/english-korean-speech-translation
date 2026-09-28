import os

dataset = "kss"
data_path = os.environ.get("KSS_DATA_PATH", os.path.join("data", "kss"))
preprocessed_path = os.path.join("preprocessed", "kss")
checkpoint_path = "checkpoint"
log_path = "log"
eval_path = "eval"
result_path = "results"

text_cleaners = ["korean_cleaners"]
train_visible_devices = "0"

batch_size = 16
epochs = 1000
acc_steps = 1
betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.0
decoder_hidden = 256
n_warm_up_step = 4000
log_offset = 1.0

encoder_layer = 4
decoder_layer = 4
encoder_head = 2
decoder_head = 2
encoder_hidden = 256
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
dropout = 0.1

n_mel_channels = 80
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
mel_fmin = 0.0
mel_fmax = 8000.0

vocoder = "vocgan"
vocoder_pretrained_model_path = os.path.join("vocoder", "pretrained", "generator.pth")
