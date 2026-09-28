import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import hparams as hp

def window_sumsquare(window,n_frames,hop_length=hp.hop_length,win_length=hp.win_length,n_fft=hp.filter_length,dtype=np.float32,norm=None):
    if win_length is None: win_length=n_fft
    n=n_fft+hop_length*(n_frames-1); x=np.zeros(n,dtype=dtype)
    win_sq=librosa_util.normalize(get_window(window,win_length,fftbins=True),norm=norm)**2
    win_sq=librosa_util.pad_center(win_sq,size=n_fft)
    for i in range(n_frames):
        sample=i*hop_length; x[sample:min(n,sample+n_fft)]+=win_sq[:max(0,min(n_fft,n-sample))]
    return x

def griffin_lim(magnitudes,stft_fn,n_iters=30):
    angles=np.angle(np.exp(2j*np.pi*np.random.rand(*magnitudes.size()))).astype(np.float32)
    angles=torch.autograd.Variable(torch.from_numpy(angles)); signal=stft_fn.inverse(magnitudes,angles).squeeze(1)
    for _ in range(n_iters): _,angles=stft_fn.transform(signal); signal=stft_fn.inverse(magnitudes,angles).squeeze(1)
    return signal

def dynamic_range_compression(x,C=1,clip_val=1e-5): return torch.log(torch.clamp(x,min=clip_val)*C)
def dynamic_range_decompression(x,C=1): return torch.exp(x)/C
