import torch
import numpy as np
from scipy.io.wavfile import read,write
import audio.stft as stft
from audio.audio_processing import griffin_lim
import hparams

_stft=stft.TacotronSTFT(hparams.filter_length,hparams.hop_length,hparams.win_length,hparams.n_mel_channels,hparams.sampling_rate,hparams.mel_fmin,hparams.mel_fmax)

def load_wav_to_torch(full_path):
    sr,data=read(full_path); return torch.FloatTensor(data.astype(np.float32)),sr

def get_mel(filename):
    audio,sr=load_wav_to_torch(filename)
    if sr!=_stft.sampling_rate: raise ValueError(f"{sr} SR doesn't match target {_stft.sampling_rate} SR")
    audio=torch.autograd.Variable((audio/hparams.max_wav_value).unsqueeze(0),requires_grad=False)
    mel,energy=_stft.mel_spectrogram(audio)
    return torch.squeeze(mel,0),torch.squeeze(energy,0)

def get_mel_from_wav(audio):
    audio=torch.autograd.Variable((audio/hparams.max_wav_value).unsqueeze(0),requires_grad=False)
    mel,energy=_stft.mel_spectrogram(audio); return torch.squeeze(mel,0),torch.squeeze(energy,0)

def inv_mel_spec(mel,out_filename,griffin_iters=60):
    mel=_stft.spectral_de_normalize(torch.stack([mel])).transpose(1,2).data.cpu()
    spec=torch.mm(mel[0],_stft.mel_basis).transpose(0,1).unsqueeze(0)*1000
    audio=griffin_lim(torch.autograd.Variable(spec[:,:,:-1]),_stft.stft_fn,griffin_iters).squeeze().cpu().numpy()
    write(out_filename,hparams.sampling_rate,audio)
