import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center,tiny
from librosa.filters import mel as librosa_mel_fn
from audio.audio_processing import dynamic_range_compression,dynamic_range_decompression,window_sumsquare

class STFT(torch.nn.Module):
    def __init__(self,filter_length,hop_length,win_length,window='hann'):
        super().__init__(); self.filter_length=filter_length; self.hop_length=hop_length; self.win_length=win_length; self.window=window
        scale=filter_length/hop_length; fb=np.fft.fft(np.eye(filter_length)); cutoff=int(filter_length/2+1)
        fb=np.vstack([np.real(fb[:cutoff,:]),np.imag(fb[:cutoff,:])])
        forward=torch.FloatTensor(fb[:,None,:]); inverse=torch.FloatTensor(np.linalg.pinv(scale*fb).T[:,None,:])
        if window is not None:
            w=torch.from_numpy(pad_center(get_window(window,win_length,fftbins=True),size=filter_length)).float()
            forward*=w; inverse*=w
        self.register_buffer('forward_basis',forward.float()); self.register_buffer('inverse_basis',inverse.float())
    def transform(self,x):
        b,n=x.size(); x=F.pad(x.view(b,1,n).unsqueeze(1),(self.filter_length//2,self.filter_length//2,0,0),mode='reflect').squeeze(1)
        basis=self.forward_basis.to(x.device)
        ft=F.conv1d(x,basis,stride=self.hop_length); cutoff=self.filter_length//2+1
        real,imag=ft[:,:cutoff,:],ft[:,cutoff:,:]
        return torch.sqrt(real**2+imag**2),torch.atan2(imag,real)
    def inverse(self,magnitude,phase):
        x=torch.cat([magnitude*torch.cos(phase),magnitude*torch.sin(phase)],dim=1)
        inv=F.conv_transpose1d(x,self.inverse_basis.to(x.device),stride=self.hop_length)
        if self.window is not None:
            ws=window_sumsquare(self.window,magnitude.size(-1),self.hop_length,self.win_length,self.filter_length)
            idx=torch.from_numpy(np.where(ws>tiny(ws))[0]).to(inv.device); ws=torch.from_numpy(ws).to(inv.device)
            inv[:,:,idx]/=ws[idx]; inv*=float(self.filter_length)/self.hop_length
        return inv[:,:,self.filter_length//2:-self.filter_length//2]
    def forward(self,x):
        m,p=self.transform(x); return self.inverse(m,p)

class TacotronSTFT(torch.nn.Module):
    def __init__(self,filter_length,hop_length,win_length,n_mel_channels,sampling_rate,mel_fmin=0.,mel_fmax=8000.):
        super().__init__(); self.n_mel_channels=n_mel_channels; self.sampling_rate=sampling_rate
        self.stft_fn=STFT(filter_length,hop_length,win_length)
        basis=librosa_mel_fn(sr=sampling_rate,n_fft=filter_length,n_mels=n_mel_channels,fmin=mel_fmin,fmax=mel_fmax)
        self.register_buffer('mel_basis',torch.from_numpy(basis).float())
    def spectral_normalize(self,x): return dynamic_range_compression(x)
    def spectral_de_normalize(self,x): return dynamic_range_decompression(x)
    def mel_spectrogram(self,y):
        magnitudes,_=self.stft_fn.transform(y); mel=torch.matmul(self.mel_basis.to(magnitudes.device),magnitudes)
        return self.spectral_normalize(mel),torch.norm(magnitudes,dim=1)
