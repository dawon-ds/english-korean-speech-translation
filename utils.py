import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile
from vocoder.vocgan_generator import Generator
import hparams as hp
import os
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_alignment(tier):
    sil=['sil','sp','spn']; phones=[]; durations=[]; start=0; end=0; end_idx=0
    for t in tier._objects:
        s,e,p=t.start_time,t.end_time,t.text
        if not phones and p in sil: continue
        if not phones: start=s
        phones.append(p)
        if p not in sil: end=e; end_idx=len(phones)
        durations.append(int(e*hp.sampling_rate/hp.hop_length)-int(s*hp.sampling_rate/hp.hop_length))
    return phones[:end_idx],np.array(durations[:end_idx]),start,end

def process_meta(path):
    with open(path,'r',encoding='utf-8') as f:
        rows=[x.strip().split('|',1) for x in f if x.strip()]
    return [x[0] for x in rows],[x[1] for x in rows]

def get_param_num(model): return sum(p.numel() for p in model.parameters())
def get_mask_from_lengths(lengths,max_len=None):
    max_len=max_len or torch.max(lengths).item(); ids=torch.arange(max_len,device=lengths.device).unsqueeze(0).expand(lengths.shape[0],-1)
    return ids>=lengths.unsqueeze(1)

def get_vocgan(ckpt_path,n_mel_channels=hp.n_mel_channels,generator_ratio=[4,4,2,2,2,2],n_residual_layers=4,mult=256,out_channels=1):
    ckpt=torch.load(ckpt_path,map_location=device); model=Generator(n_mel_channels,n_residual_layers,ratios=generator_ratio,mult=mult,out_band=out_channels)
    model.load_state_dict(ckpt['model_g']); return model.to(device).eval()

def vocgan_infer(mel,vocoder,path):
    with torch.no_grad():
        if len(mel.shape)==2: mel=mel.unsqueeze(0)
        audio=vocoder.infer(mel).squeeze(); audio=hp.max_wav_value*audio[:-(hp.hop_length*10)]
        wavfile.write(path,hp.sampling_rate,audio.clamp(-hp.max_wav_value,hp.max_wav_value-1).short().cpu().numpy())

def pad_1D(inputs,PAD=0):
    m=max(len(x) for x in inputs); return np.stack([np.pad(x,(0,m-x.shape[0]),constant_values=PAD) for x in inputs])
def pad_2D(inputs,maxlen=None):
    m=maxlen or max(x.shape[0] for x in inputs)
    return np.stack([np.pad(x,((0,m-x.shape[0]),(0,0)),constant_values=0) for x in inputs])
def pad(inputs,mel_max_length=None):
    m=mel_max_length or max(x.size(0) for x in inputs); out=[]
    for x in inputs: out.append(F.pad(x,(0,m-x.size(0))) if x.dim()==1 else F.pad(x,(0,0,0,m-x.size(0))))
    return torch.stack(out)

def standard_norm(x,mean,std,is_mel=False):
    if not is_mel: x=remove_outlier(x)
    z=np.where(x==0.0); x=(x-mean)/std; x[z]=0.; return x
def standard_norm_torch(x,mean,std):
    z=torch.where(x==0.0); x=(x-mean)/std; x[z]=0.; return x
def de_norm(x,mean,std):
    z=torch.where(x==0.0); x=mean+std*x; x[z]=0.; return x
def remove_outlier(x):
    p25,p75=np.percentile(x,25),np.percentile(x,75); lo=p25-1.5*(p75-p25); hi=p75+1.5*(p75-p25)
    x[np.logical_or(x<=lo,x>=hi)]=0.; return x
def average_by_duration(x,durs):
    c=np.cumsum(np.pad(durs,(1,0))); out=np.zeros(len(durs),dtype=np.float32)
    for i,(s,e) in enumerate(zip(c[:-1],c[1:])):
        v=x[s:e][x[s:e]!=0]; out[i]=np.mean(v) if len(v) else 0.
    return out

def plot_data(data,titles=None,filename=None):
    fig,axes=plt.subplots(len(data),1,squeeze=False); titles=titles or [None]*len(data)
    for i,(spec,pitch,energy) in enumerate(data):
        axes[i][0].imshow(spec,origin='lower',aspect='auto'); axes[i][0].set_title(titles[i] or '')
    plt.savefig(filename,dpi=200); plt.close()
