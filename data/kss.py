import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment,remove_outlier,average_by_duration
import hparams as hp
from sklearn.preprocessing import StandardScaler

def build_from_path(in_dir,out_dir,meta):
    train,val=[],[]; scalers=[StandardScaler(copy=False) for _ in range(3)]
    with open(os.path.join(in_dir,meta),encoding='utf-8') as f:
        for index,line in enumerate(f):
            parts=line.strip().split('|'); basename=parts[0]
            ret=process_utterance(in_dir,out_dir,basename,scalers)
            if ret is None: continue
            info,_=ret; (val if basename[0]=='1' else train).append(info)
            if index%100==0: print("Done %d"%index)
    for i,name in enumerate(['mel_stat.npy','f0_stat.npy','energy_stat.npy']):
        np.save(os.path.join(out_dir,name),np.array([scalers[i].mean_,scalers[i].scale_]))
    return train,val

def process_utterance(in_dir,out_dir,basename,scalers):
    original=basename.replace('.wav',''); basename=original[2:]
    wav_bak=os.path.join(in_dir,'wavs_bak',original+'.wav'); wav_path=os.path.join(in_dir,'wavs',basename+'.wav')
    tg_path=os.path.join(out_dir,'TextGrid',basename+'.TextGrid')
    if not os.path.exists(tg_path): return None
    if not os.path.isfile(wav_path): os.system(f"ffmpeg -i {wav_bak} -ac 1 -ar 22050 {wav_path}")
    grid=tgt.io.read_textgrid(tg_path); phone,duration,start,end=get_alignment(grid.get_tier_by_name('phones'))
    text='{'+ '}{'.join(phone)+'}'; text=text.replace('{$}',' ').replace('}{',' ')
    if start>=end: return None
    _,wav=read(wav_path); wav=wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)
    f0,_=pw.dio(wav.astype(np.float64),hp.sampling_rate,frame_period=hp.hop_length/hp.sampling_rate*1000); f0=f0[:sum(duration)]
    mel,energy=Audio.tools.get_mel_from_wav(torch.FloatTensor(wav)); mel=mel.numpy().astype(np.float32)[:,:sum(duration)]; energy=energy.numpy().astype(np.float32)[:sum(duration)]
    f0,energy=average_by_duration(remove_outlier(f0),duration),average_by_duration(remove_outlier(energy),duration)
    if mel.shape[1]>=hp.max_seq_len: return None
    np.save(os.path.join(out_dir,'alignment',f'{hp.dataset}-ali-{basename}.npy'),duration,allow_pickle=False)
    np.save(os.path.join(out_dir,'f0',f'{hp.dataset}-f0-{basename}.npy'),f0,allow_pickle=False)
    np.save(os.path.join(out_dir,'energy',f'{hp.dataset}-energy-{basename}.npy'),energy,allow_pickle=False)
    np.save(os.path.join(out_dir,'mel',f'{hp.dataset}-mel-{basename}.npy'),mel.T,allow_pickle=False)
    scalers[0].partial_fit(mel.T); scalers[1].partial_fit(f0[f0!=0].reshape(-1,1)); scalers[2].partial_fit(energy[energy!=0].reshape(-1,1))
    return '|'.join([basename,text]),mel.shape[1]
