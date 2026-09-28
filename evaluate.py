import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
import hparams as hp
import utils
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_FastSpeech2(step):
    model=nn.DataParallel(FastSpeech2()).to(device)
    state=torch.load(os.path.join(hp.checkpoint_path,f'checkpoint_{step}.pth.tar'),map_location=device)
    model.load_state_dict(state['model']); model.eval(); return model

def evaluate(model,step,vocoder=None):
    dataset=Dataset('val.txt',sort=False); loader=DataLoader(dataset,batch_size=hp.batch_size**2,shuffle=False,collate_fn=dataset.collate_fn,num_workers=0)
    criterion=FastSpeech2Loss().to(device); values=[[] for _ in range(5)]
    for batches in loader:
        for b in batches:
            text=torch.from_numpy(b['text']).long().to(device); mel=torch.from_numpy(b['mel_target']).float().to(device)
            D=torch.from_numpy(b['D']).long().to(device); log_D=torch.from_numpy(b['log_D']).float().to(device)
            f0=torch.from_numpy(b['f0']).float().to(device); energy=torch.from_numpy(b['energy']).float().to(device)
            src_len=torch.from_numpy(b['src_len']).long().to(device); mel_len=torch.from_numpy(b['mel_len']).long().to(device)
            with torch.no_grad():
                out=model(text,src_len,mel_len,D,f0,energy,int(np.max(b['src_len'])),int(np.max(b['mel_len'])))
                losses=criterion(out[2],log_D,out[3],f0,out[4],energy,out[0],out[1],mel,~out[5],~out[6])
            for a,v in zip(values,losses): a.append(v.item())
    means=[sum(x)/len(x) for x in values]; print(f"FastSpeech2 Step {step}: mel={means[0]:.4f}, postnet={means[1]:.4f}, duration={means[2]:.4f}, f0={means[3]:.4f}, energy={means[4]:.4f}")
    model.train(); return means[2],means[3],means[4],means[0],means[1]

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--step',type=int,default=30000); args=p.parse_args()
    model=get_FastSpeech2(args.step); vocoder=utils.get_vocgan(hp.vocoder_pretrained_model_path) if hp.vocoder=='vocgan' else None
    evaluate(model,args.step,vocoder)
