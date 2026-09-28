import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hparams as hp
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import utils

def main(args):
    torch.manual_seed(0); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=Dataset("train.txt"); loader=DataLoader(dataset,batch_size=hp.batch_size**2,shuffle=True,collate_fn=dataset.collate_fn,drop_last=True,num_workers=0)
    model=nn.DataParallel(FastSpeech2()).to(device); print("Number of FastSpeech2 Parameters:",utils.get_param_num(model))
    optimizer=torch.optim.Adam(model.parameters(),betas=hp.betas,eps=hp.eps,weight_decay=hp.weight_decay)
    sched=ScheduledOptim(optimizer,hp.decoder_hidden,hp.n_warm_up_step,args.restore_step); criterion=FastSpeech2Loss().to(device)
    os.makedirs(hp.checkpoint_path,exist_ok=True); os.makedirs(hp.log_path,exist_ok=True)
    ckpt=os.path.join(hp.checkpoint_path,f"checkpoint_{args.restore_step}.pth.tar")
    if args.restore_step and os.path.exists(ckpt):
        state=torch.load(ckpt,map_location=device); model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    vocoder=utils.get_vocgan(hp.vocoder_pretrained_model_path) if hp.vocoder=='vocgan' else None
    logger=SummaryWriter(os.path.join(hp.log_path,'train')); model.train()
    for epoch in range(hp.epochs):
        for i,batches in enumerate(loader):
            for j,b in enumerate(batches):
                step=i*hp.batch_size+j+args.restore_step+epoch*len(loader)*hp.batch_size+1
                text=torch.from_numpy(b['text']).long().to(device); mel=torch.from_numpy(b['mel_target']).float().to(device)
                D=torch.from_numpy(b['D']).long().to(device); log_D=torch.from_numpy(b['log_D']).float().to(device)
                f0=torch.from_numpy(b['f0']).float().to(device); energy=torch.from_numpy(b['energy']).float().to(device)
                src_len=torch.from_numpy(b['src_len']).long().to(device); mel_len=torch.from_numpy(b['mel_len']).long().to(device)
                out=model(text,src_len,mel_len,D,f0,energy,int(np.max(b['src_len'])),int(np.max(b['mel_len'])))
                losses=criterion(out[2],log_D,out[3],f0,out[4],energy,out[0],out[1],mel,~out[5],~out[6]); total=sum(losses)/hp.acc_steps; total.backward()
                if step%hp.acc_steps: continue
                nn.utils.clip_grad_norm_(model.parameters(),hp.grad_clip_thresh); sched.step_and_update_lr(); sched.zero_grad()
                logger.add_scalar('Loss/total_loss',float(total.item()*hp.acc_steps),step)
                if step%hp.save_step==0: torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},os.path.join(hp.checkpoint_path,f'checkpoint_{step}.pth.tar'))
                if step%hp.eval_step==0:
                    model.eval()
                    with torch.no_grad(): evaluate(model,step,vocoder)
                    model.train()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--restore_step',type=int,default=0); main(p.parse_args())
