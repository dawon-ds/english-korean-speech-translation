import torch
import torch.nn as nn

def weights_init(m):
    name=m.__class__.__name__
    if "Conv" in name and hasattr(m,'weight'): m.weight.data.normal_(0.,.02)

class ResStack(nn.Module):
    def __init__(self,channel,dilation=1):
        super().__init__()
        self.block=nn.Sequential(nn.LeakyReLU(.2),nn.ReflectionPad1d(dilation),nn.utils.weight_norm(nn.Conv1d(channel,channel,3,dilation=dilation)),nn.LeakyReLU(.2),nn.utils.weight_norm(nn.Conv1d(channel,channel,1)))
        self.shortcut=nn.utils.weight_norm(nn.Conv1d(channel,channel,1))
    def forward(self,x): return self.shortcut(x)+self.block(x)

class Generator(nn.Module):
    def __init__(self,mel_channel,n_residual_layers,ratios=[4,4,2,2,2,2],mult=256,out_band=1):
        super().__init__(); self.mel_channel=mel_channel
        self.start=nn.Sequential(nn.ReflectionPad1d(3),nn.utils.weight_norm(nn.Conv1d(mel_channel,mult*2,7)))
        self.ups=nn.ModuleList(); self.res=nn.ModuleList(); self.skips=nn.ModuleList()
        in_ch=mult*2
        for i,r in enumerate(ratios):
            out_ch=in_ch//2
            self.ups.append(nn.Sequential(nn.LeakyReLU(.2),nn.utils.weight_norm(nn.ConvTranspose1d(in_ch,out_ch,r*2,r,padding=r//2+r%2,output_padding=r%2))))
            self.res.append(nn.Sequential(*[ResStack(out_ch,dilation=3**j) for j in range(n_residual_layers)]))
            if i>=2:
                stride=32*(2**(i-2)); self.skips.append(nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel,out_ch,stride*2,stride,padding=stride//2)))
            in_ch=out_ch
        self.out=nn.Sequential(nn.LeakyReLU(.2),nn.ReflectionPad1d(3),nn.utils.weight_norm(nn.Conv1d(in_ch,out_band,7)),nn.Tanh()); self.apply(weights_init)
    def forward(self,mel):
        mel=(mel+5.)/5.; x=self.start(mel); skip_i=0
        for i,(up,res) in enumerate(zip(self.ups,self.res)):
            x=up(x)
            if i>=2: x=x+self.skips[skip_i](mel); skip_i+=1
            x=res(x)
        return self.out(x)
    def remove_weight_norm(self):
        def rm(m):
            try: nn.utils.remove_weight_norm(m)
            except ValueError: pass
        self.apply(rm)
    def infer(self,mel):
        zero=torch.full((mel.size(0),self.mel_channel,10),-11.5129,device=mel.device)
        return self.forward(torch.cat((mel,zero),dim=2))
