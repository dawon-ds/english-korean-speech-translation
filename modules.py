import torch
import torch.nn as nn
from collections import OrderedDict
import hparams as hp
import utils
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VarianceAdaptor(nn.Module):
    def __init__(self):
        super().__init__(); self.duration_predictor=VariancePredictor(); self.length_regulator=LengthRegulator(); self.pitch_predictor=VariancePredictor(); self.energy_predictor=VariancePredictor()
        self.energy_embedding_producer=Conv(1,hp.encoder_hidden,kernel_size=9,bias=False,padding=4)
        self.pitch_embedding_producer=Conv(1,hp.encoder_hidden,kernel_size=9,bias=False,padding=4)
    def forward(self,x,src_mask,mel_mask=None,duration_target=None,pitch_target=None,energy_target=None,max_len=None,dur_pitch_energy_aug=None,f0_stat=None,energy_stat=None):
        log_d=self.duration_predictor(x,src_mask); pitch=self.pitch_predictor(x,src_mask)
        if pitch_target is not None: pitch_emb=self.pitch_embedding_producer(pitch_target.unsqueeze(2))
        else:
            pitch=utils.de_norm(pitch,f0_stat[0],f0_stat[1])*dur_pitch_energy_aug[1]; pitch=utils.standard_norm_torch(pitch,f0_stat[0],f0_stat[1]); pitch_emb=self.pitch_embedding_producer(pitch.unsqueeze(2))
        energy=self.energy_predictor(x,src_mask)
        if energy_target is not None: energy_emb=self.energy_embedding_producer(energy_target.unsqueeze(2))
        else:
            energy=utils.de_norm(energy,energy_stat[0],energy_stat[1])*dur_pitch_energy_aug[2]; energy=utils.standard_norm_torch(energy,energy_stat[0],energy_stat[1]); energy_emb=self.energy_embedding_producer(energy.unsqueeze(2))
        x=x+pitch_emb+energy_emb
        if duration_target is not None: x,mel_len=self.length_regulator(x,duration_target,max_len)
        else:
            duration=torch.clamp(torch.round(torch.exp(log_d)-hp.log_offset)*dur_pitch_energy_aug[0],min=0)
            x,mel_len=self.length_regulator(x,duration,max_len); mel_mask=utils.get_mask_from_lengths(mel_len)
        return x,log_d,pitch,energy,mel_len,mel_mask

class LengthRegulator(nn.Module):
    def expand(self,batch,predicted):
        out=[vec.expand(int(predicted[i].item()),-1) for i,vec in enumerate(batch)]
        return torch.cat(out,0)
    def forward(self,x,duration,max_len):
        output=[]; mel_len=[]
        for batch,target in zip(x,duration):
            expanded=self.expand(batch,target); output.append(expanded); mel_len.append(expanded.shape[0])
        output=utils.pad(output,max_len) if max_len is not None else utils.pad(output)
        return output,torch.LongTensor(mel_len).to(device)

class VariancePredictor(nn.Module):
    def __init__(self):
        super().__init__(); f=hp.variance_predictor_filter_size; k=hp.variance_predictor_kernel_size; d=hp.variance_predictor_dropout
        self.conv_layer=nn.Sequential(OrderedDict([("conv1d_1",Conv(hp.encoder_hidden,f,k,padding=(k-1)//2)),("relu_1",nn.ReLU()),("layer_norm_1",nn.LayerNorm(f)),("dropout_1",nn.Dropout(d)),("conv1d_2",Conv(f,f,k,padding=1)),("relu_2",nn.ReLU()),("layer_norm_2",nn.LayerNorm(f)),("dropout_2",nn.Dropout(d))]))
        self.linear_layer=nn.Linear(f,1)
    def forward(self,x,mask):
        out=self.linear_layer(self.conv_layer(x)).squeeze(-1)
        return out.masked_fill(mask,0.) if mask is not None else out

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=True,w_init='linear'):
        super().__init__(); self.conv=nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
    def forward(self,x): return self.conv(x.contiguous().transpose(1,2)).contiguous().transpose(1,2)
