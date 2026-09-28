import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer.Modules import ScaledDotProductAttention
import hparams as hp

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head*d_k)
        self.w_ks = nn.Linear(d_model, n_head*d_k)
        self.w_vs = nn.Linear(d_model, n_head*d_v)
        self.attention = ScaledDotProductAttention(np.power(d_k, .5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head*d_v, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,q,k,v,mask=None):
        dk,dv,nh=self.d_k,self.d_v,self.n_head
        b,lq,_=q.size(); _,lk,_=k.size(); _,lv,_=v.size(); residual=q
        q=self.w_qs(q).view(b,lq,nh,dk).permute(2,0,1,3).contiguous().view(-1,lq,dk)
        k=self.w_ks(k).view(b,lk,nh,dk).permute(2,0,1,3).contiguous().view(-1,lk,dk)
        v=self.w_vs(v).view(b,lv,nh,dv).permute(2,0,1,3).contiguous().view(-1,lv,dv)
        output,attn=self.attention(q,k,v,mask=mask.repeat(nh,1,1))
        output=output.view(nh,b,lq,dv).permute(1,2,0,3).contiguous().view(b,lq,-1)
        output=self.layer_norm(self.dropout(self.fc(output))+residual)
        return output,attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_in,d_hid,dropout=.1):
        super().__init__()
        self.w_1=nn.Conv1d(d_in,d_hid,hp.fft_conv1d_kernel_size[0],padding=(hp.fft_conv1d_kernel_size[0]-1)//2)
        self.w_2=nn.Conv1d(d_hid,d_in,hp.fft_conv1d_kernel_size[1],padding=(hp.fft_conv1d_kernel_size[1]-1)//2)
        self.layer_norm=nn.LayerNorm(d_in); self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        residual=x; output=x.transpose(1,2)
        output=self.w_2(F.relu(self.w_1(output))).transpose(1,2)
        return self.layer_norm(self.dropout(output)+residual)
