import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class FFTBlock(torch.nn.Module):
    def __init__(self,d_model,d_inner,n_head,d_k,d_v,dropout=.1):
        super().__init__()
        self.slf_attn=MultiHeadAttention(n_head,d_model,d_k,d_v,dropout)
        self.pos_ffn=PositionwiseFeedForward(d_model,d_inner,dropout)
    def forward(self,enc_input,mask=None,slf_attn_mask=None):
        enc_output,attn=self.slf_attn(enc_input,enc_input,enc_input,mask=slf_attn_mask)
        enc_output=enc_output.masked_fill(mask.unsqueeze(-1),0)
        enc_output=self.pos_ffn(enc_output).masked_fill(mask.unsqueeze(-1),0)
        return enc_output,attn

class ConvNorm(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=None,dilation=1,bias=True,w_init_gain='linear'):
        super().__init__()
        if padding is None: padding=int(dilation*(kernel_size-1)/2)
        self.conv=torch.nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
    def forward(self,signal): return self.conv(signal)

class PostNet(nn.Module):
    def __init__(self,n_mel_channels=80,postnet_embedding_dim=512,postnet_kernel_size=5,postnet_n_convolutions=5):
        super().__init__(); self.convolutions=nn.ModuleList()
        self.convolutions.append(nn.Sequential(ConvNorm(n_mel_channels,postnet_embedding_dim,postnet_kernel_size,padding=(postnet_kernel_size-1)//2),nn.BatchNorm1d(postnet_embedding_dim)))
        for _ in range(1,postnet_n_convolutions-1):
            self.convolutions.append(nn.Sequential(ConvNorm(postnet_embedding_dim,postnet_embedding_dim,postnet_kernel_size,padding=(postnet_kernel_size-1)//2),nn.BatchNorm1d(postnet_embedding_dim)))
        self.convolutions.append(nn.Sequential(ConvNorm(postnet_embedding_dim,n_mel_channels,postnet_kernel_size,padding=(postnet_kernel_size-1)//2),nn.BatchNorm1d(n_mel_channels)))
    def forward(self,x):
        x=x.contiguous().transpose(1,2)
        for conv in self.convolutions[:-1]: x=F.dropout(torch.tanh(conv(x)),.5,self.training)
        x=F.dropout(self.convolutions[-1](x),.5,self.training)
        return x.contiguous().transpose(1,2)
