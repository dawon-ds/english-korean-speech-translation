import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import FFTBlock
from text.symbols import symbols
import hparams as hp

def get_sinusoid_encoding_table(n_position,d_hid,padding_idx=None):
    def angle(pos,i): return pos/np.power(10000,2*(i//2)/d_hid)
    table=np.array([[angle(p,i) for i in range(d_hid)] for p in range(n_position)])
    table[:,0::2]=np.sin(table[:,0::2]); table[:,1::2]=np.cos(table[:,1::2])
    if padding_idx is not None: table[padding_idx]=0.
    return torch.FloatTensor(table)

class Encoder(nn.Module):
    def __init__(self,n_src_vocab=len(symbols)+1,len_max_seq=hp.max_seq_len,d_word_vec=hp.encoder_hidden,n_layers=hp.encoder_layer,n_head=hp.encoder_head,d_k=hp.encoder_hidden//hp.encoder_head,d_v=hp.encoder_hidden//hp.encoder_head,d_model=hp.encoder_hidden,d_inner=hp.fft_conv1d_filter_size,dropout=hp.encoder_dropout):
        super().__init__(); n_position=len_max_seq+1
        self.src_word_emb=nn.Embedding(n_src_vocab,d_word_vec,padding_idx=Constants.PAD)
        self.position_enc=nn.Parameter(get_sinusoid_encoding_table(n_position,d_word_vec).unsqueeze(0),requires_grad=False)
        self.layer_stack=nn.ModuleList([FFTBlock(d_model,d_inner,n_head,d_k,d_v,dropout) for _ in range(n_layers)])
    def forward(self,src_seq,mask,return_attns=False):
        b,max_len=src_seq.shape[:2]; slf=mask.unsqueeze(1).expand(-1,max_len,-1)
        pos=(get_sinusoid_encoding_table(max_len,hp.encoder_hidden).unsqueeze(0).expand(b,-1,-1).to(src_seq.device) if not self.training and max_len>hp.max_seq_len else self.position_enc[:,:max_len,:].expand(b,-1,-1))
        out=self.src_word_emb(src_seq)+pos
        for layer in self.layer_stack: out,_=layer(out,mask=mask,slf_attn_mask=slf)
        return out

class Decoder(nn.Module):
    def __init__(self,len_max_seq=hp.max_seq_len,d_word_vec=hp.encoder_hidden,n_layers=hp.decoder_layer,n_head=hp.decoder_head,d_k=hp.decoder_hidden//hp.decoder_head,d_v=hp.decoder_hidden//hp.decoder_head,d_model=hp.decoder_hidden,d_inner=hp.fft_conv1d_filter_size,dropout=hp.decoder_dropout):
        super().__init__(); self.position_enc=nn.Parameter(get_sinusoid_encoding_table(len_max_seq+1,d_word_vec).unsqueeze(0),requires_grad=False)
        self.layer_stack=nn.ModuleList([FFTBlock(d_model,d_inner,n_head,d_k,d_v,dropout) for _ in range(n_layers)])
    def forward(self,enc_seq,mask,return_attns=False):
        b,max_len=enc_seq.shape[:2]; slf=mask.unsqueeze(1).expand(-1,max_len,-1)
        pos=(get_sinusoid_encoding_table(max_len,hp.decoder_hidden).unsqueeze(0).expand(b,-1,-1).to(enc_seq.device) if not self.training and max_len>hp.max_seq_len else self.position_enc[:,:max_len,:].expand(b,-1,-1))
        out=enc_seq+pos
        for layer in self.layer_stack: out,_=layer(out,mask=mask,slf_attn_mask=slf)
        return out
