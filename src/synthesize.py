import os
import re
import torch
import torch.nn as nn
import numpy as np
from g2pk import G2p
from jamo import h2j

import hparams as hp
from fastspeech2 import FastSpeech2
from text import text_to_sequence
import utils
import audio as Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def kor_preprocess(text):
    phone = h2j(G2p()(text))
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sil}', phone).replace('}{', ' ')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    return torch.from_numpy(np.stack([sequence])).long().to(device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, f"checkpoint_{num}.pth.tar")
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, vocoder, text, sentence, dur_pitch_energy_aug, prefix=''):
    sentence = sentence[:10]
    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = f0_stat = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = energy_stat = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)
    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    mel, mel_postnet, _, f0_output, energy_output, _, _, _ = model(
        text, src_len, dur_pitch_energy_aug=dur_pitch_energy_aug, f0_stat=f0_stat, energy_stat=energy_stat)
    mel_postnet_torch = utils.de_norm(mel_postnet.transpose(1, 2).detach().transpose(1, 2), mean_mel, std_mel).transpose(1, 2)
    f0_output = utils.de_norm(f0_output[0], mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(energy_output[0], mean_energy, std_energy).squeeze().detach().cpu().numpy()
    os.makedirs(hp.test_path, exist_ok=True)
    Audio.tools.inv_mel_spec(mel_postnet_torch[0], os.path.join(hp.test_path, f'{prefix}_griffin_lim_{sentence}.wav'))
    if vocoder is not None and hp.vocoder.lower() == "vocgan":
        utils.vocgan_infer(mel_postnet_torch, vocoder, path=os.path.join(hp.test_path, f'{prefix}_{hp.vocoder}_{sentence}.wav'))
    utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)],
                    ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, f'{prefix}_{sentence}.png'))
