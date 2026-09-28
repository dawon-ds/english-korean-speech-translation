import os
from data import kss
import hparams as hp

def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train: f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val: f.write(m + '\n')

def main():
    in_dir, out_dir = hp.data_path, hp.preprocessed_path
    meta, textgrid_name = hp.meta_name, hp.textgrid_name
    for name in ["mel", "alignment", "f0", "energy"]:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    if "kss" in hp.dataset:
        train, val = kss.build_from_path(in_dir, out_dir, meta)
    write_metadata(train, val, out_dir)

if __name__ == "__main__":
    main()
