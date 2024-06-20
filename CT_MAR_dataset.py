import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# MAR data loading 
import sys
import platform
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import *
from gecatsim.reconstruction.pyfiles import recon
from ctypes import *
from numpy.ctypeslib import ndpointer
from gecatsim.pyfiles.CommonTools import *
import matplotlib.pyplot as plt
import re

def rawread(fname, dataShape, dataType):
    # dataType is for numpy, ONLY allows: 'float'/'single', 'double', 'int'/'int32', 'uint'/'uint32', 'int8', 'int16' 
    #          they are single, double, int32, uin32, int8, int16
    with open(fname, 'rb') as fin:
        data = fin.read()
    
    # https://docs.python.org/3/library/struct.html
    switcher = {'float': ['f', 4, np.single], 
                'single': ['f', 4, np.single], 
                'double': ['d', 8, np.double], 
                'int': ['i', 4, np.int32], 
                'uint': ['I', 4, np.uint32],  
                'int32': ['i', 4, np.int32], 
                'uint32': ['I', 4, np.uint32], 
                'int8': ['b', 1, np.int8], 
                'int16': ['h', 2, np.int16]}
    fmt = switcher[dataType]
    data = struct.unpack("%d%s" % (len(data)/fmt[1], fmt[0]), data) # interpret bytes as packed binary data 
    
    data = np.array(data, dtype=fmt[2])
    if dataShape:
        data = data.reshape(dataShape)
    
    return data

def sino_normalization(path):
    img = rawread(path, [1000, 900, 1], 'float')
    img = img.transpose(2,1,0)  # channel이 맨앞으로 가게 
    img_scaled = (((img - np.min(img)) / (np.max(img) - np.min(img))))
    return img_scaled

def extract_number(file_name):
    if 'sino' in file_name:
        match = re.search(r'sino(\d+)', file_name)
    elif 'img' in file_name:
        match = re.search(r'img(\d+)', file_name)
    return int(match.group(1)) if match else print("ERROR")

def load_mask(filename):
    mask = Image.open(filename)
    return mask

def load_file_names(load_dir, option):
    body_files = [splitext(file_name)[0] for file_name in listdir(load_dir) if 'body' in file_name]
    head_files = [splitext(file_name)[0] for file_name in listdir(load_dir) if 'head' in file_name]

    sorted_body_files = sorted(body_files, key=extract_number)
    sorted_head_files = sorted(head_files, key=extract_number)

    body_label = [0 for i in range(len(sorted_body_files))] # body: 0
    head_label = [1 for i in range(len(sorted_head_files))] # head: 1

    # body_path = [os.path.join(load_dir, f) for f in sorted_body_files]
    # head_path = [os.path.join(load_dir, f) for f in sorted_head_files]


    if option == 'h':
        return sorted_head_files, head_label
    elif option == 'b':
        return sorted_body_files, body_label
    else:
        return sorted_head_files + sorted_body_files, head_label + body_label


class CustomDataset(Dataset):
    def __init__(self, metal_dir: str, nometal_dir: str, scale: float = 1.0, mask_suffix: str = '', transform=None, raw: str = '', option: str = ''):
        super().__init__()
        self.metal_dir = Path(metal_dir)
        self.nometal_dir = Path(nometal_dir)

        # self.mask_dir = Path(mask_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.option = option
        self.raw = raw
        self.option = option

        # train 파일명 리스트 
        self.metal_ids, self.label = load_file_names(metal_dir, self.option)
        self.nometal_ids, _ = load_file_names(nometal_dir, self.option)

        if len(self.metal_ids) != len(self.nometal_ids):
            raise RuntimeError(f'Check the dataset! length of metal ids: {len(self.metal_ids)}, length of nometal ids: {len(self.nometal_ids)}')
        
        print(f'Creating dataset with {len(self.metal_ids)} examples')

        # self.mask_ids = ['metal_trace_{}_{}'.format(path.split('_')[1], path.split('_')[3]) for path in self.metal_ids]

    def __len__(self):
        return len(self.metal_ids)

    @staticmethod
    def preprocess(img, scale, is_mask):
        if is_mask:
            mask = np.asarray(img)
            mask = mask / 255.0
            mask = mask[:,:,np.newaxis]
            mask = mask.transpose((2,1,0)) # (1,900,1000)
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2,1,0))   # channel을 맨 앞으로 보내고, 원래 크기인 900x1000으로 바꿔서 집어넣기 (1,900,1000)
            # normalization
            img = (((img - np.min(img)) / (np.max(img) - np.min(img))))
            return img

    def __getitem__(self, idx):
        metal_name = self.metal_ids[idx]
        nometal_name = self.nometal_ids[idx]
        # mask_name = self.mask_ids[idx]
        label = self.label[idx]

        metal_file = list(self.metal_dir.glob(metal_name + '.*'))
        nometal_file = list(self.nometal_dir.glob(nometal_name + '.*'))
        # mask_file = list(self.mask_dir.glob(mask_name + '.*'))

        if self.raw=='img':
            metal_img = rawread(metal_file[0], [512,512], 'float')
            nometal_img = rawread(nometal_file[0], [512,512], 'float')
            #mask = load_mask(mask_file[0])
        else:
            metal_img = rawread(metal_file[0], [1000, 900, 1], 'float')
            nometal_img = rawread(nometal_file[0], [1000, 900, 1], 'float')
            #mask = load_mask(mask_file[0])

        metal = self.preprocess(metal_img, self.scale, is_mask=False)
        nometal = self.preprocess(nometal_img, self.scale, is_mask=False)
        #mask = self.preprocess(mask, self.scale, is_mask=True)

        if self.transform:
            metal = self.transform(torch.as_tensor(metal))
            nometal = self.transform(torch.as_tensor(nometal))
            #mask = self.transform(torch.as_tensor(mask))
        else:
            metal = torch.as_tensor(metal)
            nometal = torch.as_tensor(nometal)


        return {
            'metal': metal.float().contiguous(),
            'nometal': nometal.float().contiguous(),
            'label': label,
            'name': metal_name,
            #'mask': mask.long().contiguous()
        }
