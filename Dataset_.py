import pandas as pd 
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import  Subset
from PIL import Image, ImageFilter , ImageOps
from torchvision import transforms
import torch 
import cv2
import random




def is_valid_string(s):
    allowed_chars = {'A', 'T', 'C', 'G'}
    return all(char in allowed_chars for char in s)

def one_hot_encode(sequence):
    mapping = {'Z':0, 'A':1, 'T':2, 'C':3, 'G':4,
                'N':5, 'K':6, 'W':7, 'M':8, 'S':9, 'Y':10,'R':11}
    sequence = sequence.replace("\n", "")
    
    integer_encoded = [mapping[base] for base in sequence]
    
    one_hot_encoded = np.zeros((2000, len(mapping)))
    
    for idx, value in enumerate(integer_encoded):
        one_hot_encoded[idx, value] = 1
        
    return one_hot_encoded

def split_Data(filename):
    df = pd.read_excel(filename)

    num_groups = 10
    np.random.seed()
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    grouped_data = np.array_split(shuffled_df, num_groups)
    for i, group in enumerate(grouped_data):
        group.to_excel('./DataExcel/group_%d.xlsx'%(i), index=False)


def remove_random_characters(input_str):

    indices_to_remove = set(random.sample(range(1866), 366))
    result = ''.join([char for idx, char in enumerate(input_str) if idx not in indices_to_remove])
    
    return result
     
class Dataset_y(Dataset):
    def __init__(self ,  dir_data ,Test_Num = 0, train_mode = True):
        self.train_mode = train_mode
        self.dir = dir_data
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                           transforms.ToTensor()])
        
        _data = []
        if train_mode == True:
            for i in range(10):
                if i != Test_Num :
                    df = pd.read_excel('./DataExcel/group_%d.xlsx'%(i), 
                                header=0,
                                usecols=[ 'lab' , 'img' , 'seqs'],
                                dtype={'lab':np.int32,
                                        'img':str,
                                        'seqs':str,
                                        })
                    _data.append(df)
            data = pd.concat(_data , ignore_index=True)
        else :
            data = df = pd.read_excel('./DataExcel/group_%d.xlsx'%(Test_Num), 
                                header=0,
                                usecols=[ 'lab' , 'img' , 'seqs'],
                                dtype={'lab':np.int32,
                                        'img':str,
                                        'seqs':str,
                                        })
        pic = data.iloc[: , 1].values
        seq = data.iloc[: , 2].values
        label = data.iloc[: , 0].values
        self.x = pic
        self.s = seq
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)    
    
    def __getitem__(self, idx):
        if os.path.exists(self.dir + self.x[idx] + ".jpeg"):
            img = Image.open(self.dir + self.x[idx] + ".jpeg")
            img = img.convert('RGB')

        elif os.path.exists(self.dir + self.x[idx] + ".jpg"):
            img = Image.open(self.dir + self.x[idx] + ".jpg")
            img = img.convert('RGB')

        elif os.path.exists(self.dir + self.x[idx] + ".png"):
            img = Image.open(self.dir + self.x[idx] + ".png")
            img = img.convert('RGB')
        
        w = img.size[0]
        h = img.size[1]
        if w > h:
            border = (0, (w - h) // 2, 0, (w - h) // 2)
        else:
            border = ((h - w) // 2, 0, (h - w) // 2, 0)

        img = ImageOps.expand(img, border=border, fill="white")
        if self.train_mode == True:
            rnd = np.random.random_sample()
            if rnd < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            rnd = np.random.random_sample()
            if rnd < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = self.transform(img)

        if self.train_mode == True:
            seq1 = remove_random_characters(self.s[idx])
            seq = one_hot_encode(seq1)
        else :
            seq = one_hot_encode(self.s[idx])


        return  img , seq , self.y[idx] 


        
    


    
    
    
    
    





    
    

    
    












