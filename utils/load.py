import torch
import nibabel as nib
import os
from torchvision import transforms
import numpy as np

RESIZE = transforms.Resize((256,256))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH = '/home/ubuntu/multiversion-unet/data/training'
TEST_DATA_PATH = '/home/ubuntu/multiversion-unet/data/testing'

def processNibImg(path):
    img = nib.load(path).dataobj[:,:,:]
    tensor_img = torch.tensor(img, dtype=torch.float32)
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.permute(3,0,1,2)
    tensor_img = RESIZE(tensor_img)
    tensor_img = tensor_img.expand(tensor_img.shape[0],3,tensor_img.shape[2],tensor_img.shape[3])

    return tensor_img.to(DEVICE)

def load():
    train_data = {
        'train': [],
        'valid': [],
    }

    test_data = {
        'train': [],
        'valid': [],
    }

    for (dirpath, dirnames, filenames) in os.walk(TRAIN_DATA_PATH):
        filenames.sort()
        for filename in filenames:
            # len(filename.split('_')) > 2, ground truth img
            if(filename.endswith('.nii.gz') and len(filename.split('_')) > 2 and filename.split('_')[1] != '4d.nii.gz'):
                train_data['valid'].append(processNibImg(os.path.join(dirpath, filename)))
            # len(filename.split('_')) == 2, input img
            elif(filename.endswith('.nii.gz') and len(filename.split('_')) == 2 and filename.split('_')[1] != '4d.nii.gz'):
                train_data['train'].append(processNibImg(os.path.join(dirpath, filename)))
    
    for (dirpath, dirnames, filenames) in os.walk(TEST_DATA_PATH):
        filenames.sort()
        for filename in filenames:
            # len(filename.split('_')) > 2, ground truth img
            if(filename.endswith('.nii.gz') and len(filename.split('_')) > 2 and filename.split('_')[1] != '4d.nii.gz'):
                test_data['valid'].append(processNibImg(os.path.join(dirpath, filename)))
            # len(filename.split('_')) == 2, input img
            elif(filename.endswith('.nii.gz') and len(filename.split('_')) == 2 and filename.split('_')[1] != '4d.nii.gz'):
                test_data['train'].append(processNibImg(os.path.join(dirpath, filename)))

    return train_data, test_data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,ifTrain=True):
        train_data, test_data = load()
        if(ifTrain):
            self.train = True
            self.train_data = train_data['train']
            self.train_label = train_data['valid']
        else:
            self.train = False
            self.train_data = test_data['train']
            self.train_label = test_data['valid']

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if(self.train):
            return self.train_data[idx], self.train_label[idx],idx
        else:
            return self.train_data[idx],idx