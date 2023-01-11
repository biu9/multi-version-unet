from re import I
import torch
import nibabel as nib
import os
from model.Unet import UNet
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from utils.load import MyDataset
from matplotlib import pyplot as plt

MODEL_FINAL_PATH = '/home/ubuntu/multi-version-unet/model/result/final/epoch_750.pth' 
MODEL_FULL_PATH = '/home/ubuntu/multi-version-unet/model/result/full/epoch_1000.pth' # model trained with full train img
MODEL_BASELINE_PATH = '/home/ubuntu/multi-version-unet/model/result/final_baseline/epoch_1000.pth'
TEST_DATA_PATH = '/home/ubuntu/multiversion-unet/data/testing'
TRAIN_DATA_PATH = '/home/ubuntu/multiversion-unet/data/training'

SEGMENTATION_RESULT_BEFORE_PATH = '/home/ubuntu/multi-version-unet/tmpSave/beforeResult'
SEGMENTATION_RESULT_AFTER_PATH = '/home/ubuntu/multi-version-unet/tmpSave/afterResult'
FINAL_RESULT_PATH = '/home/ubuntu/multi-version-unet/tmpSave/finalResult'

AFTER_PLOT_SAVE_PATH = '/home/ubuntu/multi-version-unet/tmpSave/afterVisualize'
BEFORE_PLOT_SAVE_PATH = '/home/ubuntu/multi-version-unet/tmpSave/beforeVisualize'

RESIZE = transforms.Resize((256,256))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

if __name__ == '__main__':
    model = UNet(3,3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FINAL_PATH))
    model.eval()

    for (dirpath, dirnames, filenames) in os.walk(TEST_DATA_PATH):
        filenames.sort()
        for filename in filenames:
            if 'frame' in filename:
                img = nib.load(os.path.join(dirpath, filename))
                header = img.header
                affine = img.affine
                img = img.get_fdata()
                img = img.astype(np.float32)
                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                img = img.permute(3,0,1,2)
                initSize = img.shape
                tmpResize = transforms.Resize((img.shape[2],img.shape[3]))
                img = RESIZE(img)
                img = img.expand(img.shape[0],3,img.shape[2],img.shape[3])
                if(filename.endswith('.nii.gz') and len(filename.split('_')) == 2 and filename.split('_')[1] != '4d.nii.gz'):
                    res = model(img.to(DEVICE))
                    saveImage = torch.where(res[:,0,:,:] > 0.5, 1, 0) + torch.where(res[:,1,:,:] > 0.5, 2, 0) + torch.where(res[:,2,:,:] > 0.5, 3, 0)
                    saveImage = tmpResize(saveImage)
                    saveImage = saveImage.permute(1,2,0)
                    saveImage = saveImage.detach().cpu().numpy()
                    if('01' in filename.split('_')[1]):
                        saveFileame = filename.split('_')[0] + '_ED.nii.gz'
                        save_nii(os.path.join(SEGMENTATION_RESULT_AFTER_PATH, saveFileame), saveImage, affine, header)
                    else:
                        saveFileame = filename.split('_')[0] + '_ES.nii.gz'
                        save_nii(os.path.join(SEGMENTATION_RESULT_AFTER_PATH, saveFileame), saveImage, affine, header)

