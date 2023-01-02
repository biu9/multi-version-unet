from model.unet_encoder import UNetEncoder
from re import I
import torch
import nibabel as nib
import os
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from utils.load import MyDataset
from torch.utils.data import DataLoader

TRAIN_DATA_PATH = '/home/ubuntu/multiversion-unet/data/training'
TEST_DATA_PATH = '/home/ubuntu/multiversion-unet/data/testing'
ENCODER_PATH = '/home/ubuntu/multi-version-unet/model/result/encoder'

RESIZE = transforms.Resize((256,256))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 100

if __name__ == '__main__':
    model = UNetEncoder(3,3).to(DEVICE)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    lossFunc=  nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = MyDataset()
    test_data = MyDataset(ifTrain=False)
    train_loader = DataLoader(train_data,
                            shuffle=True,
                            batch_size=1)
    test_loader = DataLoader(test_data,
                            shuffle=True,
                            batch_size=1)
    minLoss = 100
    for i in tqdm(range(EPOCH)):
        lossItem = []
        for iteration, (train_img, train_label,index) in enumerate(train_loader):
            train_img = train_img.to(device)
            train_label = train_label.to(device)
            train_img = train_img.squeeze(0)

            encoded,decoded = model(train_img)
            loss = lossFunc(decoded, train_img)

            if iteration % 20 == 0:
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossItem.append(loss.item())

        print(f'epoch: {i}, loss: {sum(lossItem)/len(lossItem)}')
        if i % 10 == 0:
            if loss.item() < minLoss:
                minLoss = loss.item()
                torch.save(model.state_dict(), os.path.join(ENCODER_PATH, 'encoder_'+str(i)+'.pth'))