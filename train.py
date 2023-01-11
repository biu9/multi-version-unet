import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import shutil
import os
from model.Unet import UNet
from utils.load import MyDataset
from utils.dice_loss import DiceLoss
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import kmeans1d
import SimpleITK as sitk
from model.unet_encoder import UNetEncoder

TOTAL_SIM_PATH = '/home/ubuntu/multi-version-unet/tmpSave/totalSim.txt'
MODEL_FULL_PATH = '/home/ubuntu/multi-version-unet/model/result/full/epoch_300.pth'
MODEL_INIT1_PATH = '/home/ubuntu/multi-version-unet/model/result/init_1/epoch_300.pth'
MODEL_INIT2_PATH = '/home/ubuntu/multi-version-unet/model/result/init_2/epoch_300.pth'
MODEL_CANNY1_PATH = '/home/ubuntu/multi-version-unet/model/result/canny_1/epoch_300.pth'
MODEL_CANNY2_PATH = '/home/ubuntu/multi-version-unet/model/result/canny_2/epoch_300.pth'
PSEUDO_PATH = '/home/ubuntu/multi-version-unet/tmpSave/pseudoLabel'
ENCODER_PATH = '/home/ubuntu/multi-version-unet/model/result/encoder/encoder_200.pth'

def trainEncoder(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetEncoder(3,3).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    lossFunc=  nn.MSELoss()
    train_data = MyDataset()
    test_data = MyDataset(ifTrain=False)
    train_loader = DataLoader(train_data,
                            shuffle=True,
                            batch_size=1)
    test_loader = DataLoader(test_data,
                            shuffle=True,
                            batch_size=1)
    minLoss = 100
    for epoch in tqdm(range(201)):
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

        print(f'epoch: {epoch}, loss: {sum(lossItem)/len(lossItem)}')
        if (epoch%10==0):
            torch.save(model.state_dict(),'/home/ubuntu/multi-version-unet/model/result/encoder/encoder_'+str(epoch)+'.pth')

def calSimilarity(train_loader):
    allTrainSim = []
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    autoEncoder = UNetEncoder(3,3).to(device)
    autoEncoder.load_state_dict(torch.load(ENCODER_PATH))
    with open(TOTAL_SIM_PATH, 'w+') as f:
        for iteration_i, (train_img_i, train_label,i) in tqdm(enumerate(train_loader)):
            oneImgSim = 0 # similarity of i index img with all other imgs
            train_img_i = train_img_i.squeeze(0)
            train_img_i = train_img_i.to(device)
            for iteration_j, (train_img_j, train_label,j) in enumerate(train_loader):
                train_img_j = train_img_j.squeeze(0)
                train_img_j = train_img_j.to(device)
                encoded1,decoded1 = autoEncoder(train_img_i)
                encoded2,decoded2 = autoEncoder(train_img_j)
                sim = torch.cosine_similarity(encoded1[0,:,:,:],encoded2[0,:,:,:],dim=0)
                avg = torch.mean(sim)
                oneImgSim += avg.item()
            allTrainSim.append(oneImgSim)
        f.write(str(allTrainSim))

def cannyProcess(img):
    for i in range(img.shape[0]):
        tmp = img[i,0,:,:].cpu().numpy()
        tmp = tmp.astype('uint8')
        tmp = cv2.Canny(tmp, 50, 200)
        tmp = torch.tensor(tmp, dtype=torch.float32)
        img[i,0,:,:] = tmp
    return img

def trainFullModel(train_loader,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3,3).to(device)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 301

    loss_item = []
    loss_RV_item = []
    loss_LV_item = []
    loss_MYO_item = []
    test_loss_item = []
    test_loss_RV_item = []
    test_loss_LV_item = []
    test_loss_MYO_item = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_RV = 0
        total_loss_LV = 0
        total_loss_MYO = 0
        test_total_loss = 0
        test_total_loss_RV = 0
        test_total_loss_LV = 0
        test_total_loss_MYO = 0
        model.train()

        for iteration, (train_img, train_label,index) in enumerate(train_loader):
            res = model(train_img.squeeze(0))
            train_label = train_label.squeeze(0)

            RV_train_label = train_label[:,0,:,:]
            RV_train_label = torch.where(RV_train_label==1, 1, 0)
            LV_train_label = train_label[:,2,:,:]
            LV_train_label = torch.where(LV_train_label==3, 1, 0)
            MYO_train_label = train_label[:,1,:,:]
            MYO_train_label = torch.where(MYO_train_label==2, 1, 0)
            loss_RV = loss_fn(res[:,0,:,:], RV_train_label)
            loss_LV = loss_fn(res[:,2,:,:], LV_train_label)
            loss_MYO = loss_fn(res[:,1,:,:], MYO_train_label)
            loss = loss_RV + loss_LV + loss_MYO
            
            total_loss += loss.item()
            total_loss_RV += loss_RV.item()
            total_loss_LV += loss_LV.item()
            total_loss_MYO += loss_MYO.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_item.append(total_loss/len(train_loader))
        loss_RV_item.append(total_loss_RV/len(train_loader))
        loss_LV_item.append(total_loss_LV/len(train_loader))
        loss_MYO_item.append(total_loss_MYO/len(train_loader))
        print(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}')

        '''
        model.eval()
        for iteration, (test_img, test_label,idx) in enumerate(test_loader):
            res = model(test_img.squeeze(0))
            test_label = test_label.squeeze(0)  

            RV_test_label = test_label[:,0,:,:]
            RV_test_label = torch.where(RV_test_label==1, 1, 0)
            LV_test_label = test_label[:,2,:,:]
            LV_test_label = torch.where(LV_test_label==3, 1, 0)
            MYO_test_label = test_label[:,1,:,:]
            MYO_test_label = torch.where(MYO_test_label==2, 1, 0)

            loss_RV = loss_fn(res[:,0,:,:], RV_test_label)
            loss_LV = loss_fn(res[:,2,:,:], LV_test_label)
            loss_MYO = loss_fn(res[:,1,:,:], MYO_test_label)
            loss = loss_RV + loss_LV + loss_MYO

            test_total_loss += loss.item()
            test_total_loss_RV += loss_RV.item()
            test_total_loss_LV += loss_LV.item()
            test_total_loss_MYO += loss_MYO.item()
        
        test_loss_item.append(test_total_loss/len(test_loader))
        test_loss_RV_item.append(test_total_loss_RV/len(test_loader))
        test_loss_LV_item.append(test_total_loss_LV/len(test_loader))
        test_loss_MYO_item.append(test_total_loss_MYO/len(test_loader))
        '''
        # write into log file
        with open ('/home/ubuntu/multi-version-unet/model/result/full/loss.txt','a') as f:
            f.write(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}\n')
        
        if(epoch%10==0):
            torch.save(model.state_dict(), '/home/ubuntu/multi-version-unet/model/result/full/epoch_{}.pth'.format(epoch))

def trainModel(train_loader,test_loader,if_preprocess,foldername,version):

    with open(TOTAL_SIM_PATH,'r') as f:
        # read total similarity
        totalSim = f.read()
        totalSim = totalSim.strip('[').strip(']').split(',')
        totalSim = [float(i) for i in totalSim]
        cluster, centroids = kmeans1d.cluster(totalSim, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3,3).to(device)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 301

    loss_item = []
    loss_RV_item = []
    loss_LV_item = []
    loss_MYO_item = []
    test_loss_item = []
    test_loss_RV_item = []
    test_loss_LV_item = []
    test_loss_MYO_item = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_RV = 0
        total_loss_LV = 0
        total_loss_MYO = 0
        test_total_loss = 0
        test_total_loss_RV = 0
        test_total_loss_LV = 0
        test_total_loss_MYO = 0
        model.train()

        for iteration, (train_img, train_label,index) in enumerate(train_loader):

            if(cluster[index] == version):
                train_img = train_img.squeeze(0)
                train_label = train_label.squeeze(0)
                if(if_preprocess):
                    train_img = cannyProcess(train_img)

                res = model(train_img)

                RV_train_label = train_label[:,0,:,:]
                RV_train_label = torch.where(RV_train_label==1, 1, 0)
                LV_train_label = train_label[:,2,:,:]
                LV_train_label = torch.where(LV_train_label==3, 1, 0)
                MYO_train_label = train_label[:,1,:,:]
                MYO_train_label = torch.where(MYO_train_label==2, 1, 0)
                loss_RV = loss_fn(res[:,0,:,:], RV_train_label)
                loss_LV = loss_fn(res[:,2,:,:], LV_train_label)
                loss_MYO = loss_fn(res[:,1,:,:], MYO_train_label)
                loss = loss_RV + loss_LV + loss_MYO
                
                total_loss += loss.item()
                total_loss_RV += loss_RV.item()
                total_loss_LV += loss_LV.item()
                total_loss_MYO += loss_MYO.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_item.append(total_loss/len(train_loader))
        loss_RV_item.append(total_loss_RV/len(train_loader))
        loss_LV_item.append(total_loss_LV/len(train_loader))
        loss_MYO_item.append(total_loss_MYO/len(train_loader))
        print(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}')

        '''
        model.eval()
        for iteration, (test_img, test_label,idx) in enumerate(test_loader):
            res = model(test_img.squeeze(0))
            test_label = test_label.squeeze(0)  

            RV_test_label = test_label[:,0,:,:]
            RV_test_label = torch.where(RV_test_label==1, 1, 0)
            LV_test_label = test_label[:,2,:,:]
            LV_test_label = torch.where(LV_test_label==3, 1, 0)
            MYO_test_label = test_label[:,1,:,:]
            MYO_test_label = torch.where(MYO_test_label==2, 1, 0)

            loss_RV = loss_fn(res[:,0,:,:], RV_test_label)
            loss_LV = loss_fn(res[:,2,:,:], LV_test_label)
            loss_MYO = loss_fn(res[:,1,:,:], MYO_test_label)
            loss = loss_RV + loss_LV + loss_MYO

            test_total_loss += loss.item()
            test_total_loss_RV += loss_RV.item()
            test_total_loss_LV += loss_LV.item()
            test_total_loss_MYO += loss_MYO.item()
        
        test_loss_item.append(test_total_loss/len(test_loader))
        test_loss_RV_item.append(test_total_loss_RV/len(test_loader))
        test_loss_LV_item.append(test_total_loss_LV/len(test_loader))
        test_loss_MYO_item.append(test_total_loss_MYO/len(test_loader))
        '''
        with open('/home/ubuntu/multi-version-unet/model/result/'+foldername+'/loss.txt', 'a') as f:
            f.write(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}\n')
        
        if(epoch%10==0):
            torch.save(model.state_dict(), '/home/ubuntu/multi-version-unet/model/result/'+foldername+'/epoch_{}.pth'.format(epoch))

def loadModels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullModel = UNet(3,3).to(device)
    fullModel.load_state_dict(torch.load(MODEL_FULL_PATH))
    initModel1 = UNet(3,3).to(device)
    initModel1.load_state_dict(torch.load(MODEL_INIT1_PATH))
    initModel2 = UNet(3,3).to(device)
    initModel2.load_state_dict(torch.load(MODEL_INIT2_PATH))
    cannyModel1 = UNet(3,3).to(device)
    cannyModel1.load_state_dict(torch.load(MODEL_CANNY1_PATH))
    cannyModel2 = UNet(3,3).to(device)
    cannyModel2.load_state_dict(torch.load(MODEL_CANNY2_PATH))

    fullModel.eval()
    initModel1.eval()
    initModel2.eval()
    cannyModel1.eval()
    cannyModel2.eval()
    return fullModel,initModel1,initModel2,cannyModel1,cannyModel2

def producePseudoLabelNew(test_loader,train_loader):

    device = torch.device('cpu')
    autoEncoder = UNetEncoder(3,3).to(device)
    autoEncoder.load_state_dict(torch.load(ENCODER_PATH))
    fullModel,initModel1,initModel2,cannyModel1,cannyModel2 = loadModels()
    initModel1 = initModel1.to(device)
    initModel2 = initModel2.to(device)
    cannyModel1 = cannyModel1.to(device)
    cannyModel2 = cannyModel2.to(device)

    with open(TOTAL_SIM_PATH,'r') as f:
        # read total similarity
        totalSim = f.read()
        totalSim = totalSim.strip('[').strip(']').split(',')
        totalSim = [float(i) for i in totalSim]
        cluster, centroids = kmeans1d.cluster(totalSim, 2)

    for iteration, (test_img,index) in tqdm(enumerate(test_loader)):
        test_img = test_img.squeeze(0).to(device)
        oneImgSim = 0 # similarity of i index img with all other imgs
        for iteration_train, (train_img, train_label,index_train) in (enumerate(train_loader)):
            train_img = train_img.squeeze(0).to(device)
            test_encoded,test_decoded = autoEncoder(test_img) 
            train_encoded,train_decoded = autoEncoder(train_img)
            sim = torch.cosine_similarity(test_encoded[0,:,:,:],train_encoded[0,:,:,:],dim=0)
            avg = torch.mean(sim)
            oneImgSim += avg.item()
    
        if(abs(oneImgSim - centroids[0]) < abs(oneImgSim - centroids[1])):
            rand = random.randint(0,1)
            if(rand == 0):
                res = initModel1(test_img)
            else:
                res = cannyModel1(test_img)
            resultImage = torch.where(res[:,0,:,:]>0.5, 1, 0) + torch.where(res[:,1,:,:]>0.5, 2, 0) + torch.where(res[:,2,:,:]>0.5, 3, 0)
            resultImage = resultImage.unsqueeze(dim=1)
            out = sitk.GetImageFromArray(resultImage.cpu().detach().numpy())
            sitk.WriteImage(out, PSEUDO_PATH + '/pseudoLabel/' + str(index.item()) + '.nii.gz')
        else:
            rand = random.randint(0,1)
            if(rand == 0):
                res = initModel2(test_img)
            else:
                res = cannyModel2(test_img)
            resultImage = torch.where(res[:,0,:,:]>0.5, 1, 0) + torch.where(res[:,1,:,:]>0.5, 2, 0) + torch.where(res[:,2,:,:]>0.5, 3, 0)
            resultImage = resultImage.unsqueeze(dim=1)
            out = sitk.GetImageFromArray(resultImage.cpu().detach().numpy())
            sitk.WriteImage(out, PSEUDO_PATH + '/pseudoLabel/' + str(index.item()) + '.nii.gz')

def preferModel(train_loader,test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullModel = UNet(3,3).to(device)
    fullModel.load_state_dict(torch.load(MODEL_FULL_PATH))
    fullModel.train()
    optimizer = torch.optim.Adam(fullModel.parameters(), lr=0.01)
    loss_fn = DiceLoss()
    epochs = 301

    loss_item = []
    loss_RV_item = []
    loss_LV_item = []
    loss_MYO_item = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_RV = 0
        total_loss_LV = 0
        total_loss_MYO = 0
        fullModel.train()

        for iteration, (train_img, index) in enumerate(test_loader):
            itkImage = sitk.ReadImage(PSEUDO_PATH + '/pseudoLabel/' + str(index.item()) + '.nii.gz')
            img = sitk.GetArrayFromImage(itkImage)
            validImg = torch.from_numpy(img).to(device)
            res = fullModel(train_img.squeeze(0))

            RV_train_label = validImg[:,0,:,:]
            RV_train_label = torch.where(RV_train_label==1, 1, 0)
            LV_train_label = validImg[:,0,:,:]
            LV_train_label = torch.where(LV_train_label==3, 1, 0)
            MYO_train_label = validImg[:,0,:,:]
            MYO_train_label = torch.where(MYO_train_label==2, 1, 0)
            loss_RV = loss_fn(res[:,0,:,:], RV_train_label)
            loss_LV = loss_fn(res[:,2,:,:], LV_train_label)
            loss_MYO = loss_fn(res[:,1,:,:], MYO_train_label)
            loss = loss_RV + loss_LV + loss_MYO
            
            total_loss += loss.item()
            total_loss_RV += loss_RV.item()
            total_loss_LV += loss_LV.item()
            total_loss_MYO += loss_MYO.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        for iteration, (train_img, train_label,index) in enumerate(train_loader):
            res = fullModel(train_img.squeeze(0))
            train_label = train_label.squeeze(0)

            RV_train_label = train_label[:,0,:,:]
            RV_train_label = torch.where(RV_train_label==1, 1, 0)
            LV_train_label = train_label[:,2,:,:]
            LV_train_label = torch.where(LV_train_label==3, 1, 0)
            MYO_train_label = train_label[:,1,:,:]
            MYO_train_label = torch.where(MYO_train_label==2, 1, 0)
            loss_RV = loss_fn(res[:,0,:,:], RV_train_label)
            loss_LV = loss_fn(res[:,2,:,:], LV_train_label)
            loss_MYO = loss_fn(res[:,1,:,:], MYO_train_label)
            loss = loss_RV + loss_LV + loss_MYO
            
            total_loss += loss.item()
            total_loss_RV += loss_RV.item()
            total_loss_LV += loss_LV.item()
            total_loss_MYO += loss_MYO.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_item.append(total_loss/(len(train_loader)+len(test_loader)))
        loss_RV_item.append(total_loss_RV/(len(train_loader)+len(test_loader)))
        loss_LV_item.append(total_loss_LV/(len(train_loader)+len(test_loader)))
        loss_MYO_item.append(total_loss_MYO/(len(train_loader)+len(test_loader)))

        print(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}')

        with open('/home/ubuntu/multi-version-unet/model/result/final/loss.txt', 'a') as f:
            f.write(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}\n')
        if epoch % 10 == 0:
            torch.save(fullModel.state_dict(), '/home/ubuntu/multi-version-unet/model/result/final/epoch_{}.pth'.format(epoch))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = MyDataset()
    test_data = MyDataset(ifTrain=False)
    train_loader = DataLoader(train_data,
                            shuffle=True,
                            batch_size=1)
    test_loader = DataLoader(test_data,
                            shuffle=True,
                            batch_size=1)
    '''
    trainEncoder()

    calSimilarity(train_loader)
    '''
    trainFullModel(train_loader, 
                test_loader)
    
    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='init_1',
                if_preprocess=False,
                version=0)
    
    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='init_2',
                if_preprocess=False,
                version=1)
    
    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='canny_1',
                if_preprocess=True,
                version=0)

    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='canny_2',
                if_preprocess=True,
                version=1)
    
    producePseudoLabelNew(test_loader=test_loader,
                        train_loader=train_loader)
    
    preferModel(test_loader=test_loader,
                train_loader=train_loader)
        
