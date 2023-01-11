import random
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.Unet import UNet
from utils.load import MyDataset
from utils.dice_loss import DiceLoss
from tqdm import tqdm
import cv2
import SimpleITK as sitk

MODEL_FULL_PATH = '/home/ubuntu/multi-version-unet/model/result/baseline_full/epoch_1000.pth'
MODEL_RANDOM_1_PATH = '/home/ubuntu/multi-version-unet/model/result/random_1/epoch_1000.pth'
MODEL_RANDOM_2_PATH = '/home/ubuntu/multi-version-unet/model/result/random_2/epoch_1000.pth'
MODEL_RANDOM_3_PATH = '/home/ubuntu/multi-version-unet/model/result/random_3/epoch_1000.pth'
MODEL_RANDOM_4_PATH = '/home/ubuntu/multi-version-unet/model/result/random_4/epoch_1000.pth'
PSEUDO_PATH = '/home/ubuntu/multi-version-unet/tmpSave/pseudoLabel_baseline'

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
    epochs = 1001

    loss_item = []
    loss_RV_item = []
    loss_LV_item = []
    loss_MYO_item = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_RV = 0
        total_loss_LV = 0
        total_loss_MYO = 0
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

        # write into log file
        with open ('/home/ubuntu/multi-version-unet/model/result/baseline_full/loss.txt','a') as f:
            f.write(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}\n')
        
        if(epoch%10==0):
            torch.save(model.state_dict(), '/home/ubuntu/multi-version-unet/model/result/baseline_full/epoch_{}.pth'.format(epoch))

def trainModel(train_loader,test_loader,if_preprocess,foldername,version):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3,3).to(device)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 1001

    loss_item = []
    loss_RV_item = []
    loss_LV_item = []
    loss_MYO_item = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        total_loss_RV = 0
        total_loss_LV = 0
        total_loss_MYO = 0
        model.train()

        for iteration, (train_img, train_label,index) in enumerate(train_loader):

            if(random.randint(0,1) == version):
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

        with open('/home/ubuntu/multi-version-unet/model/result/'+foldername+'/loss.txt', 'a') as f:
            f.write(f'epoch: {epoch}, loss: {loss_item[-1]}, loss_RV: {loss_RV_item[-1]}, loss_LV: {loss_LV_item[-1]}, loss_MYO: {loss_MYO_item[-1]}\n')
        
        if(epoch%10==0):
            torch.save(model.state_dict(), '/home/ubuntu/multi-version-unet/model/result/'+foldername+'/epoch_{}.pth'.format(epoch))

def loadModels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullModel = UNet(3,3).to(device)
    fullModel.load_state_dict(torch.load(MODEL_FULL_PATH))
    randomModel1 = UNet(3,3).to(device)
    randomModel1.load_state_dict(torch.load(MODEL_RANDOM_1_PATH))
    randomModel2 = UNet(3,3).to(device)
    randomModel2.load_state_dict(torch.load(MODEL_RANDOM_2_PATH))
    randomModel3 = UNet(3,3).to(device)
    randomModel3.load_state_dict(torch.load(MODEL_RANDOM_3_PATH))
    randomModel4 = UNet(3,3).to(device)
    randomModel4.load_state_dict(torch.load(MODEL_RANDOM_4_PATH))

    fullModel.eval()
    randomModel1.eval()
    randomModel2.eval()
    randomModel3.eval()
    randomModel4.eval()

    return fullModel,randomModel1,randomModel2,randomModel3,randomModel4

def producePseudoLabelNew(test_loader,train_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullModel,randomModel1,randomModel2,randomModel3,randomModel4 = loadModels()

    for iteration, (test_img,index) in tqdm(enumerate(test_loader)):
        test_img = test_img.squeeze(0).to(device)
        random_index = random.randint(1,4)
        if(random_index == 1):
            res = randomModel1(test_img)
        elif(random_index == 2):
            res = randomModel2(test_img)
        elif(random_index == 3):
            res = randomModel3(test_img)
        elif(random_index == 4):
            res = randomModel4(test_img)
        
        resultImage = torch.where(res[:,0,:,:]>0.5, 1, 0) + torch.where(res[:,1,:,:]>0.5, 2, 0) + torch.where(res[:,2,:,:]>0.5, 3, 0)
        resultImage = resultImage.unsqueeze(dim=1)
        out = sitk.GetImageFromArray(resultImage.cpu().detach().numpy())
        sitk.WriteImage(out, PSEUDO_PATH + '/' + str(index.item()) + '.nii.gz')

def preferModel(train_loader,test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullModel = UNet(3,3).to(device)
    fullModel.load_state_dict(torch.load(MODEL_FULL_PATH))
    fullModel.train()
    optimizer = torch.optim.Adam(fullModel.parameters(), lr=0.01)
    loss_fn = DiceLoss()
    epochs = 201

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
            itkImage = sitk.ReadImage(PSEUDO_PATH + '/' + str(index.item()) + '.nii.gz')
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
            torch.save(fullModel.state_dict(), '/home/ubuntu/multi-version-unet/model/result/final_baseline/epoch_{}.pth'.format(epoch))

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
    
    trainFullModel(train_loader=train_loader,
                    test_loader=test_loader,)
    
    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='random_1',
                if_preprocess=False,
                version=0)
    
    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='random_2',
                if_preprocess=False,
                version=1)
    
    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='random_3',
                if_preprocess=True,
                version=0)

    trainModel(train_loader=train_loader,
                test_loader=test_loader,
                foldername='random_4',
                if_preprocess=True,
                version=1)
    
    producePseudoLabelNew(test_loader=test_loader,
                        train_loader=train_loader)
    
    preferModel(test_loader=test_loader,
                train_loader=train_loader)
        
