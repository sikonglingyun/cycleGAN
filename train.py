import torch
import torchvision
import pandas as pd
import torchvision.datasets as dset
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import axes3d
from torchvision.datasets import MNIST
import os
import math
from Mydatasets import Mydatasets
from Generator import Generator
from Discriminator import Discriminator
import theano
import pylab
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm
from theano.tensor.shared_randomstreams import RandomStreams
beta1 = 0.5
cycle_late  = 1 #L1LossとadversarilLossの重要度を決定する係数
num_epochs = 10 #エポック数
batch_size = 1 #バッチサイズ
learning_rate = 1e-4 #学習率
train =False#学習を行うかどうかのフラグ
pretrained =True#事前に学習したモデルがあるならそれを使う
save_img =True#ネットワークによる生成画像を保存するかどうかのフラグ

import random
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), x.shape[1], x.shape[2],x.shape[3])
    return x


#データセットを調整する関数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
   
#訓練用データセット
#ここのパスは自分のGoogleDriveのパスに合うように変えてください


transform=transforms.Compose([
                              transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ])

#データセットをdataoaderで読み込み
#データセットをdataoaderで読み込み


dataset =  Mydatasets("./drive/My Drive/man/sub","./drive/My Drive/woman/sub",transform, transform, True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



def main():
    #もしGPUがあるならGPUを使用してないならCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    #ネットワークを呼び出し
    normal2nogi = Generator(3,3).to(device)
    

    #事前に学習しているモデルがあるならそれを読み込む
    #ここのパスは自分のGoogleDriveパスに合うように変えてください
    #./drive/My Drive/までは変えなくてできます

    if pretrained:
        param = torch.load('./drive/My Drive/normal2nogi.pth')
        normal2nogi.load_state_dict(param)

    #ネットワークを呼び出し
    nogi2normal = Generator(3,3).to(device)
    

    #事前に学習しているモデルがあるならそれを読み込む
    #ここのパスは自分のGoogleDriveパスに合うように変えてください
    #./drive/My Drive/までは変えなくてできます

    if pretrained:
        param = torch.load('./drive/My Drive/nogi2normal.pth')
        nogi2normal.load_state_dict(param)
   
   
    D_nogi = Discriminator(nch=3,nch_d=64).to(device)
    if pretrained:
        param = torch.load('./drive/My Drive/D_nogi.pth')
        
        D_nogi.load_state_dict(param)

    D_normal = Discriminator(nch=3,nch_d=64).to(device)
    if pretrained:
        param = torch.load('./drive/My Drive/D_normal.pth')
        
        D_normal.load_state_dict(param)

    #誤差関数には二乗誤差を使用
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()
    #更新式はAdamを適用
    
    optimizerD_nogi = torch.optim.Adam(D_nogi.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
    optimizernogi2normal = torch.optim.Adam(nogi2normal.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
    optimizerD_normal = torch.optim.Adam(D_normal.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
    optimizernormal2nogi = torch.optim.Adam(normal2nogi.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
  
    loss_train_list = []
    loss_test_list= []
    array = np.zeros(len(dataloader))
    for j in range(len(array)):
      array[j] = j
    print(len(dataset))
    for epoch in range(num_epochs):
        print(epoch)
        i=0
        for data,data2 in dataloader:
            
            real_image = data.to(device)   # 本物画像
            sample_size = real_image.size(0)  # 画像枚数
            real_target = torch.full((sample_size,1,1), random.uniform(1, 1), device=device)   # 本物ラベル
            fake_target = torch.full((sample_size,1,1), random.uniform(0, 0), device=device)   # 偽物ラベル
            
            
            nogi_image = data2.to(device)   # 本物画像
            #------Discriminatorの学習-------
# ------------------------------------------------------------------------------------------
            normal2nogi.zero_grad() 
            nogi2normal.zero_grad() 
            D_normal.zero_grad()
            D_nogi.zero_grad()
            
            fake_nogi = normal2nogi(real_image) #生成画像
            # fake_nogi = fake_nogi.cpu().detach().numpy()
            # fake_nogi_noise = addGaussianNoise(fake_nogi)
            # fake_nogi = torch.from_numpy(fake_nogi).float().to(device)
            # fake_nogi_noise = torch.from_numpy(fake_nogi_noise).float().to(device)
            
            output = D_nogi(fake_nogi) #生成画像に対するDiscriminatorの結果
            
            adversarial_nogi_loss_fake = criterion2(output,real_target) #Discriminatorの出力結果と正解ラベルとのBCELoss

            normal_nogi_normal = nogi2normal(fake_nogi)

            nogi_normal_nogi = normal2nogi(nogi2normal(nogi_image)) #生成画像)

            loss_normal_nogi_normal = criterion(normal_nogi_normal,real_image)

            loss_nogi_normal_nogi =criterion(nogi_normal_nogi,nogi_image)

            identify_normal =criterion(normal2nogi(nogi_image),nogi_image)

            loss_g_1 =identify_normal *cycle_late+loss_nogi_normal_nogi*cycle_late+ adversarial_nogi_loss_fake + loss_normal_nogi_normal*cycle_late #二つの損失をバランスを考えて加算
            
            loss_g_1.backward(retain_graph = True) # 誤差逆伝播
            
            if train == True :optimizernormal2nogi.step()  # Generatorのパラメータ更新
# ------------------------------------------------------------------------------------------
            #勾配情報の初期化
            normal2nogi.zero_grad() 
            nogi2normal.zero_grad() 
            D_normal.zero_grad()
            D_nogi.zero_grad()

            fake_normal = nogi2normal(nogi_image) #生成画像
            
            # fake_normal = fake_normal.cpu().detach().numpy()
            # fake_normal_noise = addGaussianNoise(fake_normal)
            # fake_normal = torch.from_numpy(fake_normal).float().to(device)
            # fake_normal_noise = torch.from_numpy(fake_normal_noise).float().to(device)

            output = D_normal(fake_normal) #生成画像に対するDiscriminatorの結果
            
            adversarial_normal_loss_fake = criterion2(output,real_target) #Discriminatorの出力結果と正解ラベルとのBCELoss

            nogi_normal_nogi = normal2nogi(fake_normal)

            normal_nogi_normal = nogi2normal(normal2nogi(real_image))

            loss_nogi_normal_nogi = criterion(nogi_normal_nogi,nogi_image)

            loss_normal_nogi_normal =criterion(normal_nogi_normal,real_image)

            identify_nogi = criterion(nogi2normal(real_image),real_image)

            loss_g_2 =identify_nogi *cycle_late+loss_normal_nogi_normal*cycle_late+adversarial_normal_loss_fake+ loss_nogi_normal_nogi*cycle_late #二つの損失をバランスを考えて加算
            
            loss_g_2.backward(retain_graph = True) # 誤差逆伝播
            if train == True :optimizernogi2normal.step()  # Generatorのパラメータ更新
            
#勾配情報の初期化
            normal2nogi.zero_grad() 
            nogi2normal.zero_grad() 
            D_normal.zero_grad()
            D_nogi.zero_grad()

            fake_nogi = normal2nogi(real_image) #生成画像
            # fake_nogi = fake_nogi.cpu().detach().numpy()
            # fake_nogi_noise = addGaussianNoise(fake_nogi)
            # fake_nogi = torch.from_numpy(fake_nogi).float().to(device)
            # fake_nogi_noise = torch.from_numpy(fake_nogi_noise).float().to(device)
            output = D_nogi(fake_nogi) #生成画像に対するDiscriminatorの結果

            adversarial_nogi_loss_fake = criterion2(output,fake_target) #Discriminatorの出力結果と正解ラベルとのBCELoss
            # nogi_image = nogi_image.cpu().detach().numpy()
            # nogi_image_noise = addGaussianNoise(nogi_image)
            # nogi_image = torch.from_numpy(nogi_image).float().to(device)
            # nogi_image_noise = torch.from_numpy(nogi_image_noise).float().to(device)
            output = D_nogi(nogi_image) #生成画像に対するDiscriminatorの結果

            adversarial_nogi_loss_real = criterion2(output,real_target) #Discriminatorの出力結果と正解ラベルとのBCELoss

            loss_d_1 = (adversarial_nogi_loss_fake+adversarial_nogi_loss_real)*1#単純に加算
            loss_d_1.backward(retain_graph = True) # 誤差逆伝播
            if train == True :optimizerD_nogi.step()  # Discriminatorのパラメータ更新
            

# ------------------------------------------------------------------------------------------
            #勾配情報の初期化
            normal2nogi.zero_grad() 
            nogi2normal.zero_grad() 
            D_normal.zero_grad()
            D_nogi.zero_grad()

            fake_normal = nogi2normal(nogi_image) #生成画像
            # fake_normal = fake_normal.cpu().detach().numpy()
            # fake_normal_noise = addGaussianNoise(fake_normal)
            # fake_normal = torch.from_numpy(fake_normal).float().to(device)
            # fake_normal_noise = torch.from_numpy(fake_normal_noise).float().to(device)
            output = D_normal(fake_normal) #生成画像に対するDiscriminatorの結果
            
            adversarial_normal_loss_fake = criterion2(output,fake_target) #Discriminatorの出力結果と正解ラベルとのBCELoss
            # real_image = real_image.cpu().detach().numpy()
            # real_image_noise = addGaussianNoise(real_image)
            # real_image = torch.from_numpy(real_image).float().to(device)
            # real_image_noise = torch.from_numpy(real_image_noise).float().to(device)
            output = D_normal(real_image) #生成画像に対するDiscriminatorの結果

            adversarial_normal_loss_real = criterion2(output,real_target) #Discriminatorの出力結果と正解ラベルとのBCELoss


            loss_d_2 =  (adversarial_normal_loss_fake+adversarial_normal_loss_real)*1#単純に加算
            loss_d_2.backward(retain_graph = True) # 誤差逆伝播
            if train == True :optimizerD_normal.step()  # Discriminatorのパラメータ更新
            
# ------------------------------------------------------------------------------------------
           
           
         

            fake_nogi = normal2nogi(real_image) #生成画像
            fake_normal = nogi2normal(nogi_image) #生成画像
            i=i+1
            if i % 10==0:
              
              if save_img == True:
                value = int(math.sqrt(batch_size))
                pic = to_img(nogi_image.cpu().data)
                pic = torchvision.utils.make_grid(pic,nrow = value)
                save_image(pic, './drive/My Drive/nogi_image_{}.png'.format(int(epoch)))  #白黒画像を保存

                pic = to_img(real_image.cpu().data)
                pic = torchvision.utils.make_grid(pic,nrow = value)
                save_image(pic, './drive/My Drive/real_image_{}.png'.format(int(epoch)))  #生成画像を保存

                pic = to_img(fake_nogi.cpu().data)
                pic = torchvision.utils.make_grid(pic,nrow = value)
                save_image(pic, './drive/My Drive/fake_nogi_image_{}.png'.format(int(epoch)))  #白黒画像を保存

                pic = to_img(fake_normal.cpu().data)
                pic = torchvision.utils.make_grid(pic,nrow = value)
                save_image(pic, './drive/My Drive/fake_normal_{}.png'.format(int(epoch)))  #生成画像を保存

                pic = to_img(nogi_normal_nogi.cpu().data)
                pic = torchvision.utils.make_grid(pic,nrow = value)
                save_image(pic, './drive/My Drive/nogi_normal_nogi_image_{}.png'.format(int(epoch)))  #白黒画像を保存

                pic = to_img(normal_nogi_normal.cpu().data)
                pic = torchvision.utils.make_grid(pic,nrow = value)
                save_image(pic, './drive/My Drive/normal_nogi_normal_{}.png'.format(int(epoch)))  #生成画像を保存

              print(i, len(dataloader),loss_g_1,loss_g_2,loss_d_1,loss_d_2)   
              # if i == 4:break 
              if train == False and i == 10:break
          
        if train == True:
                #モデルを保存
                torch.save(nogi2normal.state_dict(), './drive/My Drive/nogi2normal.pth')
                torch.save(normal2nogi.state_dict(), './drive/My Drive/normal2nogi.pth')
                torch.save(D_nogi.state_dict(), './drive/My Drive/D_nogi.pth')
                torch.save(D_normal.state_dict(), './drive/My Drive/D_normal.pth')

                #ここのパスは自分のGoogleDriveのパスに合うように変えてください
    
if __name__ == '__main__':
    main() 