import torch
import torchvision
import pandas as pd
import torchvision.datasets as dset
from torch import nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import axes3d
from torchvision.datasets import MNIST
import os
import math
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
train =True#学習を行うかどうかのフラグ
pretrained =False#事前に学習したモデルがあるならそれを使う
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
class Mydatasets(Dataset):
    def __init__(self, path1,path2, transform1 = None, transform2 = None, train = True):
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        self.path1 = path1
        self.path2 = path2
        self.file1 = os.listdir(self.path1)
        self.file2 = os.listdir(self.path2)

        self.datanum = len(self.file1)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        image1 = Image.open(self.path1+"/"+self.file1[idx])
        image2 = Image.open(self.path2+"/"+self.file2[idx])
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        image1 = self.transform1(image1)
        image2 = self.transform2(image2)
        return image1, image2

transform=transforms.Compose([
                              transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ])

#データセットをdataoaderで読み込み
#データセットをdataoaderで読み込み


dataset =  Mydatasets("./drive/My Drive/man/sub","./drive/My Drive/woman/sub",transform1=transform, transform2=transform, train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#pix2pixのGenerator部分
class Generator(nn.Module):
    def __init__(self,nch,nch_d):
        super(Generator, self).__init__()
        nch_g = 64
        #U-net部分
        self.layer1 = self.conv_layer_forward(nch, nch_g , 3, 2, 1,False)
        self.Res1 = self.Residual_module(nch_g)
        self.layer2 = self.conv_layer_forward(nch_g , nch_g*2 , 3, 2, 1,False)
        self.Res2 = self.Residual_module(nch_g*2)
        self.layer3 = self.conv_layer_forward(nch_g*2 , nch_g*4 , 3, 2, 1,False)
        self.Res3 = self.Residual_module(nch_g*4)
        self.layer4= self.conv_layer_forward(nch_g*4 , nch_g*8 , 3, 2, 1,False)
        self.Res4 = self.Residual_module(nch_g*8)
        self.layer5= self.conv_layer_forward(nch_g*8 , nch_g*16 , 3, 2, 1,False)
        self.Res5 = self.Residual_module(nch_g*16)
        self.layer6= self.conv_layer_forward_image_size_1(nch_g*16 , nch_g*32 , 4, 1, 1)
        self.layer7= self.conv_layer_transpose(nch_g*32 , nch_g*16 , 4, 2, 1,False)
        self.Res7 = self.Residual_module(nch_g*32)
        self.layer8 = self.conv_layer_transpose(nch_g*32 , nch_g*8 , 4, 2, 1,False)
        self.Res8 = self.Residual_module(nch_g*16)
        self.layer9 = self.conv_layer_transpose(nch_g*16 , nch_g*4 , 4, 2, 1,False)
        self.Res9 = self.Residual_module(nch_g*8)
        self.layer10= self.conv_layer_transpose(nch_g*8 , nch_g*2 , 4, 2, 1,False)
        self.Res10 = self.Residual_module(nch_g*4)
        self.layer11= self.conv_layer_transpose(nch_g*4 , nch_g , 4, 2, 1,False)
        self.Res11 = self.Residual_module(nch_g*2)
        self.layer12 = self.conv_layer_transpose(nch_g*2 , nch_g , 4, 2, 1,False)
        self.Res12 = self.Residual_module(nch_g)
        self.layer13= self.conv_layer_forward(nch_g , nch_d , 1, 1, 0,True)

    def forward(self, z):
        z,z1 = self.convolution_forward(self.layer1,z)
        z, _ = self.convolution(self.Res1,z)
        z = z+z1
        z = z/2
        z1 = z
        z,z2= self.convolution_forward(self.layer2,z)
        z,_ = self.convolution(self.Res2,z)
        z = z+z2
        z = z/2
        z2 = z
        z,z3 = self.convolution_forward(self.layer3,z)
        z,_ = self.convolution(self.Res3,z)
        z = z+z3
        z = z/2
        z3 = z
        z,z4 = self.convolution_forward(self.layer4,z)
        z,_ = self.convolution(self.Res4,z)
        z = z+z4
        z = z/2
        z4 = z
        z,z5 = self.convolution_forward(self.layer5,z)
        z,_ = self.convolution(self.Res5,z)
        z = z+z5
        z = z/2
        z5 = z
        z,_ = self.convolution(self.layer6,z)
        z = self.convolution_deconv(self.layer7,z,z5)
        z,z_copy = self.convolution(self.Res7,z)
        z = z+z_copy
        z = z/2
        z = self.convolution_deconv(self.layer8,z,z4)
        z,z_copy = self.convolution(self.Res8,z)
        z = z+z_copy
        z = z/2
        z = self.convolution_deconv(self.layer9,z,z3)
        z,z_copy = self.convolution(self.Res9,z)
        z = z+z_copy
        z = z/2
        z = self.convolution_deconv(self.layer10,z,z2)
        z,z_copy = self.convolution(self.Res10,z)
        z = z+z_copy
        z = z/2
        z = self.convolution_deconv(self.layer11,z,z1)
        z,z_copy = self.convolution(self.Res11,z)
        z = z+z_copy
        z = z/2
        z,_ = self.convolution(self.layer12,z)
        z,z_copy = self.convolution(self.Res12,z)
        z = z+z_copy
        z,_ = self.convolution(self.layer13,z)
        return z

    def convolution(self,layer_i,z):
      for layer in layer_i.values(): 
            z = layer(z)
      z_copy = z
      return z,z_copy

    def Residual_module(self,input):
      return nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Conv2d(input,int(input/2),3,1,1,bias = False),
                    nn.InstanceNorm2d(int(input/2)),
                    nn.ReLU(),  
                    ),  
                'layer1': nn.Sequential(
                    nn.Conv2d(int(input/2),int(input/2),1,1,0,bias = False),
                    nn.InstanceNorm2d(int(input/2)),
                    nn.ReLU(),  
                    ),  
                'layer2': nn.Sequential(
                    nn.Conv2d(int(input/2),input,3,1,1,bias = False),
                    nn.InstanceNorm2d(input),
                    nn.ReLU(),  
                    ),  
                })
    
    def conv_layer_forward(self,input,out,kernel_size,stride,padding,is_last):
        if is_last == False:
          return nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Conv2d(input,out,kernel_size,stride,padding,bias = False),
                    nn.InstanceNorm2d(out),
                    nn.ReLU(),  
                    ),  
                })
        else:
          return nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Conv2d(input,out,kernel_size,stride,padding,bias = False),
                    nn.Tanh()
                    ),  
                })
        
    def conv_layer_forward_image_size_1(self,input,out,kernel_size,stride,padding):
        return nn.ModuleDict({
              'layer0': nn.Sequential(
                  nn.Conv2d(input,out,kernel_size,stride,padding,bias = False),
                  nn.ReLU(),  
                  ),  
              })
        
    def conv_layer_transpose(self,input,out,kernel_size,stride,padding,is_last):
      if is_last == True:
        return nn.ModuleDict({
              'layer0': nn.Sequential(
                  nn.ConvTranspose2d(input , out , kernel_size, stride, padding,bias = False),
                  nn.Tanh()  
                  ),
              })
      else :
        return nn.ModuleDict({
               'layer0': nn.Sequential(
                  nn.ConvTranspose2d(input , out , kernel_size, stride, padding,bias = False),
                  nn.InstanceNorm2d(out),
                  nn.ReLU(),  
                  ), 
              })
        
    def convolution_forward(self,layer,z):
        z,z_copy = self.convolution(layer,z)
        return z,z_copy
    def convolution_deconv(self,layer,z,z_copy):
        z,_ = self.convolution(layer,z)
        z = torch.cat([z,z_copy],dim = 1)
        return z

 
class Discriminator(nn.Module):
  #Dicriminator部分
  def __init__(self, nch=3, nch_d=16):
     super(Discriminator, self).__init__()
     self.layer1 = self.conv_layer(nch, nch_d, 3, 2, 1,False)
     self.layer2 = self.conv_layer(nch_d, nch_d * 2, 3, 2, 1,False)
     self.layer3 = self.conv_layer(nch_d * 2, nch_d * 4,3, 2, 1,False)
     self.layer4 = self.conv_layer(nch_d * 4, nch_d * 8, 3, 2, 1,False)
     self.layer5 = self.conv_layer(nch_d * 8, nch_d * 16, 3, 2, 1,False)
     self.layer6 = self.conv_layer(nch_d * 16, 1, 4, 1, 1,True)
  def multi_layers(self,layer1,layer2,layer3,z):
      z1 = self.convolution(layer1,z)
      z2 = self.convolution(layer2,z)
      z3 = self.convolution(layer2,z)
      z = z1+z2+z3
      z_copy = z

      return z

  def conv_layer(self,input,out,kernel_size,stride,padding,is_last):
      if is_last == True:
        return nn.ModuleDict({
              'layer0': nn.Sequential(
                  nn.Conv2d(input , out , kernel_size, stride, padding,bias = False),
                  
                  ),
              })
      else :
        return nn.ModuleDict({
               'layer0': nn.Sequential(
                  nn.Conv2d(input , out , kernel_size, stride, padding,bias = False),
                  nn.InstanceNorm2d(out),
                  nn.ReLU(),  
                  ), 
              })
        
  def convolution(self,layer_i,z):
      for layer in layer_i.values(): 
            z = layer(z)
      return z
  def forward(self, x):
      x = self.convolution(self.layer1,x)
      x = self.convolution(self.layer2,x)
      x = self.convolution(self.layer3,x)
      x = self.convolution(self.layer4,x)
      x = self.convolution(self.layer5,x)
      x = self.convolution(self.layer6,x)
      return x
def addGaussianNoise(src):
    row= src.shape[2]
    col =src.shape[3]
    ch =src.shape[1]
    mean = 0
    var = 0.01
    sigma = 0.1
    gauss = np.random.normal(mean,sigma,(1,ch,row,col))
    gauss = gauss.reshape(1,ch,row,col)
    noisy = src + gauss

    return noisy
def logcoshobj(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
def hinge(output,target):
    hinge_loss = 1 - torch.mul(output, target)
    hinge_loss[hinge_loss < 0] = 0
    return hinge_loss
def NoiseImage(image):
  return noise_image
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