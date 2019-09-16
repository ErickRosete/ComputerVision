# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Defining generator (Inverse CNN)
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
              nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
              nn.BatchNorm2d(512),
              nn.ReLU(True),
              nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
              nn.BatchNorm2d(256),
              nn.ReLU(True),
              nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
              nn.BatchNorm2d(128),
              nn.ReLU(True),
              nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
              nn.BatchNorm2d(64),
              nn.ReLU(True),
              nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
              nn.Tanh()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output
    
#Defining discriminator (CNN)
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
              nn.Conv2d(3, 64, 4, 2, 1, bias = False),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(64, 128, 4, 2, 1, bias = False),
              nn.BatchNorm2d(128),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(128, 256, 4, 2, 1, bias = False),
              nn.BatchNorm2d(256),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(256, 512, 4, 2, 1, bias = False),
              nn.BatchNorm2d(512),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(512, 1, 4, 1, 0, bias = False),
              nn.Sigmoid()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output.view(-1) #1D


# Hyperparameters
batchSize = 64 
imageSize = 64 

# Transformations
transform = transforms.Compose([transforms.Scale(imageSize), \
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), \
            (0.5, 0.5, 0.5)),])
 
# Dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, \
             batch_size = batchSize, shuffle = True, num_workers = 4) 

# Neural networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = G()
netG.to(device)
netG.apply(weights_init)

netD = D()
netD.to(device)
netD.apply(weights_init)

# Training
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.99))

if __name__ == "__main__": 
    for epoch in range(25):
        for i, data in enumerate(dataloader, 0):
            
            #training discriminator
            netD.zero_grad()
            real, _  = data
            input = Variable(real).to(device)
            target = Variable(torch.ones(input.size()[0])).to(device)
            output = netD(input)
            errD_real = criterion(output, target)
            
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1)).to(device)
            fake  = netG(noise)
            target = Variable(torch.zeros(input.size()[0])).to(device)
            output = netD(fake.detach())
            errD_fake = criterion(output, target)
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            
            #training generator
            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0])).to(device)
            output = netD(fake)
            errG = criterion(output, target)
    
            errG.backward()
            optimizerG.step()
            
            print('epoch: [%d/%d] step: [%d/%d] errD: %.4f errG: %.4f'\
                  % (epoch, 25, i, len(dataloader), errD.data, errG.data))
            
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' \
                                 % './results', normalize = True)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' \
                                  % ('./results', epoch), normalize = True)
