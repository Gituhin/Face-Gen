import torch.nn as nn

class decoder_ae(nn.Module):
  def __init__(self, latent_dim):
    super(decoder_ae, self).__init__()
    self.fcu1 = nn.Sequential(nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True))
    self.fcu2 = nn.Sequential(nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True))
    self.fcu3 = nn.Sequential(nn.Linear(1024, 4096), nn.BatchNorm1d(4096), nn.LeakyReLU(0.2, inplace=True))
    self.fcu4 = nn.Sequential(nn.Linear(4096, 12800), nn.BatchNorm1d(12800), nn.LeakyReLU(0.2, inplace=True), 
                              nn.Unflatten(1, (512, 5, 5)))

    self.upconv1 = nn.Sequential(nn.ConvTranspose2d(512, 256, stride=2, padding=1, kernel_size=2),
                              nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
    
    self.upconv2 = nn.Sequential(nn.ConvTranspose2d(256, 128, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
    
    self.upconv3 = nn.Sequential(nn.ConvTranspose2d(128, 64, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
    
    self.upconv4 = nn.Sequential(nn.ConvTranspose2d(64, 32, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True))

    self.upconv5 = nn.Sequential(nn.ConvTranspose2d(32, 3, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(3), nn.Sigmoid())
    
  def forward(self, x):
    x = self.fcu1(x)
    x = self.fcu2(x)
    x = self.fcu3(x)
    x = self.fcu4(x)
    x = self.upconv1(x)
    x = self.upconv2(x)
    x = self.upconv3(x)
    x = self.upconv4(x)
    x = self.upconv5(x)
    return x


class decoder_vae(nn.Module):
  def __init__(self, latent_dim):
    super(decoder_vae, self).__init__()
    self.fcu1 = nn.Sequential(nn.Linear(latent_dim, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(0.2, inplace=True))
    self.fcu2 = nn.Sequential(nn.Linear(2048, 16384), nn.BatchNorm1d(16384), nn.LeakyReLU(0.2, inplace=True),
                              nn.Unflatten(1, (256, 8, 8)))

    self.upconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
    
    self.upconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
    
    self.upconv3 = nn.Sequential(nn.ConvTranspose2d(64, 3, stride=2, padding=1, kernel_size=4),
                              nn.BatchNorm2d(3), nn.Sigmoid())
    
  def forward(self, x):
    x = self.fcu1(x)
    x = self.fcu2(x)
    x = self.upconv1(x)
    x = self.upconv2(x)
    x = self.upconv3(x)
    return x
