import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.cuda.amp as amp
import tqdm.autonotebook as tqdm
import torchinfo

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = 32
img_channel = 1
batch_size = 64
img_shape = torch.tensor([img_channel, img_size, img_size])
noise_dim = 100
lr = 2e-4
b1 = 0.5
b2 = 0.999
n_epochs = 200
sample_interval = 1000
batch_done = 0

os.makedirs("./generated", exist_ok=True)
os.makedirs("./reals", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mnist = datasets.MNIST(root="reals/", transform=transform, train=True, download=True)

dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim: int, img_shape: torch.Tensor):
        super(Generator, self).__init__()
        self.block1 = self._block(noise_dim, 128, normalize=False)
        self.block2 = self._block(128, 256)
        self.block3 = self._block(256, 512)
        self.block4 = self._block(512, 1024)
        self.block5 = nn.Linear(1024, torch.prod(img_shape))
        self.tanh = nn.Tanh()
        self.img_shape = img_shape

    def _block(self, in_feat: int, out_feat: int, normalize: bool = True, activation: bool = True):
        layers = [nn.Linear(in_features=in_feat, out_features=out_feat)]
        if normalize:
            layers.append(nn.LayerNorm(out_feat))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, noise):
        z = self.block1(noise)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = self.block5(z)
        z = self.tanh(z)
        z = z.view(z.shape[0], *self.img_shape)
        return z


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(torch.prod(img_shape), 512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        out = self.model(img_flat)
        return out


generator = Generator(noise_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

torchinfo.summary(generator, (batch_size, noise_dim))
torchinfo.summary(discriminator, (batch_size, *img_shape))

optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=lr,
                               betas=(b1, b2))

optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=lr,
                               betas=(b1, b2))

criterion = nn.BCELoss()

for epoch in range(n_epochs):
    loop = tqdm.tqdm(iterable=dataloader, total=len(dataloader), leave=True)
    for i, (images, _) in enumerate(loop):
        real_imgs = images.to(device)
        noises = torch.normal(mean=0, std=1, size=(images.shape[0], noise_dim)).to(device)

        ### Train Discriminator
        optimizer_D.zero_grad()
        fake_imgs = generator(noises)
        real_pred = discriminator(real_imgs)
        fake_pred = discriminator(fake_imgs)
        loss_real = criterion(real_pred, torch.ones_like(real_pred).to(device))
        loss_fake = criterion(fake_pred, torch.zeros_like(fake_pred).to(device))
        loss_d = (loss_fake + loss_real) / 2
        loss_d.backward()
        optimizer_D.step()

        ### Train Generator
        optimizer_G.zero_grad()
        fake_imgs = generator(noises)
        fake_pred = discriminator(fake_imgs)
        loss_g = criterion(fake_pred, torch.ones_like(fake_pred).to(device))
        loss_g.backward()
        optimizer_G.step()

        loop.set_description(f"Epoch: {epoch}, Step: {i}, D Loss: {loss_d}, G_Loss: {loss_g}")
        if batch_done % sample_interval == 0:
            utils.save_image(fake_imgs, f"generated/images{batch_done}.jpg", normalize=True)
            with open("results.txt", "a") as f:
                f.write(f"Images{batch_done}: D Loss: {loss_d}, G_Loss: {loss_g}\n")
        batch_done += 1