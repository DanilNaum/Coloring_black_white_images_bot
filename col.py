import numpy as np 
import pandas as pd

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import glob
from PIL import Image

batch_size = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)

def generate_l_ab(images):
    lab = rgb2lab(images.permute(0, 2, 3, 1).cpu().numpy())
    X = lab[:,:,:,0]
    X = X.reshape(X.shape+(1,))
    Y = lab[:,:,:,1:] / 128
    return to_device(torch.tensor(X, dtype = torch.float).permute(0, 3, 1, 2), device),to_device(torch.tensor(Y, dtype = torch.float).permute(0, 3, 1, 2), device)

class BaseModel(nn.Module):
    def training_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return loss

    def validation_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return {'val_loss' : loss.item()}

    def validation_end_epoch(self, outputs):
        epoch_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        return {'epoch_loss' : epoch_loss}

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class Encoder_Decoder(BaseModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=3, padding=get_padding(3)),
            nn.Upsample(size=(64, 64)),
            nn.Conv2d(128, 64, kernel_size=3, padding=get_padding(3)),
            nn.Upsample(size=(128, 128)),
            nn.Conv2d(64, 32, kernel_size=3, padding=get_padding(3)),
            nn.Conv2d(32, 16, kernel_size=3, padding=get_padding(3)),
            nn.Conv2d(16, 2, kernel_size=3, padding=get_padding(3)),
            nn.Tanh(),
            nn.Upsample(size=(256, 256))
        )

    def forward(self, images):
        return self.network(images)

def load_checkpoint(filepath):
    model = Encoder_Decoder()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))

    return model

model = load_checkpoint('./landscapes.pth')
to_device(model, device)

def to_rgb(grayscale_input, ab_output):
    color_image = torch.cat((grayscale_input, ab_output), 0).numpy() 
    print(color_image.shape)
    color_image = color_image.transpose((1, 2, 0))  
    color_image[:, :, 0:1] = color_image[:, :, 0:1]
    color_image[:, :, 1:3] = (color_image[:, :, 1:3]) * 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    return color_image

def prediction(img):
    a = rgb2lab(img.permute(1, 2, 0))
    a = torch.tensor(a[:,:,0]).type(torch.FloatTensor)
    a = a.unsqueeze(0)
    a = a.unsqueeze(0)
    xb = to_device(a, device)
    ab_img = model(xb)
    xb = xb.squeeze(0)
    ab_img = ab_img.squeeze(0)
    return to_rgb(xb.detach().cpu(), ab_img.detach().cpu())


def Col(path,mode = False):
    images = glob.glob(path)

    img = images[0]
    image = Image.open(img)
    trans = T.Compose([T.Resize((256, 256)),T.ToTensor()])
    img = trans(image)

    A = prediction(img)
    if mode:
        f, arr = plt.subplots(1, 2, sharey=True)
        arr[0].imshow(img.permute(1, 2, 0))
        arr[1].imshow(A)
        f=plt.imshow(A,aspect="auto")
        plt.show()
    f = plt.imsave(f"Res{path}",A)
    return A

if __name__ == '__main__':

    Col('1.jpg', mode = True)