# %%
import torch
import os
import math
import torch.distributed as dist
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
torch.manual_seed(0)
# %%
if torch.cuda.is_available():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend = 'nccl')
    device = torch.device('cuda', local_rank)
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class Resmodel(nn.Module):

    def __init__(self, base_model):
        super(Resmodel, self).__init__()

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 64), nn.ReLU(), nn.Dropout(0), nn.Linear(64, 2))

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="IMAGENET1K_V1")
        return model

    def forward(self, x):
        return self.backbone(x)

def Gradwrapper(test_dir):
    accuracy = 87.61/100
    I = 1.96*math.sqrt((accuracy*(1-accuracy))/815)
    CI = (accuracy-I, accuracy+I)
    checkpoint = torch.load('grad.pth.tar', map_location = device)
    state_dict = checkpoint['state_dict']

    model = Resmodel('resnet50')
    log = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    # model = DDP(model, device_ids = [local_rank], output_device=local_rank)
    model.eval()

    img = Image.open(test_dir)
    data_transforms = transforms.Compose([Resize((512, 640),interpolation=BICUBIC),
                                          _convert_image_to_rgb,
                                          ToTensor(),
                                          Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          ])
    img = data_transforms(img)
    img = img.unsqueeze(0)

    with torch.no_grad():

        img = img.to(device)
        logits = model(img.type(torch.float32))
        prob = nn.Softmax(dim=1)(logits)[:,1]
        predict = (prob >= 0.043)*1
    if predict == 1:
        cls = 'Gradable'
        confidence = 0.5 + (prob - 0.043)*0.5/(1 - 0.043)
    elif predict == 0:
        cls = 'Ungradable'
        confidence = prob *0.5/0.043

    return (prob.item(), 0.043, cls, confidence.item())

def RDRwrapper(test_dir):
    accuracy = 88.61/100
    I = 1.96*math.sqrt((accuracy*(1-accuracy))/572)
    CI = (accuracy-I, accuracy+I)
    checkpoint = torch.load('RDR.pth.tar', map_location = device)
    state_dict = checkpoint['state_dict']

    model = Resmodel('resnet50')
    log = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    # model = DDP(model, device_ids = [local_rank], output_device=local_rank)
    model.eval()

    img = Image.open(test_dir)
    data_transforms = transforms.Compose([Resize((512, 640),interpolation=BICUBIC),
                                          _convert_image_to_rgb,
                                          ToTensor(),
                                          Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          ])
    img = data_transforms(img)
    img = img.unsqueeze(0)

    with torch.no_grad():

        img = img.to(device)
        logits = model(img.type(torch.float32))
        prob = nn.Softmax(dim=1)(logits)[:,1]
        predict = (prob >= 0.9315)*1
    if predict == 1:
        cls = 'RDR'
        confidence = 0.5 + (prob - 0.9315)*0.5/(1 - 0.9315)
    elif predict == 0:
        cls = 'Non RDR'
        confidence = prob *0.5/0.9315

    return (prob.item(), 0.9315, cls, confidence.item())

def VTDRwrapper(test_dir):
    accuracy = 88.09/100
    I = 1.96*math.sqrt((accuracy*(1-accuracy))/571)
    CI = (accuracy-I, accuracy+I)
    checkpoint = torch.load('VTDR.pth.tar', map_location = device)
    state_dict = checkpoint['state_dict']

    model = Resmodel('resnet50')
    log = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    # model = DDP(model, device_ids = [local_rank], output_device=local_rank)
    model.eval()

    img = Image.open(test_dir)
    data_transforms = transforms.Compose([Resize((512, 640),interpolation=BICUBIC),
                                          _convert_image_to_rgb,
                                          ToTensor(),
                                          Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          ])
    img = data_transforms(img)
    img = img.unsqueeze(0)

    with torch.no_grad():

        img = img.to(device)
        logits = model(img.type(torch.float32))
        prob = nn.Softmax(dim=1)(logits)[:,1]
        predict = (prob >= 0.57)*1
    if predict == 1:
        cls = 'VTDR'
        confidence = 0.5 + (prob - 0.57)*0.5/(1 - 0.57)
    elif predict == 0:
        cls = 'Non VTDR'
        confidence = prob *0.5/0.57

    return (prob.item(), 0.57, cls, confidence.item())

if __name__ == "__main__":
    a,b,c,d  = Gradwrapper('/home/pwuaj/data/RDRraw/test/1/STDR389-20170320@111304-L1-S.jpg')
    print(a,b,c,d)
