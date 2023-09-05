# %%
import torch
import os
import torch.distributed as dist
from model import modeltrainer
from torch import nn
from trainer import Restrainer
import torch.backends.cudnn as cudnn
from dataset import Modeldataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
from sklearn.metrics import accuracy_score
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
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 64), nn.ReLU(), nn.Linear(64, 2))

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.11.2', model_name, pretrained = True)
        return model

    def forward(self, x):
        return self.backbone(x)

def wrapper(test_dir):
    accuracy = 85.67/100
    I = 1.96*math.sqrt((accuracy*(1-accuracy))/571)
    CI = (accuracy-I, accuracy+I)
    checkpoint = torch.load('VTDR.pth.tar', map_location = device)
    state_dict = checkpoint['state_dict']

    model = Resmodel('resnet50')
    log = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model = DDP(model, device_ids = [local_rank], output_device=local_rank)
    model.eval()

    img = Image.open(test_dir)
    data_transforms = transforms.Compose([Resize((512, 640),interpolation=BICUBIC),
                                          _convert_image_to_rgb,
                                          ToTensor(),
                                          ])
    img = data_transforms(img)
    img = img.unsqueeze(0)

    with torch.no_grad():

        img = img.to(device)
        logits = model.module(img.type(torch.float32))
        prob = nn.Softmax(dim=1)(logits)[:,1]
        predict = (prob >= 0.3921)*1
    if predict == 1:
        cls = 'VTDR'
    elif predict == 0:
        cls = 'Non VTDR'

    return (cls, prob.item(), CI, 0.3921)


if __name__ == "__main__":
    a,b,c  = wrapper('/home/pwuaj/data/RDRraw/test/1/STDR389-20170320@111304-L1-S.jpg')
    print(a,b,c)
