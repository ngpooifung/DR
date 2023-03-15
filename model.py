import torch
import torch.nn as nn
import torchvision.models as models

from exceptions import InvalidBackboneError

class modeltrainer():
    def __init__(self):
        self.model_dict = {'resnet'}

    def _get_model(self, base_model, out_dim, pretrained):
        type = torch.hub.load('pytorch/vision:v0.11.2', base_model, pretrained=pretrained).__module__

        if type == 'torchvision.models.resnet':
            return Resmodel(base_model, out_dim, pretrained)
        else:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: {}".format(self.model_dict))


class Resmodel(nn.Module):

    def __init__(self, base_model, out_dim, pretrained):
        super(Resmodel, self).__init__()

        self.backbone = self._get_basemodel(base_model, pretrained = pretrained)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=dim_mlp, out_features=out_dim, bias=True)
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(2), nn.Sigmoid())

    def _get_basemodel(self, model_name, pretrained=False):
        model = torch.hub.load('pytorch/vision:v0.11.2', model_name, pretrained=pretrained)
        return model

    def forward(self, x):
        return self.backbone(x)
