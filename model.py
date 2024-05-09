import torch
import torch.nn as nn
import torchvision.models as models

from exceptions import InvalidBackboneError

class modeltrainer():
    def __init__(self):
        self.model_dict = {'resnet'}

    def _get_model(self, base_model, out_dim, dropout):
        type = torch.hub.load('pytorch/vision:v0.13.0', base_model).__module__

        if type == 'torchvision.models.resnet':
            return Resmodel(base_model, out_dim, dropout)
        elif type == 'torchvision.models.inception':
            return Inceptionmodel(base_model, out_dim, dropout)
        elif type == 'torchvision.models.densenet':
            return densemodel(base_model, out_dim, dropout)
        elif type == 'torchvision.models.vision_transformer':
            return Vision(base_model, out_dim, dropout)
        elif type == 'torchvision.models.efficientnet':
            return efficient(base_model, out_dim)
        else:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: {}".format(self.model_dict))


class Resmodel(nn.Module):

    def __init__(self, base_model, out_dim, dropout):
        super(Resmodel, self).__init__()

        self.backbone = self._get_basemodel(base_model)
        self.dropout = dropout
        self.out_dim = out_dim
        dim_mlp = self.backbone.fc.in_features
        # self.backbone.fc = nn.Linear(in_features=dim_mlp, out_features=out_dim, bias=True)
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 64), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(64, self.out_dim))

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="IMAGENET1K_V1")
        return model

    def forward(self, x):
        return self.backbone(x)

class Vision(nn.Module):

    def __init__(self, base_model, out_dim, dropout):
        super(Vision, self).__init__()

        self.backbone = self._get_basemodel(base_model)
        self.dropout = dropout
        self.out_dim = out_dim
        dim_mlp = self.backbone.fc.in_features
        self.backbone.heads.head = nn.Linear(in_features=dim_mlp, out_features=out_dim, bias=True)

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="IMAGENET1K_SWAG_E2E_V1")
        return model

    def forward(self, x):
        return self.backbone(x)


class Inceptionmodel(nn.Module):

    def __init__(self, base_model, out_dim, dropout):
        super(Inceptionmodel, self).__init__()

        self.backbone = self._get_basemodel(base_model)
        self.dropout = dropout
        self.out_dim = out_dim
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 64), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(64, self.out_dim))        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1), nn.Sigmoid())

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="IMAGENET1K_V1")
        return model

    def forward(self, x):
        return self.backbone(x)


class densemodel(nn.Module):

    def __init__(self, base_model, out_dim, dropout):
        super(densemodel, self).__init__()

        self.backbone = self._get_basemodel(base_model)
        self.dropout = dropout
        self.out_dim = out_dim
        dim_mlp = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(nn.Linear(dim_mlp, 64), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(64, self.out_dim))        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1), nn.Sigmoid())

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="IMAGENET1K_V1")
        return model

    def forward(self, x):
        return self.backbone(x)


class efficient(nn.Module):

    def __init__(self, base_model, out_dim):
        super(efficient, self).__init__()

        self.backbone = self._get_basemodel(base_model)
        self.out_dim = out_dim
        dim_mlp = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(dim_mlp, out_dim)       # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1), nn.Sigmoid())

    def _get_basemodel(self, model_name):
        model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights="IMAGENET1K_V1")
        return model

    def forward(self, x):
        return self.backbone(x)
