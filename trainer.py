# %%
import numpy as np
import torch
from PIL import Image
from utils import topacc, save_checkpoint, bceacc
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os
import scipy
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
torch.manual_seed(0)

# %%


class Restrainer(object):
    def  __init__(self, model, optimizer, scheduler, args):
        self.model = model.float()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss(weight = torch.FloatTensor(max(self.args.weight)/np.array(self.args.weight))).to(self.args.device)
        log_dir = self.args.dir
        if self.args.output is not None:
            self.writer = SummaryWriter(log_dir = os.path.join(self.args.output, self.args.process))
        else:
            self.writer = SummaryWriter(log_dir = log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    @property
    def dtype(self):
        # return self.model.module.conv1.weight.dtype
        return torch.float32

    def train(self, train_loader, test_loader = None):
        self.model.train()

        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        logging.info(f"Total GPU device: {self.args.device_count}.")

        for epoch_counter in range(self.args.epochs):
            train_loader.sampler.set_epoch(epoch_counter)
            iterator = iter(train_loader)
            if test_loader is not None:
                test_loader.sampler.set_epoch(epoch_counter)
            top1_train_accuracy = 0
            top1_valid_accuracy = 0

            for counter, (img, lbl) in enumerate(train_loader):
                img = img.to(self.args.device)
                lbl = lbl[1].to(self.args.device)

                logits = self.model.module(img.type(self.dtype))
                try:
                    logits = logits.logits
                except:
                    pass
                loss = self.criterion(logits.squeeze(), lbl)

                top1 = topacc(logits, lbl, topk = (1,))
                # top1 = accuracy_score(lbl.cpu(), (logits>0.5).cpu())
                top1_train_accuracy += top1[0]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            top1_train_accuracy /= (counter + 1)

            if test_loader is not None:
                with torch.no_grad():
                    for counter, (img, lbl) in enumerate(test_loader):
                        img = img.to(self.args.device)
                        lbl = lbl[1].to(self.args.device)

                        logits = self.model.module(img.type(self.dtype))
                        try:
                            logits = logits.logits
                        except:
                            pass
                        top1 = topacc(logits, lbl, topk = (1,))
                        # top1 = accuracy_score(lbl.cpu(), (logits>0.5).cpu())
                        top1_valid_accuracy += top1[0]
                    top1_valid_accuracy /= (counter + 1)

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1_train_accuracy.item()}\tTop1 valid accuracy: {top1_valid_accuracy.item()}\tLR: {self.scheduler.get_last_lr()}")
            print(f"Epoch: {epoch_counter}\tLoss: {loss:.5f}\tTop1 accuracy: {top1_train_accuracy.item()}\tTop1 valid accuracy: {top1_valid_accuracy.item()}\tLR: {self.scheduler.get_last_lr()}")

            if (epoch_counter + 1) % self.args.checkpoint_n_steps == 0:
                checkpoint_name = '%s_%04d.pth.tar'%(self.args.process, epoch_counter+1)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'state_dict': self.model.module.state_dict()}, is_best = False, filename = os.path.join(self.writer.log_dir, checkpoint_name))

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def get_features(self, test_loader):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.args.device)
                features = self.model.module(images.type(self.dtype))

                all_features.append(features)
                all_labels.append(labels[1])

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    def tsne(self, test_loader):
        self.model.eval()
        test_features, test_labels = self.get_features(test_loader)
        tsne = TSNE(n_components=2, perplexity=self.args.per).fit_transform(test_features)

        fig, ax = plt.subplots()
        scatter = ax.scatter(tsne[:,0], tsne[:,1], c = test_labels, cmap = 'tab10', s=self.args.ms)
        handles, labels = scatter.legend_elements(prop = "colors")
        labels = ['UWF/no RDR', 'UWF/RDR', 'fundus/non RDR', 'fundus/RDR']
        ax.set_title('TSNE scatter plot for the new RDR framework')
        ax.legend(handles, labels, loc = "upper right", title = "classes")
        plt.show()
        ax.figure.savefig('/home/pwuaj/hkust/DR/tsne.png')

    def eval(self, test_loader):
        self.model.eval()

        logging.info(f"Testing with gpu: {not self.args.disable_cuda}.")
        logging.info(f"Total GPU device: {self.args.device_count}.")

        top1_accuracy = 0
        result = []
        with torch.no_grad():
            for counter, (img, lbl) in enumerate(test_loader):

                img = img.to(self.args.device)
                path = lbl[0]
                lbl = lbl[1].to(self.args.device)

                logits = self.model.module(img.type(self.dtype))
                try:
                    logits = logits.logits
                except:
                    pass
                prob = nn.Softmax(dim=1)(logits)[:,1]
                top1, predict = topacc(logits, lbl, topk=(1,), predict = True)
                top1_accuracy += top1[0]
                result.append(pd.DataFrame({'Path':path, 'True label': lbl.cpu().numpy(), 'Predicted label': predict, 'model output': prob.cpu().numpy()}))

        top1_accuracy /= (counter + 1)
        result = pd.concat(result, ignore_index=True)
        result.to_csv(os.path.join(self.writer.log_dir, self.args.test) + '.csv')
        logging.debug(f"Top1 Test accuracy: {top1_accuracy.item()}")
        print(f"Top1 Test accuracy: {top1_accuracy.item()}")

    def class_activation(self, test_loader):
        activation = []
        predicts = []
        def hook(model, input, output):
            activation.append(output.clone().detach())


        self.model.eval()
        self.model.module.backbone.layer4.register_forward_hook(hook)
        weight = self.model.module.state_dict()['backbone.fc.weight'].detach() #(2, 2048)

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        data_transforms = transforms.Compose([
                                              Resize(self.args.resize, interpolation=BICUBIC),
                                              CenterCrop((self.args.resize, int(self.args.resize*1.25))),
                                              _convert_image_to_rgb,
                                              ToTensor(),
                                              # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                              ])

        with torch.no_grad():
            for image, lbl in tqdm(test_loader):
                path= lbl[0][0]
                name = os.path.split(path)[1]
                lbl = lbl[1].to(self.args.device)
                feature = self.model.module(image.to(self.args.device))
                top1, predict = topacc(feature, lbl, topk=(1,), predict = True)
                features = activation[-1]   #(1, 2048, 16, 20)
                # predicts = torch.from_numpy(predict)
                predicts = torch.tensor([1])
                weight_winner = weight[predicts, :].unsqueeze(2).unsqueeze(3) # (1, 2048, 1, 1)
                cam = (weight_winner * features).sum(1, keepdim=True)
                final_cam = F.interpolate(cam, (self.args.resize, int(self.args.resize*1.25)), mode="bilinear", align_corners=True)

                image = Image.open(path)
                image = data_transforms(image)

                if lbl.item() ==1:
                    plt.figure()
                    plt.imshow(np.asarray(image).squeeze().transpose(1,2,0))
                    plt.imshow(final_cam.squeeze().detach().cpu().numpy(), alpha=0.25, cmap = self.args.cmap)
                    plt.savefig(os.path.join(*['/home/pwuaj/data/cam/1', name]))
                    plt.close()

                elif lbl.item() ==0:
                    plt.figure()
                    plt.imshow(np.asarray(image).squeeze().transpose(1,2,0))
                    plt.imshow(final_cam.squeeze().detach().cpu().numpy(), alpha=0.25, cmap = self.args.cmap)
                    plt.savefig(os.path.join(*['/home/pwuaj/data/cam/0', name]))
                    plt.close()


    def gradcam(self, test_loader):
        self.model.eval()
        targets = [ClassifierOutputTarget(0)]
        target_layers = [self.model.module.backbone.layer4[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        data_transforms = transforms.Compose([
                                              Resize(self.args.resize, interpolation=BICUBIC),
                                              CenterCrop((self.args.resize, int(self.args.resize*1.25))),
                                              _convert_image_to_rgb,
                                              ToTensor(),
                                              # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                              ])

        for image, lbl in tqdm(test_loader):
            path= lbl[0][0]
            name = os.path.split(path)[1]
            lbl = lbl[1].to(self.args.device)
            grayscale_cams = cam(image.to(self.args.device), targets=targets)
            img = Image.open(path)
            img = data_transforms(img)
            img = np.asarray(img).squeeze().transpose(1,2,0)
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

            if lbl.item() ==1:
                images = np.hstack((np.uint8(255*img), cam_image))
                image = Image.fromarray(images)
                image.save(os.path.join(*['/home/pwuaj/data/cam/1', name]))

            elif lbl.item() ==0:
                images = np.hstack((np.uint8(255*img), cam_image))
                image = Image.fromarray(images)
                image.save(os.path.join(*['/home/pwuaj/data/cam/0', name]))
