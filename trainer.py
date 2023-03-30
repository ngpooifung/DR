# %%
import numpy as np
import torch
from utils import topacc, save_checkpoint, bceacc
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import os
import logging
from tqdm import tqdm
import pandas as pd
torch.manual_seed(0)

# %%
class Restrainer(object):
    def  __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        # self.criterion = torch.nn.BCELoss().to(self.args.device)
        log_dir = self.args.dir
        self.writer = SummaryWriter(log_dir = log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def train(self, train_loader, test_loader = None):
        self.model.train()

        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        logging.info(f"Total GPU device: {self.args.device_count}.")

        for epoch_counter in range(self.args.epochs):
            train_loader.sampler.set_epoch(epoch_counter)
            test_loader.sampler.set_epoch(epoch_counter)
            top1_train_accuracy = 0
            top1_valid_accuracy = 0
            for counter, (img, lbl) in enumerate(train_loader):
                img = img.to(self.args.device)
                lbl = lbl[1].to(self.args.device)

                logits = self.model(img)
                try:
                    logits = logits.logtis
                except:
                    pass
                # logits = logits.squeeze()
                loss = self.criterion(logits, lbl)

                top1 = topacc(logits, lbl, topk = (1,))
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

                        logits = self.model(img)
                        try:
                            logits = logits.logtis
                        except:
                            pass
                        top1 = topacc(logits, lbl, topk = (1,))
                        top1_valid_accuracy += top1[0]
                    top1_valid_accuracy /= (counter + 1)

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1_train_accuracy.item()}\tTop1 valid accuracy: {top1_valid_accuracy.item()}\tLR: {self.scheduler.get_last_lr()}")

            if (epoch_counter + 1) % self.args.checkpoint_n_steps == 0:
                checkpoint_name = '%s_%04d.pth.tar'%(self.args.process, epoch_counter)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'state_dict': self.model.module.state_dict()}, is_best = False, filename = os.path.join(self.writer.log_dir, checkpoint_name))

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

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

                logits = self.model(img)
                try:
                    logits = logits.logtis
                except:
                    pass
                loss = self.criterion(logits, lbl)

                top1, predict = topacc(logits, lbl, topk=(1,), predict = True)
                top1_accuracy += top1[0]
                result.append(pd.DataFrame({'Path':path, 'True label':lbl.cpu().numpy(), 'Predicted label': predict}))

        top1_accuracy /= (counter + 1)
        result = pd.concat(result, ignore_index=True)
        result.to_csv(os.path.join(self.writer.log_dir, 'test.csv'))
        logging.debug(f"Top1 Test accuracy: {top1_accuracy.item()}")
