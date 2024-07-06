# %%
import argparse
import torch
import torchvision
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from dataset import Modeldataset, Imagefolder
from model import modeltrainer
from trainer import Restrainer
from exceptions import NotImplementError
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
torch.manual_seed(0)
# %%
parser = argparse.ArgumentParser(description='DR')
function_names = ['main', 'eval',  'transfer', 'tsne','class_activation']

# %% process
parser.add_argument('--process', choices=function_names,
                    help = 'process: ' +  ' | '.join(function_names))

# Data
parser.add_argument('--dir', type = str,
                    help = 'Path to training data directory')
parser.add_argument('--dir2', type = str, default = None,
                    help = 'Path to training data directory2')
parser.add_argument('--test-dir', type = str, default = None,
                    help = 'Path to test data directory')
parser.add_argument('--resize', default = 336, type = int,
                    help = 'resize images in training')
parser.add_argument('--output', type = str, default = '',
                    help = 'Path to output folder')
parser.add_argument('--test', type = str, default = 'test',
                    help = 'Path to output file')

# Model
parser.add_argument('--arch', default='resnet50', type = str,
                    help = 'model architecture')
parser.add_argument('--out_dim', default=2, type=int,
                    help = 'feature dimension (default: 2)')

# Training
parser.add_argument('--epochs', type = int, default = 100,
                    help = 'Number of training epochs')
parser.add_argument('--weight', type = int, nargs = 2, default = (1, 1),
                    help = 'weight')
parser.add_argument('--lr', type = float, default = 0.0001,
                    help = 'Learning rate')
parser.add_argument('--wf', type = float, default = 0.1,
                    help = 'Learning rate')
parser.add_argument('--dropout', type = float, default = 0.,
                    help = 'dropout')
parser.add_argument('--weight_decay', type = float, default = 1e-4,
                    help = 'Weight decay')
parser.add_argument('--disable_cuda', action = 'store_true',
                    help = 'Disable CUDA')
parser.add_argument('--fundus', action = 'store_true',
                    help = 'fundus pretraining')
parser.add_argument('--workers', type = int, default = 16,
                    help = 'Number of data loading workers')
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'Batch size')
parser.add_argument('--per', type = int, default = 30,
                    help = 'perplexity')
parser.add_argument('--ms', type = float, default = 1,
                    help = 'markersize')
parser.add_argument('--checkpoint_n_steps', type = int, default = 50,
                    help = 'Save checkpoins per n steps')
parser.add_argument('--finetune', default='',
                    help='finetune from checkpoint')
parser.add_argument('--cmap', type = str,
                    help = 'cmap')
args = parser.parse_args()


# %%
def main():
    if not args.disable_cuda and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        print(local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend = 'nccl')
        args.device = torch.device('cuda', local_rank)
        args.device_count = torch.cuda.device_count()
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
        args.device_count = -1

    train_dataset = Modeldataset(args.dir).get_dataset(resize = args.resize, transform = True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, sampler = train_sampler)

    if args.dir2 is not None:
        train_dataset2 = Modeldataset(args.dir2).get_dataset(resize = args.resize, transform = True)
        train_sampler2 = DistributedSampler(train_dataset2)
        train_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, sampler = train_sampler2)
    else:
        train_loader2 = None

    if args.test_dir is not None:
        test_dataset = Modeldataset(args.test_dir).get_dataset(resize = args.resize, transform = False)
        test_sampler = DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, sampler = test_sampler)
    else:
        test_loader = None

    model = modeltrainer()._get_model(base_model = args.arch, out_dim = args.out_dim, dropout = args.dropout).to(args.device)

    if args.fundus:
        path = os.path.join('RDR', args.finetune)
        checkpoint = torch.load(path, map_location = args.device)
        state_dict = checkpoint['state_dict']
        log = model.load_state_dict(state_dict, strict=True)
        print(log)
        # for name, param in model.named_parameters():
        #     if name not in ['backbone.fc.0.weight', 'backbone.fc.0.bias','backbone.fc.3.weight', 'backbone.fc.3.bias']:
        #         param.requires_grad = False
        model = model.to(args.device)

    if args.process == 'transfer':
        path = os.path.join(args.output, args.finetune)
        checkpoint = torch.load(path, map_location = args.device)
        state_dict = checkpoint['state_dict']
        log = model.load_state_dict(state_dict, strict=False)
        model.backbone.fc = nn.Linear(2048, 2)
        print(log)
        for name, param in model.named_parameters():
            if name not in ['backbone.fc.weight', 'backbone.fc.bias']:
                param.requires_grad = False

        model = model.to(args.device)
    # model,_ = clip.load('RN50', device = args.device)
    model = DDP(model, device_ids = [local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)

    trainer = Restrainer(model, optimizer, scheduler, args)
    trainer.train(train_loader, test_loader, train_loader2)

def eval():
    if not args.disable_cuda and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend = 'nccl')
        args.device = torch.device('cuda', local_rank)
        args.device_count = torch.cuda.device_count()
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
        args.device_count = -1

    test_dataset = Modeldataset(args.test_dir).get_dataset(resize = args.resize, transform = False)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, sampler = test_sampler)

    path = os.path.join(args.output, args.finetune)
    checkpoint = torch.load(path, map_location = args.device)
    state_dict = checkpoint['state_dict']

    model = modeltrainer()._get_model(base_model = args.arch, out_dim = args.out_dim, dropout = args.dropout).to(args.device)
    if args.process == 'class_activation':
        model.backbone.fc = nn.Linear(model.backbone.fc[0].in_features, 2)
    log = model.load_state_dict(state_dict, strict=True)
    if args.process == 'tsne':
        model.backbone.fc[3] = nn.Identity()
    model = model.to(args.device)
    model = DDP(model, device_ids = [local_rank], output_device=local_rank)

    optimizer = None
    scheduler = None
    trainer = Restrainer(model, optimizer, scheduler, args)
    if args.process == 'eval':
        trainer.eval(test_loader)
    elif args.process == 'class_activation':
        trainer.class_activation(test_loader)
    elif args.process == 'tsne':
        trainer.tsne(test_loader)


def allocate():
    torch.set_num_threads(args.workers)
    if args.process == 'main':
        main()
    elif args.process == 'transfer':
        main()
    elif args.process == 'eval':
        eval()
    elif args.process == 'class_activation':
        eval()
    elif args.process == 'tsne':
        eval()


if __name__ == "__main__":
    allocate()
