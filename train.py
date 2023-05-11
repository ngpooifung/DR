# %%
import argparse
import torch
import numpy as np
import clip
from torch import nn
import torch.backends.cudnn as cudnn
from dataset import Modeldataset, Imagefolder
from model import modeltrainer
from trainer import Restrainer, Classictrainer
from exceptions import NotImplementError
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# %%
parser = argparse.ArgumentParser(description='DR')
function_names = ['main', 'eval', 'Clip', 'Clipfulltune', 'Cliplayertune']

# %% process
parser.add_argument('--process', choices=function_names,
                    help = 'process: ' +  ' | '.join(function_names))

# Data
parser.add_argument('--dir', type = str,
                    help = 'Path to training data directory')
parser.add_argument('--test-dir', type = str, default = None,
                    help = 'Path to test data directory')
parser.add_argument('--clip_csv', type = str, default = None,
                    help = 'Path to clip text')
parser.add_argument('--resize', default = (384, 480), nargs = 2, type = int,
                    help = 'resize images in training')
parser.add_argument('--min_size', default = (100, 100), nargs = 2, type = int,
                    help = 'min image size in training')
parser.add_argument('--output', type = str, default = '',
                    help = 'Path to output folder')

# Model
parser.add_argument('--arch', default='resnet50', type = str,
                    help = 'model architecture')
parser.add_argument('--out_dim', default=2, type=int,
                    help = 'feature dimension (default: 2)')

# Training
parser.add_argument('--epochs', type = int, default = 100,
                    help = 'Number of training epochs')
parser.add_argument('--lr', type = float, default = 0.0003,
                    help = 'Learning rate')
parser.add_argument('--weight_decay', type = float, default = 1e-4,
                    help = 'Weight decay')
parser.add_argument('--disable_cuda', action = 'store_true',
                    help = 'Disable CUDA')
parser.add_argument('--checkpoint-n-steps', default = 50, type = int,
                    help = 'Save checkpoint every n steps')
parser.add_argument('--workers', type = int, default = 16,
                    help = 'Number of data loading workers')
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'Batch size')
parser.add_argument('--checkpoint_n_steps', type = int, default = 50,
                    help = 'Save checkpoins per n steps')
parser.add_argument('--clip_grad_norm', type = float, default = 5,
                    help = 'Clip gradients norm (0 to disable)')
parser.add_argument('--finetune', default='',
                    help='finetune from checkpoint')
parser.add_argument('--use_mlp', action='store_true',
                    help='Perform mlp')
parser.add_argument('--no_visual_proj', action='store_true',
                    help='no visual projection')
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True, sampler = train_sampler)

    if args.test_dir is not None:
        test_dataset = Modeldataset(args.test_dir).get_dataset(resize = args.resize, transform = False)
        test_sampler = DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True, sampler = test_sampler)
    else:
        test_loader = None

    model = modeltrainer()._get_model(base_model = args.arch, out_dim = args.out_dim).to(args.device)
    model = DDP(model, device_ids = [local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)

    trainer = Restrainer(model, optimizer, scheduler, args)
    trainer.train(train_loader, test_loader)

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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True, sampler = test_sampler)

    path = os.path.join(args.output, args.finetune)
    checkpoint = torch.load(path, map_location = args.device)
    state_dict = checkpoint['state_dict']

    model = modeltrainer()._get_model(base_model = args.arch, out_dim = args.out_dim).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model = DDP(model, device_ids = [local_rank], output_device=local_rank)

    optimizer = None
    scheduler = None
    trainer = Restrainer(model, optimizer, scheduler, args)
    trainer.eval(test_loader)

def Clip():
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

    model, preprocess = clip.load(args.arch, device=args.device)
    train_dataset = Modeldataset(args.dir).get_dataset(resize = args.resize, transform = True, preprocess = preprocess)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True, sampler = train_sampler)

    if args.test_dir is not None:
        test_dataset = Modeldataset(args.test_dir).get_dataset(resize = args.resize, transform = False, preprocess = preprocess)
        test_sampler = DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True, sampler = test_sampler)
    else:
        test_loader = None

    if args.finetune:
        path = os.path.join(args.output, args.finetune)
        checkpoint = torch.load(path, map_location = args.device)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=True)
        model = model.to(args.device)
    if args.no_visual_proj:
        model.visual.proj = None
    model = DDP(model, device_ids = [local_rank], output_device=local_rank)

    optimizer = None
    scheduler = None
    trainer = Classictrainer(model, optimizer, scheduler, args)
    trainer.Logistic(train_loader, test_loader)

def Cliptune():
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

    model, preprocess = clip.load(args.arch, device=args.device, jit=False) #ViT-B/16
    train_dataset = Modeldataset(args.dir).get_dataset(resize = args.resize, transform = True, preprocess = preprocess, clip_csv = args.clip_csv)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True, sampler = train_sampler)

    if args.process == 'Cliplayertune':
        model.visual.proj = None
        for name, param in model.named_parameters():
            # if name not in ['visual.proj', 'text_projection']:
            if name not in ['ffn.0.weight', 'ffn.0.bias', 'ffn.2.weight', 'ffn.2.bias', 'text_projection']:
                param.requires_grad = False

    model = DDP(model, device_ids = [local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)
    trainer = Classictrainer(model, optimizer, scheduler, args)
    trainer.finetune(train_loader)

def allocate():
    torch.set_num_threads(args.workers)
    if args.process == 'main':
        main()
    elif args.process == 'eval':
        eval()
    elif args.process == 'Clip':
        Clip()
    elif args.process == 'Cliplayertune':
        Cliptune()
    elif args.process == 'Clipfulltune':
        Cliptune()

if __name__ == "__main__":
    allocate()
