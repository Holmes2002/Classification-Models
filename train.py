import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import get_dataloader_train
from torchvision import transforms
import os
import argparse
import utils
from utils import get_lr
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--num_classes",
        # default='CarBandDataset',
        type=int,
        required = True,
        help="Total number of classes",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=512,
        help="The size of embedding",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default='laion2b_s34b_b88k',
        help="The pretrained weight of model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Choose Backbone for model",
    )

    parser.add_argument(
        "--backup_fol", 
        type=str, 
        required=True,
        help="Destination to save checkpoints",
    )
    parser.add_argument(
        "--eval-train",
        default=False,
        action="store_true",
        help="Whether or not to evaluate training results."
    )
    parser.add_argument(
        "--freeze",type = str,
        default=False,
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        '--gpu-id', default='0', type=str,
        help='id(s) for CUDA_VISIBLE_DEVICES'
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args


device = torch.device('cuda')
def val(val_loader,model,type = 'val'):
  print("EVAL START: ....")
  model.eval()
  acc = 0
  j = 0
  iter_train = iter(val_loader)
  for i in tqdm(val_loader):
    j+=1
    img, label = next(iter_train)
    img, label = img.to(device), label.to(device)
    predict = model(img)
    predict - torch.softmax(predict, dim=1)
    predict = torch.argmax(predict,dim =1)
    acc += accuracy_score(predict.cpu().numpy(), label.cpu().numpy())
  print(f"{type} :",acc/j)
  print("EVAL END: .....")
  return acc/j
def main(args):
    log_file = open('log_file.txt', 'w')
    max_acc = 0
    train_transform, test_transform = utils.get_transforms(args.backbone)
    train_loader = get_dataloader_train(args.root,'train', train_transform,args.batch_size)
    num_batches = len(train_loader)
    val_loader = get_dataloader_train(args.root,'test', test_transform,4)
    model = utils.get_model(args)
    params      = [p for name, p in model.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model.named_parameters() if p.requires_grad]
    if args.freeze :
            model.requires_grad_(False)
            model.model.fc.requires_grad_(True)
    print('  - Total {} params to training: {}'.format(len(params_name), [pn for pn in params_name]))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = utils.cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    print(f'  - Init  with cosine learning rate scheduler {args.lr}')
    print(f'  - Distill model from   to {args.backbone}')
    params      = [p for name, p in model.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model.named_parameters() if p.requires_grad]
    print('  - Total {} params to training: {}'.format(len(params_name), [pn for pn in params_name]))
    cross = torch.nn.CrossEntropyLoss()

    # state_dict = torch.load('backup/head_model.pt')
    # model.load_state_dict(state_dict)

    for epoch in range(0,args.epochs):
        model.train()
        iter_train = iter(train_loader)
        # acc_val = val(val_loader,model)

        if args.freeze :
            model.requires_grad_(False)
            model.model.fc.requires_grad_(True)
        count = 0
        # acc_val = val(val_loader,model)

        acc_val = val(val_loader,model)
        # acc_train = val(train_loader,model,type='train')
        
        if acc_val > max_acc:
                    os.makedirs(args.backup_fol, exist_ok=True)
                    torch.save(model.state_dict(),f"{args.backup_fol}/best_{epoch}.pt")
                    max_acc = acc_val

        for i in tqdm(train_loader):
            step = count + epoch * num_batches
            count+=1
            lr = scheduler(step)
            optimizer.zero_grad()
            img, label = next(iter_train)
            img, label = img.to(device), label.to(device)
            # img = torch.tensor(img, requires_grad=True)
            predict = model(img)
            loss  = cross(predict, label)
            
            loss.backward()
            optimizer.step()
            print(f"EPOCH {epoch}, Loss {loss} max_acc {max_acc}")
        log_file.write(f"EPOCH {epoch}, Loss {loss} Acc_val {acc_val}  Best_acc {max_acc}"+'\n')
if __name__ == '__main__':
    args = parse_arguments()
    main(args)