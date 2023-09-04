
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
        "--pretrained_student",
        type=str,
        default='',
        help="The pretrained weight of model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
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
    parser.add_argument(
        '--optim', default='Adam', type=str,
        help='optimizer used to optimize '
    )
    parser.add_argument(
        '--momentum', default=0.9, type=float,
        help='Momentum used in SGD'
    )
    parser.add_argument(
        '--model_teacher', default='unicom', type=str,
        help='Name of model teahcer used'
    )
    parser.add_argument(
        '--pretrained_teacher',required = True, type=str,
        help='Name of model teahcer used'
    )
    parser.add_argument(
        '--alpha',default = 0.5, type=float,
        help='Name of model teahcer used'
    )

    parser.add_argument(
        '--temperature',default = 3, type=int,
        help='Name of model teahcer used'
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args

errors = 0
device = torch.device('cuda')
def val(val_loader,model,type = 'val'):
  global errors
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
    predict = torch.softmax(predict, dim=1)
    predict = torch.argmax(predict,dim =1)
    test_predict = predict.cpu().numpy()
    test_label = label.cpu().numpy()
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
    val_loader = get_dataloader_train(args.root,'test', test_transform,args.batch_size)

    model_teacher = utils.get_pretrained_unicom('ViT-B/32', args.num_classes).to(device)
    # model_teacher.load_state_dict(torch.load( args.pretrained_teacher ))
    if args.backbone == 'resnet':
      model_student = utils.get_pretrained_resnet(num_classes = args.num_classes).to(device)
    elif args.backbone == 'deit':
      model_student = utils.get_pretrained_deit(num_classes = args.num_classes).to(device)
    elif args.backbone == 'pit':
      model_student = utils.get_pretrained_pit(num_classes = args.num_classes).to(device)
    elif args.backbone == 'LeVit':
      model_student = utils.get_pretrained_LeVit(num_classes = args.num_classes).to(device)
    elif args.backbone == 'SepViT':
      model_student = utils.get_pretrained_SepViT(num_classes = args.num_classes).to(device)
    elif args.backbone == 'dino':
      model_student = utils.get_pretrained_dino(num_classes = args.num_classes).to(device)
    elif args.backbone == 'dinov2':
      model_student = utils.get_pretrained_dinov2(num_classes = args.num_classes).to(device)
    elif args.backbone == 'TinyVit':
      model_student = utils.get_pretrained_TinyVit(num_classes = args.num_classes).to(device)

    else:
      model_student = utils.get_pretrained_unicom().to(device)
    params      = [p for name, p in model_student.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model_student.named_parameters() if p.requires_grad]
    for name, p in model_student.named_parameters():
      p.requires_grad_(False)
    model_student.model.fc.requires_grad_(True)
    print('  - Total {} params to training: {}'.format(len(params_name), [pn for pn in params_name]))
    if args.optim == "Adam":
      optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    else:
      optimizer = torch.optim.SGD(params, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.wd)

    scheduler = utils.cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    print(f'  - Init {args.optim} with cosine learning rate scheduler {args.lr}')
    print(f'  - Distill model from  {args.model_teacher} to {args.backbone}')
    if args.pretrained_student:
      model_student.load_state_dict(torch.load(args.pretrained_student))
      print(f'  - Load pretrained from: {args.pretrained_student}')
    print(model_student)
    cross = torch.nn.CrossEntropyLoss()
    model_teacher.eval()
    for name, p in model_teacher.named_parameters():
      p.requires_grad_(False)
    for epoch in range(4,args.epochs):
        model_student.train()
        iter_train = iter(train_loader)
        acc_val = val(val_loader,model_student)
        for name, p in model_student.named_parameters():
          p.requires_grad_(False)
        model_student.model.fc.requires_grad_(True)

        count = 0
        for i in tqdm(train_loader):
            step = count + epoch * num_batches
            count+=1
            scheduler(step)
            optimizer.zero_grad()
            img, label = next(iter_train)
            img, label = img.to(device), label.to(device)
            # img = torch.tensor(img, requires_grad=True)
            predict_student = model_student(img)
            predict_teacher = model_teacher(img)
            loss_KD  = utils.loss_fn_kd(predict_student, label, predict_teacher, args)
            
            loss_KD.backward()
            optimizer.step()
            print(f"EPOCH {epoch}, Loss {loss_KD}  ")

        acc_val = val(val_loader,model_student)
        
        if acc_val > max_acc:
                    os.makedirs(args.backup_fol, exist_ok=True)
                    torch.save(model_student.state_dict(),f"{args.backup_fol}/best_{epoch}.pt")
                    max_acc = acc_val
        log_file.write(f"EPOCH {epoch}, Loss {loss_KD} Acc_val {acc_val}  Best_acc {max_acc}"+'\n')
if __name__ == '__main__':
    args = parse_arguments()
    main(args)