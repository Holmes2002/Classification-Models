from torchvision import models
import torch.nn as nn
import numpy as np
import torch
import unicom
from torchvision import transforms
import PIL
from SepVit import SepVit
from LeVit import LeVit
from MobileVit import MobileVit
from TinyVit.tiny_vit import tiny_vit_5m_224


class WarpModule(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self,  x):
        x = self.model(x)
        x = self.model.fc(x)
        return x

        
def get_transform(
        image_size: int = 224,
        is_train: bool = True
):
    from timm.data import create_transform
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if image_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(image_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
    t.append(transforms.CenterCrop(image_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):

        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
                print("LR {lr}", end = ' ')
            else:

                
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def get_pretrained_unicom(model_name = 'ViT-B/32', num_classes = 1000 ):
    model, transform_clip = unicom.load(model_name)
    model.fc = nn.Linear(512, num_classes)
    model2 = WarpModule(model)

    return model2
def get_pretrained_resnet(num_classes):
    """
    Fetches a pretrained resnet model (downloading if necessary) and chops off the top linear
    layer. If new_fc_dim isn't None, then a new linear layer is added.
    :param new_fc_dim: 
    :return: 
    """
    resnet152 = models.resnet34(pretrained=True)
    resnet152.fc = nn.Linear(512, num_classes, bias=True)
    return resnet152 


def get_transforms(backbone):
  if backbone != 'unicom':
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),      # Random rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
  else:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
  return train_transform, test_transform


def loss_fn_kd(outputs, labels, teacher_outputs, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KL = torch.nn.KLDivLoss(reduction='sum',log_target=True)
    alpha = args.alpha
    T = args.temperature
    KL_loss = KL(torch.nn.functional.log_softmax(outputs/T, dim=1),
                             torch.nn.functional.log_softmax(teacher_outputs/T, dim=1)) * (T*T*alpha/ outputs.numel())
    CE_loss = torch.nn.functional.cross_entropy(outputs, labels) * (1. - alpha)
    KD_loss =  KL_loss + CE_loss
    print('')
    print(f'Loss KL {KL_loss} Loss CE {CE_loss}', end = ' ')
              

    return KD_loss
def get_pretrained_deit(num_classes):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model.fc = torch.nn.Linear(1000, num_classes)
    model2 = WarpModule(model)
    
    return model2
def get_pretrained_pit(num_classes):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model.fc = torch.nn.Linear(1000, num_classes)
    model2 = WarpModule(model)

    return model2
def get_pretrained_LeVit(num_classes):
    model = LeVit.LeViT_128S(num_classes=1000, distillation=True,
              pretrained=True)
    model.fc = torch.nn.Linear(1000, num_classes)
    model2 = WarpModule(model)
    
    return model2

def get_pretrained_SepVit(num_classes):

    model = SepViT.SepViT(
        num_classes = 1000,
        dim = 32,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
        dim_head = 32,          # attention head dimension
        heads = (1, 2, 4, 8),   # number of heads per stage
        depth = (1, 2, 6, 2),   # number of transformer blocks per stage
        window_size = 7,        # window size of DSS Attention block
        dropout = 0.1           # dropout
        )    
    model.fc = torch.nn.Linear(1000, num_classes)
    model2 = WarpModule(model)
    
    return model2
def get_pretrained_MobileViT(num_classes):
    model = MobileVit.mobilevit_s()
    model.fc = torch.nn.Linear(1000, num_classes)
    model2 = WarpModule(model)
    return model2
def get_pretrained_TinyViT(num_classes):

    model = tiny_vit_5m_224(pretrained=True)
    model.load_state_dict(torch.load('/content/Imagenet_Unicom/tiny_vit_5m_22k_distill.pth')['model'])
    model.fc = torch.nn.Linear(21841, num_classes)
    model2 = WarpModule(model)
    
    return model2
def get_pretrained_dinov2(num_classes):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.fc = torch.nn.Linear(768, num_classes)
    model2 = WarpModule(model)
    
    return model2

def get_model(args):
    device = 'cuda'
    if args.backbone == 'resnet':
      model = get_pretrained_resnet(num_classes = args.num_classes).to(device)
    elif args.backbone == 'deit':
      model = get_pretrained_deit(num_classes = args.num_classes).to(device)
    elif args.backbone == 'pit':
      model = get_pretrained_pit(num_classes = args.num_classes).to(device)
    elif args.backbone == 'LeVit':
      model = get_pretrained_LeVit(num_classes = args.num_classes).to(device)
    elif args.backbone == 'SepViT':
      model = get_pretrained_SepViT(num_classes = args.num_classes).to(device)
    elif args.backbone == 'dino':
      model = get_pretrained_dino(num_classes = args.num_classes).to(device)
    elif args.backbone == 'dinov2':
      model = get_pretrained_dinov2(num_classes = args.num_classes).to(device)
    elif args.backbone == 'TinyVit':
      model = get_pretrained_TinyVit(num_classes = args.num_classes).to(device)
    else:
      model = get_pretrained_unicom().to(device)
    return model