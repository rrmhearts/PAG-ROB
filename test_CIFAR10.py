import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset
import argparse
from models.resnet import *
from models.ViT import ViT
from pgd_attack import eval_adv_test_whitebox
import wandb
import numpy as np
import random
import yaml
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do PAG Imply Robustness?')
    parser.add_argument('--config_path', type=str, default='./configs/cifar10_sbg_rn18.yaml', help='training config path')
    args, _ = parser.parse_known_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["chekpoint_folder"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reproducibility
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])


    model_name = f'{config["arch"]}-cifar10-grad_source-{config["grad_source"]}-pag_coeff-{config["pag_coeff"]}' \
                f'-subsample-{config["num_grads_per_image"]}-seed-{config["seed"]}'

    if config["use_wandb"]:
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
        wandb.run.name = model_name

    # get data - train
    data_stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

    # get data - test
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # get model
    # if config["arch"] == 'vit':
    #     model = torch.nn.DataParallel(ViT()).cuda()
    # else:
    model = torch.nn.DataParallel(ResNet18()).cuda()
    # print architecture details
    print(f'Using {config["arch"]} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} learnable params')

    # get optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])

    model.load_state_dict(torch.load("models/weights/rn18-c10-SBG.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    c_acc, r_acc = eval_adv_test_whitebox(model=model, device=device, test_loader=testloader, epsilon=0.5, num_steps=7,
                                            step_size=1.5 * 0.5 / 7, random=True, verbose=False, norm='l_2',
                                            stats=data_stats)
    if config["use_wandb"]:
        wandb.log({"L2: c_acc": c_acc, "r_acc": r_acc})
    print(f"L2: c_acc: {c_acc} r_acc: {r_acc}")

    c_acc, r_acc = eval_adv_test_whitebox(model=model, device=device, test_loader=testloader, epsilon=0.5, num_steps=7,
                                            step_size=1.5 * 0.5 / 7, random=True, verbose=False, norm='l_inf',
                                            stats=data_stats)
    if config["use_wandb"]:
        wandb.log({"Linf: c_acc": c_acc, "r_acc": r_acc})
    print(f"Linf: c_acc: {c_acc} r_acc: {r_acc}")