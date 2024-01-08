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
import matplotlib.pyplot as plt
import random
import yaml
import os
import ssl
from collections import defaultdict
import art
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

ssl._create_default_https_context = ssl._create_unverified_context

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# train function
def train_func(model, train_loader, optimizer):
    centroids = defaultdict(lambda: 0, {})
    model.train()
    #
    for batch_idx, (images, targets) in enumerate(train_loader):
        # if device == "cuda":
        images, targets = images.cuda(), targets.cuda()
        # images.requires_grad = True
        # 4, 3, 32, 32 .... 4
        # print(f"images: {images.shape}, target: {target.shape}")
        # print(f"images min: {torch.min(images)}, max: {torch.max(images)}")
        optimizer.zero_grad()

        pred = model(aug(images))
        latents = model.latent(aug(images))

        # init_pred = pred.max(1, keepdim=True)[1] # get the index of the max     log-probability

        # CE loss
        criterion = torch.nn.CrossEntropyLoss()

        loss_ce = criterion(pred, targets)

        # Step 3: Create the ART classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(-1, 1), # min and max pixel value
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 32, 32),
            nb_classes=10,
        )
        # loss_ce.backward(retain_graph=True)
        # model.zero_grad()
        # loss_ce.backward(retain_graph=True)
        # epsilon = 0.03
        # images_grad = images.grad.data
        # perturbed_data = fgsm_attack(images, epsilon, images_grad)
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.03)
        # numpy_images = np.transpose(images.cpu().numpy(), (0, 3, 1, 2)).astype(np.float32)
        numpy_images = images.cpu().numpy().astype(np.float32)

        # perturbed_data = torch.from_numpy(np.transpose(attack.generate(x=numpy_images), (0, 2, 3, 1)) )
        perturbed_data = torch.from_numpy(attack.generate(x=numpy_images)).cuda()
        perturbed_data = torch.nan_to_num(perturbed_data, nan=0)
        perturbed_latents = model.latent(aug(perturbed_data))
        # PAG loss
        pag_loss = 0
        if pag_coeff != 0:# and images.grad is not None:
            # print(f"images grad: {images.grad.shape}")
            # each in batch
            # for rep in range(config["batch_size"]):
            centroid_accumulator = defaultdict(lambda: 0, {})
            class_counter = defaultdict(lambda: 0, {})

            # Accumulate latents for current centroid
            for lat, label in zip(latents, labels):
                # print(target[rep])
                # print(image, label)
                # image_latent = model.latent(torch.unsqueeze(image, dim=0))
                class_counter[label] += 1
                centroid_accumulator[label] = \
                    lat + centroid_accumulator[label]
            
            centroid_latent_vector = torch.clone(latents)
            # Add centroid to longterm stored value
            for i, class_center in enumerate(centroid_accumulator.keys()):
                centroids[class_center] *= 9
                centroids[class_center] += (centroid_accumulator[class_center]/class_counter[label])
                centroids[class_center] /= 10
                centroid_latent_vector[i] -= centroids[class_center]

            # print(centroids)
            pag_loss += torch.mean(torch.nn.CosineSimilarity(dim=1)\
                (latents-perturbed_latents, centroid_latent_vector))
# Train loss in batch 500: -1.3597179651260376 | CE loss: 0.12432830035686493 | PAG loss (before | after coeff): -0.9893641471862793 | -1.484046220779419
# 1.4810724258422852
# Pag loss: -0.987381637096405
# Train loss in batch 700: -1.2283903360366821 | CE loss: 0.25120019912719727 | PAG loss (before | after coeff): -0.9863936901092529 | -1.4795905351638794
# Pag loss: -0.9863936901092529
# ================================================================
# clean accuracy:  0.6635
# robust accuracy:  0.5893
        model.zero_grad()
        loss = loss_ce + pag_coeff * pag_loss
        # report training statistics
        if batch_idx % 100 == 0:
            print(f'Train loss in batch {batch_idx}: {loss} | CE loss: {loss_ce} | PAG loss (before | after coeff): '
                  f'{pag_loss} | {pag_coeff * pag_loss}')
            print(f"Pag loss: {pag_loss}")

        loss.backward()
        optimizer.step()

def adjust_learning_rate(optimizer, epoch):
    """a multistep learning rate drop mechanism"""
    lr = config["lr"]
    epochs = config["epochs"]
    if epoch >= 0.5 * epochs:
        lr = config["lr"] * 0.1
    if epoch >= 0.75 * epochs:
        lr = config["lr"] * 0.01
    if epoch >= epochs:
        lr = config["lr"] * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'epoch : {epoch} -> lr : {lr}')


def get_pag_coeff(epoch):
    """returns the current pag_coeff"""
    if "step_pag_coeff" in config.keys() and config["step_pag_coeff"] and epoch <= 50:
        return config["pag_coeff"] * 2. / 3.
    return config["pag_coeff"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do PAG Imply Robustness?')
    parser.add_argument('--config_path', type=str, default='./configs/cifar10_CM_rn18.yaml', help='training config path')
    args, _ = parser.parse_known_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["chekpoint_folder"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    if device == "cpu":
        torch.manual_seed(config["seed"])
    else:
        torch.cuda.manual_seed_all(config["seed"])


    model_name = f'{config["arch"]}-cifar10-grad_source-{config["grad_source"]}-pag_coeff-{config["pag_coeff"]}' \
                f'-subsample-{config["num_grads_per_image"]}-seed-{config["seed"]}'

    if config["use_wandb"]:
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
        wandb.run.name = model_name

    data_stats = ( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(*data_stats)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"],
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

    # get model
    if config["arch"] == 'vit':
        model = ViT() #torch.nn.DataParallel(ViT())
    else:
        model = ResNet18() #torch.nn.DataParallel(ResNet18())

    if device == "cuda":
        model = model.to(torch.device(device))

    # print architecture details
    print(f'Using {config["arch"]} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} learnable params')

    # get optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(config["batch_size"])))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # exit()

    for epoch in range(1, config["epochs"] + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # get pag_coeff
        pag_coeff = get_pag_coeff(epoch)
        # training for one epoch
        print(f"Training epoch {epoch}")
        train_func(model, trainloader, optimizer) # dataloader to trainloader
        # evaluate robustness on L2, epsilon 0.5 (not AutoAttack)
        c_acc, r_acc = eval_adv_test_whitebox(
            model=model, 
            device=device,
            test_loader=testloader, 
            epsilon=0.5, num_steps=7,
            step_size=1.5 * 0.5 / 7, random=True,
            verbose=False, norm='l_2',
            stats=data_stats
        )
        if config["use_wandb"]:
            wandb.log({"c_acc": c_acc, "r_acc": r_acc})
        # save model
        if epoch % config["checkpoint_freq"] == 0 or epoch == config["epochs"]:
            torch.save(model.state_dict(), f'{config["chekpoint_folder"]}/{model_name}-epoch-{epoch}.pt')

