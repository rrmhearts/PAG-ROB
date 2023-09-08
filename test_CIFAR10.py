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
import matplotlib.pyplot as plt
import numpy as np

import gc
# import torch
# import torchvision
# import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
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
    # torch.cuda.manual_seed_all(config["seed"])
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

    batch_size = 1
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # get model
    # if config["arch"] == 'vit':
    #     model = torch.nn.DataParallel(ViT()).cuda()
    # else:
    model = torch.nn.DataParallel(ResNet18()).cuda()
    # print architecture details
    # print(f'Using {config["arch"]} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} learnable params')

    # get optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
    #                             weight_decay=config["weight_decay"])

    model.load_state_dict(torch.load("models/weights/rn18-c10-SBG.pt"))
    # model.load_state_dict(torch.load("models/weights/rn18-c10-CM.pt"))
    # model.load_state_dict(torch.load("models/weights/rn18-c10-NN.pt"))
    # model.load_state_dict(torch.load("models/weights/rn18-c10-OI.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # c_acc, r_acc = eval_adv_test_whitebox(model=model, device=device, test_loader=testloader, epsilon=0.5, num_steps=7,
    #                                         step_size=1.5 * 0.5 / 7, random=True, verbose=False, norm='l_2',
    #                                         stats=data_stats)
    # if config["use_wandb"]:
    #     wandb.log({"L2: c_acc": c_acc, "r_acc": r_acc})
    # print(f"L2: c_acc: {c_acc} r_acc: {r_acc}")

    # c_acc, r_acc = eval_adv_test_whitebox(model=model, device=device, test_loader=testloader, epsilon=0.5, num_steps=7,
    #                                         step_size=1.5 * 0.5 / 7, random=True, verbose=False, norm='l_inf',
    #                                         stats=data_stats)
    # if config["use_wandb"]:
    #     wandb.log({"Linf: c_acc": c_acc, "r_acc": r_acc})
    # print(f"Linf: c_acc: {c_acc} r_acc: {r_acc}")


    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img, transpose = True):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = next(dataiter) # 2nd image
    images, labels = next(dataiter) # 3rd image
    images, labels = next(dataiter) # 4rd image

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


    outputs = model(images.cuda())

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(batch_size)))
    
    # Choose an image for captum
    ind = 0
    input = images[ind].unsqueeze(0).cuda()
    input.requires_grad = True
    model.eval()

    # testloader = None
    # testset = None
    # gc.collect() # Python thing
    # torch.cuda.empty_cache() # PyTorch thing


    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                target=labels[ind],
                                                **kwargs
                                                )
        
        return tensor_attributions
    
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=labels[ind].item())
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))


    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                        nt_samples=100, stdevs=0.2)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    #dl = DeepLift(model)
    #attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    #attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    print('Original Image')
    print('Predicted:', classes[predicted[ind]], 
        ' Probability:', torch.max(F.softmax(outputs, 1)).item())

    original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

    _ = viz.visualize_image_attr(None, original_image, 
                        method="original_image", title="Original Image")
    plt.savefig("original.png")

    _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                             show_colorbar=True, title="Overlayed Gradient Magnitudes")
    plt.savefig("Gradient_magnitude.png")

    _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                             show_colorbar=True, title="Overlayed Integrated Gradients")
    plt.savefig("IG.png")

    _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
                                outlier_perc=10, show_colorbar=True, 
                                title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
    plt.savefig("IG_with_smoothgrad.png")

    # _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
    #                         title="Overlayed DeepLift")
    # plt.savefig("DeepLift.png")
