import torch
import torchvision

def test(model, img_dir, transform, batch_size=4):
    testset = torchvision.datasets.CIFAR10(root=img_dir, train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    