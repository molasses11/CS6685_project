import time

import torch
import torch.optim
import torch.utils.data
import torchvision

from dwt_transform import DWT2Numpy
import network


if __name__ == "__main__":
    total_time = -time.perf_counter()
    transform = torchvision.transforms.Compose(
        [
            DWT2Numpy("haar"),
            # [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    net = network.Net(
        conv_layers=[(24, 5), (48, 3)], linear_layers=[256, 128], output_size=10
    )
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    network.run(trainloader, testloader, net, loss, optimizer, device)
    total_time += time.perf_counter()
    print(f"Time to run: {total_time}")
