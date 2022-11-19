from collections import OrderedDict
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

class Net(nn.Module):
    def __init__(self, conv_layers, linear_layers, output_size):
        super().__init__()

        self.conv = nn.Sequential(
            OrderedDict(
                (
                    f"conv{i}",
                    nn.Sequential(
                        nn.LazyConv2d(channels, kernel), nn.ReLU(), nn.MaxPool2d(2, 2)
                    ),
                )
                for i, (channels, kernel) in enumerate(conv_layers)
            )
        )
        self.fc = nn.Sequential(
            OrderedDict(
                (f"fc{i}", nn.Sequential(nn.LazyLinear(size), nn.ReLU()))
                for i, size in enumerate(linear_layers)
            )
        )
        self.out = nn.LazyLinear(output_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        # x = torch.sigmoid(self.out(x))
        x = self.out(x)
        return x


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float32), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float32), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def run(trainloader, testloader, net, loss_fn, optimizer, device, epochs=300):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, net, loss_fn, optimizer, device)
        test(testloader, net, loss_fn, device)
    print("Done!")
