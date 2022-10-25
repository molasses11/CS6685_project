import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pywt
import numpy as np

class DWT2(object):

    def __init__(self, wavelet):
        self.wavelet = wavelet

    def __call__(self, sample):
        coeffs2 = pywt.dwt2(sample, self.wavelet)
        cA, (cH, cV, cD) = coeffs2
        wave_param = np.vstack((np.hstack((cA, cH)), np.hstack((cV, cD))))
        return np.float32(wave_param)
        # return sample

training_data = datasets.KMNIST(
    root = "data",
    train=True,
    download=True,
    transform=transforms.Compose([
        DWT2('haar'),
        ToTensor(),
        transforms.Normalize((0.5,), (0.5))
        ]))


test_data = datasets.KMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([
        DWT2('haar'),
        ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ]))


batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from torch import nn, optim
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        #28
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
        #28
        #14
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        #14
        #7
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 0)
        #5
        #3 pool2
        self.conv4 = nn.Conv2d(64, 96, kernel_size = 3, stride = 1, padding = 1)
        #3
        self.conv5 = nn.Conv2d(96, 96, kernel_size = 2, stride = 1, padding = 0)
        #2
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.fc1 = nn.Linear(384, 270)
        self.fc2 = nn.Linear(270, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 49)
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.shape[0], -1)
        #x = self.dropout(x)
        x = self.dropout( F.relu(self.fc1(x)) )
        x = self.dropout( F.relu(self.fc2(x)) )
        x = self.dropout( F.relu(self.fc3(x)) )
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x

model = Network()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.NLLLoss()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            test(test_dataloader, model, loss_fn)

        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(dataloader.dataset),
        #         100. * batch_idx / len(dataloader), loss.item()))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = np.zeros((10,10)) # initialize confusion matrix
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_fn(output, target).item()
            # determine index with maximal log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(dataloader.dataset),
    #     100. * correct / len(dataloader.dataset)))

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")