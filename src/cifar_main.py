import torch
import torchvision
import torchvision.transforms as transforms
import pywt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DWT2(object):

	def __init__(self, wavelet):
		self.wavelet = wavelet

	def __call__(self, sample):

		sample_array = np.array(sample)
		sample_r = sample_array[:, :, 0]
		sample_g = sample_array[:, :, 1]
		sample_b = sample_array[:, :, 2]

		coeffs2_r = pywt.dwt2(sample_r, self.wavelet)
		cA_r, (cH_r, cV_r, cD_r) = coeffs2_r
		wave_param_r = np.vstack((np.hstack((cA_r, cH_r)), np.hstack((cV_r, cD_r))))

		coeffs2_g = pywt.dwt2(sample_g, self.wavelet)
		cA_g, (cH_g, cV_g, cD_g) = coeffs2_g
		wave_param_g = np.vstack((np.hstack((cA_g, cH_g)), np.hstack((cV_g, cD_g))))

		coeffs2_b = pywt.dwt2(sample_b, self.wavelet)
		cA_b, (cH_b, cV_b, cD_b) = coeffs2_b
		wave_param_b = np.vstack((np.hstack((cA_b, cH_b)), np.hstack((cV_b, cD_b))))

		wave_param = np.array([wave_param_r, wave_param_g, wave_param_b])
		wave_param = np.transpose(wave_param, (1, 2, 0))

		return np.float32(wave_param)
		# return sample

transform = transforms.Compose(
	[DWT2('haar'),
	# [
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
									    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
										  shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
									   download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
										 shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

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



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 300
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, net, loss_fn, optimizer)
    test(testloader, net, loss_fn)
print("Done!")