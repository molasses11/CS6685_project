import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
	acc_list, loss_list = [], []
	with open(filename) as file_obj:
		lines = file_obj.readlines()
		for line in lines:
			if 'Accuracy' in line and 'loss' in line:
				word = line.split()
				accuracy = float(word[1][:-2])
				loss = float(word[-1])
				acc_list.append(accuracy)
				loss_list.append(loss_list)

	return acc_list, loss_list



mnist_original = './result/KMNIST_original.txt'
mnist_o_acc, mnist_o_loss = read_file(mnist_original)


mnist_haar = './result/KMNIST_haar.txt'
mnist_h_acc, mnist_h_loss = read_file(mnist_haar)

plt.plot(mnist_o_acc, label='Original')
plt.plot(mnist_h_acc, label='Haar')
plt.title("KMNIST, CNN")
plt.ylabel("Accuracy (%)")
plt.xlabel('Test Round')
plt.legend()
plt.show()