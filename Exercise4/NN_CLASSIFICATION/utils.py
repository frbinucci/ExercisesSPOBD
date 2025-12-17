import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class GaussianDataset(Dataset):
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        t = self.transform(self.images[item].astype(np.float32)),self.labels[item].astype(np.longlong)
        return t

def make_gaussian_images(data,N,size,**kwargs):
    n_ch = 1
    mode = kwargs.get('mode','FFT')
    new_dataset = np.zeros((N,n_ch,size**2))
    labels = np.zeros((N))
    i =0
    if mode == 'FFT':
        for image in data:
            img = image[0].numpy()
            j=0
            for ch in img:
                res = np.fft.fft2(ch)
                res = np.hstack((np.real(res.flatten())[0:int(size**2/2)],np.imag(res.flatten())[0:int(size**2/2)]))
                new_dataset[i,j, :] = res
                j = j+1
            labels[i] = image[1]
            i += 1
            if i==N:
                break
        for i in range(0,size**2):
            new_dataset[:,0,i] = (new_dataset[:,0,i]-np.mean(new_dataset[:,0,i]))
    elif mode=='PCA':
        pca = PCA(n_components=size**2)
        new_dataset = np.zeros((N, size**2))
        i = 0
        for img in data:
            new_dataset[i, :] = img[0].squeeze().flatten()
            labels[i] = img[1]
            i = i + 1
            if i == N:
                break
        new_dataset = pca.fit_transform(new_dataset)
    elif mode == 'ID':
        new_dataset = np.zeros((N,n_ch, size**2))
        i = 0
        for image in data:
            img = image[0].numpy()
            j=0
            for ch in img:
                new_dataset[i,j, :] = ch.flatten()
                j = j+1
            labels[i] = image[1]
            i = i + 1
            if i==N:
                break
        for i in range(0,size**2):
            new_dataset[:,0,i] = (new_dataset[:,0,i]-np.mean(new_dataset[:,0,i]))


    transformed_dataset = np.reshape(new_dataset, (N, size,size,n_ch))

    return transformed_dataset,labels


def load_dataset(N_train,N_test,size,**kwargs):
    mode=kwargs.get('mode','FFT')
    dataset_name = kwargs.get('dataset_name','MNIST')

    if dataset_name == 'MNIST':
        training_set = torchvision.datasets.MNIST('./files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize((size,size)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))]))
        test_set = torchvision.datasets.MNIST('./files/', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize((size,size)),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))]))

    gaussian_training_set,training_labels = make_gaussian_images(training_set, N_train, size,mode=mode)
    gaussian_test_set,test_labels = make_gaussian_images(test_set, N_test, size,mode=mode)

    return {'training_set': gaussian_training_set,
            'training_labels': training_labels,
            'test_set': gaussian_test_set,
            'test_labels': test_labels,
            'images_training_set': training_set,
            'images_test_set': test_set}

def generate_loaders(**kwargs):

    batch_size_train = kwargs.get('batch_size_train',32)
    batch_size_test = kwargs.get('batch_size_test',32)

    training_set = kwargs.get('training_set',None)
    test_set = kwargs.get('test_set',None)

    shuffle_train = kwargs.get('shuffle_train',True)
    shuffle_test = kwargs.get('shuffle_test',True)

    train_loader = None
    test_loader = None

    if training_set is not None:
        train_loader = torch.utils.data.DataLoader(training_set,
                                                   batch_size=batch_size_train, shuffle=shuffle_train)

    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size_test, shuffle=shuffle_test)

    data_loaders = {'train': train_loader, 'test': test_loader}

    return data_loaders