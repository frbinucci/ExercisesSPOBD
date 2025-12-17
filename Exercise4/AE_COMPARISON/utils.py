import math
import os

import numpy as np
import torch
import torchvision.datasets
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class RoadSignalDataset(dataset.Dataset):
    def __init__(self,**kwargs):
        super(RoadSignalDataset, self).__init__()
        self.size = kwargs.get('size',32)
        self.data_root = kwargs.get('data_root','./dataset/')
        self.data = list()
        self.pre_compressed = kwargs.get('pre_compressed',False)
        self.data_out = kwargs.get('data_out','./testDS')
        self.downsampling = kwargs.get('downsampling',False)
        self.ds_factor = kwargs.get('ds_factor',1)
        self.disable_data_loading = kwargs.get('disable_data_loading',False)
        self.augmentation_factor = kwargs.get('augmentation_factor',0)
        torch.manual_seed(0)

        self.random_rotation = transforms.RandomRotation((-45,45))


        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        if self.disable_data_loading == False:
          for current_directory in os.listdir(self.data_root):
              for current_photo in os.listdir(os.path.join(self.data_root,current_directory)):
                  image = Image.open(os.path.join(self.data_root,current_directory,current_photo))
                  image = image.resize((self.size, self.size), Image.BICUBIC)
                  tensor = self.transform(image)
                  self.data.append([(tensor),str(current_directory)])

                  for aug in range(0, self.augmentation_factor):
                    self.data.append([self.random_rotation(tensor), str(current_directory)])

    #This method must be overrided. It returns information about features and labels.
    def __getitem__(self, item):

        return self.data[item][0],int(self.data[item][1])

    #This method must be override. It returns information about length of data vector
    def __len__(self):
        return len(self.data)

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

class CompressedDataset(Dataset):
    def __init__(self,latent,labels):
        self.latent = latent
        self.labels = labels

    def __len__(self):
        return len(self.latent)

    def __getitem__(self, item):
        t = self.latent[item].astype(np.float32),self.labels[item].astype(np.longlong)
        return t

def make_gaussian_images(data,N,size,**kwargs):
    n_ch = 3
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
            for j in range(0,n_ch):
                new_dataset[:,j,i] = (new_dataset[:,j,i]-np.mean(new_dataset[:,j,i]))
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
            for j in range(0,n_ch):
                new_dataset[:,j,i] = (new_dataset[:,j,i]-np.mean(new_dataset[:,j,i]))


    transformed_dataset = np.reshape(new_dataset, (N, size,size,n_ch))

    return transformed_dataset,labels


def load_dataset(N_train,N_test,size,**kwargs):
    mode=kwargs.get('mode','FFT')
    compressed_dir = kwargs.get('compressed_dir',None)
    components_limiter = kwargs.get('components_limiter',None)
    training_set_tmp = None
    test_set_tmp = None
    if compressed_dir!=None:
        training_set = np.load(f'{compressed_dir}/compressed_train_{components_limiter}.npy')
        test_set = np.load(f'{compressed_dir}/compressed_test_{components_limiter}.npy')
        training_labels = np.load(f'{compressed_dir}/training_labels.npy')
        test_labels = np.load(f'{compressed_dir}/test_labels.npy')
    else:
        dataset = RoadSignalDataset(data_root='./dataset/', size=size,augmentation_factor=9)
        lengths = [math.floor(len(dataset) * 0.8), math.ceil(len(dataset) * 0.2)]
        training_set_tmp, test_set_tmp = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
        N_train = lengths[0]
        N_test = lengths[1]

        training_set,training_labels = make_gaussian_images(training_set_tmp, N_train, size,mode=mode)
        test_set,test_labels = make_gaussian_images(test_set_tmp, N_test, size,mode=mode)

        print(f'Len train = {len(training_set)}')
    return {'training_set': training_set,
            'training_labels': training_labels,
            'test_set': test_set,
            'test_labels': test_labels}
            # 'images_training_set': training_set,
            # 'images_test_set': test_set}


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