import os

import numpy as np
import scipy
import torch

from Solver import Solver
from autoencoders import CompressionNetwork
from networks import ClassificationNetwork, GIBNetwork, AENetwork
from utils import load_dataset, generate_loaders, GaussianDataset, CompressedDataset


def ib_train(**kwargs):

    torch.manual_seed(46)
    path = kwargs.get('path_to_load',None)
    N_train = kwargs.get('N_train',None)
    N_test = kwargs.get('N_test', None)
    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    beta = kwargs.get('beta',None)
    n_ch = kwargs.get('n_ch',3)
    components_limiter = kwargs.get('components_limiter',None)
    saving_path = kwargs.get('saving_path', None)

    #Setting up the neural network model in order to train the Gaussian-IB transformation
    net = GIBNetwork().to('cuda')
    net.load_state_dict(torch.load(path))
    net.set_intermediate_output(True)
    net.set_test_bottleneck(False)

    #Loading the dataset and Gaussianization
    dataset = load_dataset(N_train,N_test,size_x)
    training_set = dataset['training_set']
    training_labels = dataset['training_labels']
    test_set = dataset['test_set']
    test_labels = dataset['test_labels']
    gaussian_test_set = GaussianDataset(test_set,test_labels)
    gaussian_training_set = GaussianDataset(training_set,training_labels)


    #Generating data-loaders, then fitting the IB transformation
    data_loaders = generate_loaders(test_set=gaussian_test_set,
                                    training_set=gaussian_training_set,
                                    training_labels=training_labels,
                                    test_labels=test_labels,
                                    batch_size_test=1,
                                    batch_size_train=1,
                                    shuffle_test=False,
                                    shuffle_train=False)
    s = Solver(net = net)

    y = s.intermediate_output(data_loaders['train'])

    np.random.seed(0)

    flattened_x = np.zeros((N_train,n_ch*size_x**2))
    flattened_y = np.zeros((N_train,size_y**2))

    i = 0
    for x in training_set:
        flattened_channels = np.zeros((n_ch,size_x**2))
        for ch in range(0,n_ch):
            flattened_channels[ch,:] = x[:,:,ch].flatten()
        flattened_x[i,:] = np.hstack(flattened_channels)
        flattened_y[i,:] = y[i,:].flatten()
        i = i+1

    A,T,index,mean = compute_gaussian_ib(n_ch*size_x**2,
                                         size_y**2,
                                         flattened_x,
                                         flattened_y,
                                         N_train,
                                         components_limiter=components_limiter,
                                         lambda1=0,
                                         lambda2=1)

    m = T.shape[0]  # Number of training examples.

    A = A[0:components_limiter,:]

    if saving_path != None:
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        np.save(f'{saving_path}/compressed_train_{components_limiter}.npy',T)
        np.save(f'{saving_path}/training_labels.npy',training_labels)

    # The Normal Equation
    T = np.append(T, 1*np.ones((m, size_y**2)),axis=1)
    flattened_y = (flattened_y - mean)
    theta_ls = np.dot(np.linalg.pinv(np.dot(T.T, T)), np.dot(T.T, flattened_y))

    return A,theta_ls,index,T,mean

def ib_test(**kwargs):
    n_ch = kwargs.get('n_ch',3)
    test_performances = kwargs.get('test_performances',False)
    path = kwargs.get('path_to_load',None)
    A = kwargs.get('A',None)
    theta_ls = kwargs.get('theta',None)
    index = kwargs.get('index',None)
    N_train = kwargs.get('N_train',None)
    N_test = kwargs.get('N_test',None)
    saving_path = kwargs.get('saving_path', None)

    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    mean = kwargs.get('mean',None)

    net = GIBNetwork().to('cuda')

    net.load_state_dict(torch.load(path))

    net.set_intermediate_output(True)
    net.set_test_bottleneck(False)

    dataset = load_dataset(N_train, N_test, size_x,mode='FFT')

    test_set = dataset['test_set']
    test_labels =dataset['test_labels']

    gaussian_test_set = GaussianDataset(test_set,test_labels)

    data_loaders = generate_loaders(test_set =gaussian_test_set,  batch_size_train=1, batch_size_test=1,shuffle_test=False)
    s = Solver(net = net)

    y_test = s.intermediate_output(data_loaders['test'])

    flattened_x_test = np.zeros((N_test,n_ch*size_x**2))
    flattened_y_test = np.zeros((N_test,size_y**2))

    i = 0


    for x in test_set:
        flattened_channels = np.zeros((n_ch,size_x**2))
        for ch in range(0,n_ch):
            flattened_channels[ch,:] = x[:,:,ch].flatten()
        flattened_x_test[i,:] = np.hstack(flattened_channels)
        flattened_y_test[i,:] = y_test[i,:].flatten()
        i = i+1

    new_dataset = np.zeros((N_test,size_y,size_y))

    T_test = np.dot(flattened_x_test,A.T)
    T_test = T_test[:,0:index]

    components_limiter = index
    if saving_path != None:
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        np.save(f'{saving_path}/compressed_test_{components_limiter}.npy', T_test)
        np.save(f'{saving_path}/test_labels.npy', test_labels)

    mt = T_test.shape[0]
    T_test = np.append(T_test,1*np.ones((mt, size_y**2)), axis=1)

    y_hat = np.dot(T_test,theta_ls)

    y_hat = (y_hat+mean)

    i = 0
    for y in y_hat:
        new_dataset[i, :] = np.reshape(y,(size_y,size_y))
        i = i + 1
    if test_performances == True:
        i=0
        nmse=0
        for y in y_hat:
            gt = flattened_y_test[i,:]
            diff = y-flattened_y_test[i,:]
            nmse +=np.sum(np.dot(diff.T,diff))/np.dot(gt.T,gt)
            i=i+1
        nmse/=N_test

        gaussian_dataset_y = GaussianDataset(new_dataset,test_labels)


        loader = torch.utils.data.DataLoader(gaussian_dataset_y,
                                                  batch_size=1, shuffle=False)

        net.set_test_bottleneck(True)
        net.set_intermediate_output(False)
        accuracy = s.evaluate(loader=loader)

    return accuracy

def compute_gaussian_ib(nx, ny, X, y, N, **kwargs):
    lambda1 = kwargs.get('lambda1', 0)
    lambda2 = kwargs.get('lambda2', 0)
    mean = np.mean(y, axis=0)
    y = y - mean

    cov_matrix = np.cov(np.hstack((X, y)), rowvar=False)

    testcovix = cov_matrix[0:nx, 0:nx]
    testcoviy = cov_matrix[nx:nx + ny, nx:nx + ny]

    sigma_xx = testcovix + lambda1 * np.identity(nx)
    sigma_yy = testcoviy + lambda2 * np.identity(ny)
    sigma_xy = cov_matrix[0:nx, nx:nx + ny]
    sigma_yx = sigma_xy.T

    precision_x = np.linalg.pinv(sigma_xx)
    precision_y = np.linalg.pinv(sigma_yy)

    components_limiter = kwargs.get('components_limiter', N)

    eye_matrix = np.identity(nx)
    cca_matrix = eye_matrix - np.dot(np.dot(sigma_xy, np.dot(precision_y, sigma_yx)), precision_x)

    [eigenvalues, eigenvectors] = scipy.linalg.eig(cca_matrix, left=True, right=False)

    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = np.minimum(eigenvalues, 1)

    eigenvectors = np.real(eigenvectors)

    ind = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[ind]
    eigenvectors = eigenvectors[:, ind]

    A = np.zeros((nx, nx))

    j = 0

    beta_vector = 1 / (1 - eigenvalues)

    n_component = 0
    for i in range(0, len(beta_vector)):
        if n_component >= components_limiter:
            break
        if beta_vector[i] > 1:
            n_component += 1

    beta = beta_vector[components_limiter]
    for i in range(0, nx):
        if eigenvalues[i] < 1 and eigenvalues[i] > 0 and beta >= beta_vector[i]:
            rj = np.matmul(np.matmul(eigenvectors[:, i].T, sigma_xx), eigenvectors[:, i])
            alpha = ((beta) * (1 - eigenvalues[i]) - 1) / ((eigenvalues[i]) * rj)
            A[j, :] = np.sqrt(alpha) * eigenvectors[:, i]
            j = j + 1

    T = np.matmul(X, A.T)
    T = T[:, 0:components_limiter]
    j = components_limiter
    return A, T, j, mean

def fit_predict_gaussian_ib(**kwargs):

    path_to_load = kwargs.get('path_to_load',None)
    training_set_size = kwargs.get('training_set_size',None)
    test_set_size = kwargs.get('test_set_size',None)
    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    components_limiter = kwargs.get('components_limiter',None)
    compressed_output_dir = kwargs.get('saving_path',None)

    A, theta, index, _, mean = ib_train(path_to_load=path_to_load,
                                             N_train=training_set_size,
                                             N_test=test_set_size,
                                             size_x=size_x,
                                             size_y=size_y,
                                             components_limiter=components_limiter,
                                             saving_path=compressed_output_dir)

    accuracy = ib_test(path_to_load=path_to_load,
            A=A,
            theta=theta,
            index=index,
            N_train=training_set_size,
            N_test=test_set_size,
            size_x=size_x,
            size_y=size_y,
            mean=mean,
            test_performances=True,
            saving_path=compressed_output_dir)
    return accuracy

def train(**kwargs):
    training_mode = kwargs.get('training_mode','encoder')
    N_train = kwargs.get('N_train',int(6e4))
    N_test = kwargs.get('N_test',int(1e4))
    size = kwargs.get('base_size',28)
    random_seed = kwargs.get('random_seed',46)
    batch_size_train = kwargs.get('batch_size_train', 32)
    batch_size_test = kwargs.get('batch_size_test', 32)
    n_epochs = kwargs.get('n_epochs',100)
    cpkt_dir = kwargs.get('cpkt_dir','./checkpoint')
    lr = kwargs.get('lr',1e-3)
    transformation_mode = kwargs.get('transformation_mode','FFT')
    compressed_size = kwargs.get('compressed_size',None)
    hidden_filters = kwargs.get('hidden_filters',None)
    compressed_dir = kwargs.get('compressed_dir',None)

    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    dataset = load_dataset(N_train,N_test,size,mode=transformation_mode,compressed_dir=compressed_dir,components_limiter=compressed_size)

    training_set = dataset['training_set']
    training_labels = dataset['training_labels']
    test_set = dataset['test_set']
    test_labels = dataset['test_labels']

    if compressed_dir==None:
        gaussian_training_set = GaussianDataset(training_set, training_labels)
        gaussian_test_set = GaussianDataset(test_set, test_labels)
    else:
        gaussian_training_set = CompressedDataset(training_set, training_labels)
        gaussian_test_set = CompressedDataset(test_set, test_labels)


    data_loaders = generate_loaders(training_set=gaussian_training_set,
                                        test_set=gaussian_test_set,
                                        shuffle_train=True,
                                        shuffle_test=False,
                                        batch_size_train=batch_size_train,
                                        batch_size_test=batch_size_test)
    if training_mode=="encoder":
        ae = CompressionNetwork(hidden_filters, compressed_size).to('cuda')
        net = ClassificationNetwork(compressed_size,256,43).to('cuda')
    elif training_mode=="gib":
        ae = None
        net = GIBNetwork().to('cuda')
    elif training_mode=="gib_retrain":
        ae = None
        net = ClassificationNetwork(compressed_size,256,43).to('cuda')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    s = Solver(net=net,
               ae = ae,
                criterion=criterion,
                lr = lr,
                data_loaders=data_loaders,
                num_epochs=n_epochs,
                output_dir=cpkt_dir,)
    s.fit()

def evaluate_network(compressed_size,hidden_filters,loading_path,**kwargs):

    N_train = kwargs.get('N_train',int(6e4))
    N_test = kwargs.get('N_test',int(1e4))
    size = kwargs.get('size_x',32)
    compressed_dir = kwargs.get('compressed_dir',None)
    transformation_mode = kwargs.get('transformation_mode','ID')


    dataset = load_dataset(N_train,N_test,size,mode=transformation_mode,compressed_dir=compressed_dir,components_limiter=compressed_size)



    testing_mode = kwargs.get('testing_mode','gib')

    training_set = dataset['training_set']
    training_labels = dataset['training_labels']
    test_set = dataset['test_set']
    test_labels = dataset['test_labels']

    if compressed_dir==None:
        gaussian_training_set = GaussianDataset(training_set, training_labels)
        gaussian_test_set = GaussianDataset(test_set, test_labels)
    else:
        gaussian_training_set = CompressedDataset(training_set, training_labels)
        gaussian_test_set = CompressedDataset(test_set, test_labels)


    data_loaders = generate_loaders(training_set=gaussian_training_set,
                                        test_set=gaussian_test_set,
                                        shuffle_train=True,
                                        shuffle_test=False,
                                        batch_size_train=32,
                                        batch_size_test=32)

    if testing_mode=='ae':
        ae = CompressionNetwork(hidden_filters,compressed_size).to('cuda')
        ae.load_state_dict(torch.load(f"{loading_path}/ae.pth"))
    else:
        ae = None

    classifier = ClassificationNetwork(compressed_size,256,43).to('cuda')
    classifier.load_state_dict(torch.load(f"{loading_path}/network.pth"))

    s = Solver(net=classifier,ae=ae)
    acc = s.evaluate(loader=data_loaders['test'])

    return acc