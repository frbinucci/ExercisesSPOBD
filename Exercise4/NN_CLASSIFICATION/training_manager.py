import numpy as np
import scipy
import torch
from sklearn.decomposition import PCA

from net import Net
from solver import Solver
from utils import load_dataset, GaussianDataset, generate_loaders


def train(**kwargs):
    N_train = kwargs.get('N_train',int(6e4))
    N_test = kwargs.get('N_test',int(1e4))
    size = kwargs.get('base_size',28)
    random_seed = kwargs.get('random_seed',1)
    batch_size_train = kwargs.get('batch_size_train', 32)
    batch_size_test = kwargs.get('batch_size_test', 32)
    n_epochs = kwargs.get('n_epochs',100)
    cpkt_dir = kwargs.get('cpkt_dir','./checkpoint')
    lr = kwargs.get('lr',1e-3)
    transformation_mode = kwargs.get('transformation_mode','FFT')

    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    dataset = load_dataset(N_train,N_test,size,mode=transformation_mode)

    training_set = dataset['training_set']
    training_labels = dataset['training_labels']
    test_set = dataset['test_set']
    test_labels = dataset['test_labels']



    gaussian_training_set = GaussianDataset(training_set, training_labels)
    gaussian_test_set = GaussianDataset(test_set, test_labels)


    data_loaders = generate_loaders(training_set=gaussian_training_set,
                                        test_set=gaussian_test_set,
                                        shuffle_train=True,
                                        shuffle_test=False,
                                        batch_size_train=batch_size_train,
                                        batch_size_test=batch_size_test)

    net = Net().to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    s = Solver(net=net,
                criterion=criterion,
                optimizer=optimizer,
                data_loaders=data_loaders,
                num_epochs=n_epochs,
                output_dir=cpkt_dir)
    s.fit()

def ib_train(**kwargs):

    torch.manual_seed(46)
    path = kwargs.get('path_to_load',None)
    N_train = kwargs.get('N_train',None)
    N_test = kwargs.get('N_test', None)
    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    beta = kwargs.get('beta',None)
    components_limiter = kwargs.get('components_limiter',None)

    #Setting up the neural network model in order to train the Gaussian-IB transformation
    net = Net().to('cuda')
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

    matrix_params = list(net.parameters())[0]
    A = matrix_params.cpu().detach().numpy()

    y = s.intermediate_output(data_loaders['train'])

    np.random.seed(0)

    flattened_x = np.zeros((N_train,size_x**2))
    flattened_y = np.zeros((N_train,size_y**2))

    i = 0
    for x in training_set:
        flattened_x[i,:] = x.flatten()
        flattened_y[i,:] = y[i,:].flatten()
        i = i+1

    A,T,index,mean = compute_gaussian_ib(size_x**2,
                                         size_y**2,
                                         flattened_x,
                                         flattened_y,
                                         N_train,
                                         components_limiter=components_limiter,
                                         lambda1=0,
                                         lambda2=1)

    m = T.shape[0]  # Number of training examples.

    A = A[0:components_limiter,:]
    T = np.append(T, 1*np.ones((m, size_y**2)),axis=1)

    # The Normal Equation
    flattened_y = (flattened_y - mean)
    theta_ls = np.dot(np.linalg.pinv(np.dot(T.T, T)), np.dot(T.T, flattened_y))

    return A,theta_ls,index,T,mean


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

def ib_test(**kwargs):
    test_performances = kwargs.get('test_performances',False)
    path = kwargs.get('path_to_load',None)
    A = kwargs.get('A',None)
    theta_ls = kwargs.get('theta',None)
    index = kwargs.get('index',None)
    N_train = kwargs.get('N_train',None)
    N_test = kwargs.get('N_test',None)

    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    print(f'Size y = {size_y}')
    mean = kwargs.get('mean',None)

    net = Net().to('cuda')

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

    flattened_x_test = np.zeros((N_test,size_x**2))
    flattened_y_test = np.zeros((N_test,size_y**2))

    i = 0
    for x in test_set:
        flattened_x_test[i,:] = x.flatten()
        flattened_y_test[i,:] = y_test[i,:].flatten()
        i = i+1

    new_dataset = np.zeros((N_test,size_y,size_y))

    T_test = np.dot(flattened_x_test,A.T)
    T_test = T_test[:,0:index]

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
        accuracy = s.evaluate(loader)
    return accuracy

def fit_predict_gaussian_ib(**kwargs):

    path_to_load = kwargs.get('path_to_load',None)
    training_set_size = kwargs.get('training_set_size',None)
    test_set_size = kwargs.get('test_set_size',None)
    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    components_limiter = kwargs.get('components_limiter',None)

    A, theta, index, _, mean = ib_train(path_to_load=path_to_load,
                                             N_train=training_set_size,
                                             N_test=test_set_size,
                                             size_x=size_x,
                                             size_y=size_y,
                                             components_limiter=components_limiter)

    accuracy = ib_test(path_to_load=path_to_load,
            A=A,
            theta=theta,
            index=index,
            N_train=training_set_size,
            N_test=test_set_size,
            size_x=size_x,
            size_y=size_y,
            mean=mean,
            test_performances=True)
    return accuracy

def prepare_pca_dataset(fit,test,train_labels,test_labels,N_train,N_test,size,n_components,**kwargs):

    pca = PCA(n_components = n_components)
    print("*** COMPUTING PCA ***")
    flattened_dataset = np.zeros((N_train,size**2))
    i = 0
    for img in fit:
        flattened_dataset[i,:] = img.squeeze().flatten()
        i = i+1

    flattened_test_set = np.zeros((N_test,size**2))

    i = 0
    for img in test:
        flattened_test_set[i,:] = img.squeeze().flatten()
        i = i+1

    pca.fit(flattened_dataset)
    pca_dataset = pca.transform(flattened_dataset)

    pca_test_set = pca.transform(flattened_test_set)

    return pca_dataset,pca_test_set,train_labels,test_labels

def pca_test(**kwargs):

    mode = kwargs.get('mode','pca')
    N_train =kwargs.get('N_train',None)
    N_test = kwargs.get('N_test',None)
    size_x = kwargs.get('size_x',None)
    size_y = kwargs.get('size_y',None)
    path_to_load = kwargs.get('path_to_load',None)
    n_components = kwargs.get('n_components',None)
    snr = kwargs.get('snr',1e10)
    print(f'N components={n_components}')
    #dataset = load_dataset(N_train, N_test, size_x)

    fft_dataset = load_dataset(N_train,N_test,size_x,mode='ID')
    print(f'HAI SCELTO={mode}')
    pca_train,pca_test,_,_ = prepare_pca_dataset(fft_dataset['training_set'],
                                       fft_dataset['test_set'],
                                       fft_dataset['training_labels'],
                                       fft_dataset['test_labels'],
                                       N_train,
                                       N_test,
                                       size_x,
                                       n_components)


    net = Net().to('cuda')
    net.load_state_dict(torch.load(path_to_load))

    data_fft = generate_loaders(test_set=GaussianDataset(fft_dataset['test_set'],fft_dataset['test_labels']),
                                training_set=GaussianDataset(fft_dataset['training_set'],fft_dataset['training_labels']),
                                batch_size_train=1,
                                    batch_size_test=1,
                                    shuffle_test=False,
                                    shuffle_train=False)


    net.set_test_bottleneck(False)
    net.set_intermediate_output(True)

    s = Solver(net = net)

    out = s.intermediate_output(data_fft['train'])
    out_test = s.intermediate_output(data_fft['test'])

    flattened_y = np.zeros((N_train,size_y**2))
    flattened_y_test = np.zeros((N_test,size_y**2))

    i=0
    for y in out:
        flattened_y[i,:] = y.flatten()
        i+=1

    i=0
    for y in out_test:
        flattened_y_test[i,:] = y.flatten()
        i+=1


    m = pca_train.shape[0]  # Number of training examples.
    # Appending a cloumn of ones in X to add the bias term.
    pca_train_raw = np.append(pca_train, 1*np.ones((m, size_y**2)), axis=1)

    # The Normal Equation
    theta_ls = np.dot(np.linalg.pinv(np.dot(pca_train_raw.T, pca_train_raw)), np.dot(pca_train_raw.T, flattened_y))

    mt = pca_test.shape[0]
    pca_test_raw = np.append(pca_test, 1*np.ones((mt,size_y**2)), axis=1)
    y_hat = np.dot(pca_test_raw,theta_ls)

    y_hat_train = None

    new_dataset = np.zeros((N_test,size_y,size_y))

    i=0

    for y in y_hat:
        new_dataset[i,:] = np.reshape(y,(size_y,size_y))
        i+=1

    gaussian_dataset_y = GaussianDataset(new_dataset,fft_dataset['test_labels'])

    loader = torch.utils.data.DataLoader(gaussian_dataset_y,
                                              batch_size=1, shuffle=False)

    net.set_test_bottleneck(True)
    net.set_intermediate_output(False)

    s.set_network(net)

    accuracy = s.evaluate(loader)
    return accuracy