import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA

from training_manager import prepare_pca_dataset
from utils import load_dataset

def show_pca(N_train,N_test,size,n_components):

    N_train = int(N_train)
    N_test = int(N_test)
    size = int(size)

    fft_dataset = load_dataset(N_train,N_test,size,mode='ID')



    flattened_dataset = np.zeros((N_train,size**2))


    i = 0
    for img in fft_dataset['training_set']:
        flattened_dataset[i,:] = img.squeeze().flatten()
        i = i+1

    pca = PCA()
    pca.fit(flattened_dataset)
    plt.figure()
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("Number of Components",fontsize=14)
    plt.ylabel("Explained Variance",fontsize=14)
    plt.grid()
    plt.show()



    n_row = int(len(n_components)/4)
    n_col = 4

    _, axs = plt.subplots(n_row, n_col, figsize=(5, 5))
    axs = axs.flatten()


    i=0
    for ca in n_components:
        pca = PCA(n_components = ca)
        pca.fit(flattened_dataset)
        pca_dataset = pca.transform(flattened_dataset)

        pca_rec = pca.inverse_transform(pca_dataset)
        x = np.reshape(pca_rec[55],(28,28))
        axs[i].imshow(x,cmap='gray')
        axs[i].title.set_text(f" {ca} comp.")
        i+=1
    plt.show()



def plot_results(file_names):
    label_array = ["OIB","PCA"]
    i = 0
    for file in file_names:
        features_axis = np.load(f"./results/features_axis.npy")
        accuracy_axis = np.load(f"./results/{file}")
        plt.plot(features_axis,accuracy_axis,marker='v',label=label_array[i])
        i=i+1

    plt.xlabel("Number of Features",fontsize=14)
    plt.ylabel("Correct Classification Rate",fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def plot_histograms(training_set_size,test_set_size,size,n_hist):

    np.random.seed(46)
    original_dataset = load_dataset(int(training_set_size), int(test_set_size), size ,mode='ID')
    transformed_dataset = load_dataset(int(training_set_size), int(test_set_size), size ,mode='FFT')


    original_training_set = original_dataset['training_set']
    transformed_training_set = transformed_dataset['training_set']

    flattened_original = np.zeros((int(training_set_size),size**2))
    flattened_gaussian_set = np.zeros((int(training_set_size),size ** 2))

    pixel_indexes = np.random.randint(500,550,size=n_hist)

    for i in range(0,int(training_set_size)):
        flattened_original[i,:] = original_training_set[i,:].flatten()
    for i in range(0,int(training_set_size)):
        flattened_gaussian_set[i,:] = transformed_training_set[i,:].flatten()

    ncols = 2
    nrows =  int(n_hist/2)

    fig, axes_transformed = plt.subplots(nrows=nrows, ncols=ncols)


    index = 0
    for row in range(0,nrows):
        for col in range(0,ncols):
            px = pixel_indexes[index]
            ax = axes_transformed[row,col]
            ax.hist(flattened_gaussian_set[:, px], bins=50, density=True, alpha=0.6, color='b')
            x_min = np.min(flattened_gaussian_set[:, px])
            x_max = np.max(flattened_gaussian_set[:, px])
            mu = np.mean(flattened_gaussian_set[:, px])
            std = np.std(flattened_gaussian_set[:, px])
            x = np.linspace(x_min, x_max, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, color='r', linewidth=2)
            ax.grid()
            ax.set_xlabel("Pixel Intensities")
            ax.set_ylabel("Density")
            index+=1


    fig, axes_norm = plt.subplots(nrows=nrows, ncols=ncols)

    index = 0
    for row in range(0,nrows):
        for col in range(0,ncols):
            px = pixel_indexes[index]
            ax = axes_norm[row,col]
            ax.hist(flattened_original[:, px], bins=50, density=True, alpha=0.6, color='b')
            ax.grid()
            ax.set_xlabel("Pixel Z - scores values")
            ax.set_ylabel("Density")
            index+=1
    plt.show()



def main():

    results_output_dir = "./results/"

    print("What do you want to plot?")
    print("1)Comparison between PCA and GIB")
    print("2)Show PCA of some images")
    print("3)Feature Histograms before and after the transformation")
    choice = int(input("Select an option... "))

    if choice == 1:
        plot_results(["accuracy_bottleneck.npy","accuracy_pca.npy"])
    elif choice ==2:
        N_train = 6e4
        N_test = 1e4
        size = 28
        show_pca(N_train,N_test,size,[10,22,34,46,58,70,82,94])
    elif choice == 3:
        N_train = 6e4
        N_test = 1e4
        size = 28
        n_hist = int(input("How many histograms do you want to plot? "))
        plot_histograms(N_train,N_test,size,n_hist)

if __name__ == '__main__':
    main()