import matplotlib.pyplot as plt
import numpy as np


def plot_comparisons(feature_axis,dir_list,marker_array,label_array):

    print(label_array[0])
    i =0
    for dir in dir_list:
        data = np.load(dir)
        print(len(data))
        plt.plot(feature_axis,data,marker=marker_array[i],label=label_array[i])
        i=i+1
    plt.grid()
    plt.xlabel("Number of Features",fontsize=14)
    plt.ylabel("Correct Classification Rate",fontsize=14)
    plt.legend()
    plt.show()


if __name__=="__main__":
    dir_list = ["./final_results/gib_res.npy","./final_results/ae_res.npy","./final_results/gib_retrained.npy"]
    marker_array = ['v','o','H']
    label_array = ["OIB","AE COMPRESSION","OIB + RETRAIN"]
    feature_axis = np.array([5,10,15,20,25,30,35])
    plot_comparisons(feature_axis,dir_list,marker_array,label_array)