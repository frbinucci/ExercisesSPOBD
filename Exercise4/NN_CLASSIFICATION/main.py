# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path

import numpy as np

from training_manager import train, fit_predict_gaussian_ib, pca_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",type=str,default="pca")
    parser.add_argument("--transformation_mode",type=str,default="FFT")

    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_test",type=int, default=32)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--n_epochs",type=int,default=10)

    parser.add_argument("--training_set_size",type=int,default=int(6E4))
    parser.add_argument("--test_set_size",type=int,default=int(1e4))
    parser.add_argument("--size_x",type=int,default=28)
    parser.add_argument("--size_y",type=float,default=28)
    parser.add_argument("--components_limiter",type=int,default=7)
    parser.add_argument("--ckpt_dir",type=str,default="./ib_ese/")
    parser.add_argument("--path_to_load",type=str,default="./pca_ese/network.pth")
    parser.add_argument("--ae_ckpt_dir",type=str,default="./out_ae/cf32")
    parser.add_argument("--path_to_test",type=str,default=None)
    parser.add_argument("--results_output_dir",type=str,default="./results")


    args, unknown = parser.parse_known_args()

    mode = args.mode
    transformation_mode = args.transformation_mode
    results_output_dir = args.results_output_dir
    print(f"You selected the following operational mode = {mode}")
    if mode == "train":
        train(cpkt_dir=args.ckpt_dir,
              batch_size_train=args.batch_size_train,
              batch_size_test=args.batch_size_test,
              lr=args.lr,
              n_epochs=args.n_epochs,
              transformation_mode = transformation_mode,
              base_size=args.size_x)
    if mode == "ib":
        components_array = np.array([8,10,12,14,16,18,20,22,24,26,28,30])
        accuracy_array = np.zeros(len(components_array))
        i = 0
        for component in components_array:
            print(f'Testing GIB for {components_array[i]} components...')
            accuracy_array[i] = fit_predict_gaussian_ib(path_to_load=args.path_to_load,
                                    training_set_size=args.training_set_size,
                                    test_set_size=args.test_set_size,
                                    size_x=args.size_x,
                                    size_y=args.size_y,
                                    components_limiter=component
                                    )
            i = i+1
        if not os.path.exists(results_output_dir):
            os.makedirs(results_output_dir)

        np.save(f"{results_output_dir}/features_axis.npy",components_array)
        np.save(f"{results_output_dir}/accuracy_bottleneck.npy",accuracy_array)

    if mode == "pca":
        components_array = np.array([8,10,12,14,16,18,20,22,24,26,28,30])
        accuracy_array = np.zeros(len(components_array))
        i = 0
        for component in components_array:
            print(f'Testing PCA for {components_array[i]} components...')
            accuracy_array[i] = pca_test(N_train=args.training_set_size,
                 N_test=args.test_set_size,
                 size_x=args.size_x,
                 size_y=args.size_y,
                 path_to_load=args.path_to_load,
                 n_components=component)
            i = i+1
        if not os.path.exists(results_output_dir):
            os.makedirs(results_output_dir)
        np.save(f"{results_output_dir}/features_axis.npy",components_array)
        np.save(f"{results_output_dir}/accuracy_pca.npy",accuracy_array)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
