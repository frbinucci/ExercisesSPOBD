# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path

import numpy as np

from training_manager import train, evaluate_network, fit_predict_gaussian_ib
import csv

def main():


    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",type=str,default="gib_retrain_test")
    parser.add_argument("--transformation_mode",type=str,default="FFT")

    parser.add_argument("--batch_size_train", type=int, default=16)
    parser.add_argument("--batch_size_test",type=int, default=32)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--n_epochs",type=int,default=30)

    parser.add_argument("--training_set_size",type=int,default=int(9704))
    parser.add_argument("--test_set_size",type=int,default=int(2426))
    parser.add_argument("--size_x",type=int,default=32)
    parser.add_argument("--size_y",type=float,default=16)
    parser.add_argument("--components_limiter",type=int,default=7)
    parser.add_argument("--ckpt_dir",type=str,default="./gib_network_final2")
    parser.add_argument("--path_to_load",type=str,default="./gib_network_final2/network.pth")
    parser.add_argument("--ae_ckpt_dir",type=str,default="./out_ae/cf32")
    parser.add_argument("--path_to_test",type=str,default=None)
    parser.add_argument("--results_output_dir",type=str,default="./final_results")
    parser.add_argument("--results_retrained_output_dir",type=str,default="./final_results")
    parser.add_argument("--compressed_size",type=int,default=30)
    parser.add_argument("--results_name",type=str,default="gib_retrained.npy")
    parser.add_argument("--compressed_output_dir",type=str,default="./compressed_gib_final2")

    parser.add_argument("--ckpt_retrain",type=str,default="gib_retrain/")





    args, unknown = parser.parse_known_args()

    mode = args.mode
    transformation_mode = args.transformation_mode
    compressed_output_dir = args.compressed_output_dir
    print(f"You selected the following operational mode = {mode}")
    res_output_dir = args.results_output_dir

    if mode == "train":
        train(cpkt_dir=args.ckpt_dir,
              batch_size_train=args.batch_size_train,
              batch_size_test=args.batch_size_test,
              lr=args.lr,
              n_epochs=args.n_epochs,
              transformation_mode=transformation_mode,
              base_size=args.size_x,
              N_train=args.training_set_size,
              training_mode='gib',
              N_test=args.test_set_size)
    elif mode == "retrain":
        compressed_size_mappings = np.array([5,10,15,20,25,30,35])
        for cs in compressed_size_mappings:
            ckpt_dir = f"{args.ckpt_retrain}/compression_{cs}"
            print(ckpt_dir)
            train(cpkt_dir=ckpt_dir,
                    training_mode='gib_retrain',
                    batch_size_train=args.batch_size_train,
                    batch_size_test=args.batch_size_test,
                    lr=args.lr,
                    n_epochs=args.n_epochs,
                    transformation_mode = transformation_mode,
                    base_size=args.size_x,
                    compressed_size=cs,
                    compressed_dir='./compressed_gib_final2',
                    N_train=args.training_set_size,
                    N_test=args.test_set_size)

    elif mode == "gib_test":
        components_array = np.array([5,10,15,20,25,30,35])
        accuracy_array = np.zeros(len(components_array))
        i = 0
        for component in components_array:
            print(f'Testing GIB for {components_array[i]} components...')
            accuracy_array[i] = fit_predict_gaussian_ib(path_to_load=args.path_to_load,
                                    training_set_size=args.training_set_size,
                                    test_set_size=args.test_set_size,
                                    size_x=args.size_x,
                                    size_y=args.size_y,
                                    components_limiter=component,
                                    saving_path=compressed_output_dir
                                    )
            print(f'Accuracy={accuracy_array[i]}')
            i = i+1
        np.save(f"{res_output_dir}/gib_res.npy",accuracy_array)
    elif mode == "gib_retrain_test":
        path_to_load = args.ckpt_retrain
        compressed_size_mappings = np.array([5,10,15,20,25,30,35])
        accuracy_array = np.zeros(len(compressed_size_mappings))
        results_output_dir = args.results_retrained_output_dir
        results_name = args.results_name
        i = 0
        for compressed_size in compressed_size_mappings:
            print(results_output_dir)
            accuracy = evaluate_network(compressed_size,None,f"{path_to_load}/compression_{compressed_size}",
                                        compressed_dir='./compressed_gib_final2',
                                        N_train=args.training_set_size,
                                        N_test=args.test_set_size,
                                        testing_mode='gib')
            accuracy_array[i] = accuracy
            print(accuracy)
            i = i+1
        if not os.path.exists(results_output_dir):
            os.makedirs(results_output_dir)
        np.save(f'{results_output_dir}/{results_name}',accuracy_array)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
