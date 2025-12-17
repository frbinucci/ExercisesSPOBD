# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path

import numpy as np

from training_manager import train, evaluate_network
import csv

def hidden_filter_mapping(mapping_path,compressed_size):
    with open(mapping_path, newline='') as csvfile:
        encoder_mapping = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in encoder_mapping:
            if int(row[0])==compressed_size:
                hidden_filters = int(row[1])
    return hidden_filters

def main():


    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",type=str,default="evaluate")
    parser.add_argument("--transformation_mode",type=str,default="ID")

    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_test",type=int, default=32)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--n_epochs",type=int,default=30)

    parser.add_argument("--training_set_size",type=int,default=int(9704))
    parser.add_argument("--test_set_size",type=int,default=int(2426))
    parser.add_argument("--size_x",type=int,default=32)
    parser.add_argument("--size_y",type=float,default=28)
    parser.add_argument("--components_limiter",type=int,default=7)
    parser.add_argument("--ckpt_dir",type=str,default="./ae_exercise_final/compression")
    parser.add_argument("--path_to_load",type=str,default="./ae_exercise_final/")
    parser.add_argument("--path_to_test",type=str,default=None)

    parser.add_argument("--results_output_dir",type=str,default="./final_results")
    parser.add_argument("--results_name",type=str,default="ae_res.npy")


    args, unknown = parser.parse_known_args()

    mode = args.mode
    transformation_mode = args.transformation_mode
    print(f"You selected the following operational mode = {mode}")

    if mode == "train":
        compressed_size_mappings = np.array([5,10,15,20,25,30,35])
        for cs in compressed_size_mappings:
            hidden_filters = hidden_filter_mapping('./encoder_mapping.txt', cs)
            ckpt_dir = f"{args.ckpt_dir}_{cs}"
            print(ckpt_dir)
            train(cpkt_dir=ckpt_dir,
                    batch_size_train=args.batch_size_train,
                    batch_size_test=args.batch_size_test,
                    lr=args.lr,
                    n_epochs=args.n_epochs,
                    transformation_mode = transformation_mode,
                    base_size=args.size_x,
                    compressed_size=cs,
                    hidden_filters=hidden_filters,
                    N_train=args.training_set_size,
                    N_test=args.test_set_size)
    elif mode == "evaluate":
        mapping_path = './encoder_mapping.txt'
        path_to_load = args.path_to_load
        compressed_size_mappings = np.array([5,10,15,20,25,30,35])
        accuracy_array = np.zeros(len(compressed_size_mappings))
        results_output_dir = args.results_output_dir
        results_name = args.results_name
        i = 0
        for compressed_size in compressed_size_mappings:
            hidden_filters = hidden_filter_mapping(mapping_path,compressed_size)
            print(hidden_filters)
            accuracy = evaluate_network(compressed_size,hidden_filters,f"{path_to_load}/compression_{compressed_size}",
                                        N_train=args.training_set_size,
                                        N_test=args.test_set_size,
                                        testing_mode='ae')
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
