from os.path import dirname
import os

dirpath = dirname(__file__) #/project/common

def get_data_directory():
    dir_path_parent = dirname(dirpath)  #/project
    data_file_name = os.path.join(dir_path_parent, "data")
    return data_file_name

def get_data_directory_mnist():
    dir_path_parent = get_data_directory()
    data_file_name = os.path.join(dir_path_parent, "MNIST")
    return data_file_name

