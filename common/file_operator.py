from os.path import dirname
import os

dirpath = dirname(__file__) #/project/common

def get_data_file_path():
    dir_path_parent = dirname(dirpath)
    data_file_name = os.path.join(dir_path_parent, )
    pass