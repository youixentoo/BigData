# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:52:16 2023

@author: Thijs Weenink

File logic for the model
"""
try:
    from memory_profiler import profile
except ImportError:
    pass # Message in network.py
    
from h5py import File as h5File
from numpy import ravel, array
import pathlib
import re
import sys
import tensorflow as tf

# As to not load the entire dataset into memory when the dataset is created
# The arguments get passed as bytes and need decoding
def _load_as_generator(dataset_loc, set_type):
    images = h5File(f"{dataset_loc.decode()}/camelyonpatch_level_2_split_{set_type.decode()}_x.h5", 'r')["x"]
    labels = h5File(f"{dataset_loc.decode()}/camelyonpatch_level_2_split_{set_type.decode()}_y.h5", 'r')["y"]
    
    # Loads the dataset in chunks of 32
    for chunk_id in images.iter_chunks():
        img_chunk = images[chunk_id]
        lb_chunk = labels[chunk_id]
        yield (img_chunk, lb_chunk)
        

# Same idea as above, but for loading a single dataset file       
def _load_single_file_tfDataset(dataset_loc, set_type, x_or_y):
    file_data = h5File(f"{dataset_loc.decode()}/camelyonpatch_level_2_split_{set_type.decode()}_{x_or_y.decode()}.h5", 'r')[x_or_y.decode()]
    
    # Loads the dataset in chunks of 32
    for chunk_id in file_data.iter_chunks():
        yield file_data[chunk_id]
        

# Dataset configuration
def _dataset_config(ds):
    # ds = ds.cache()
    ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Could fine tune it myself, but why
    return ds


# Cannot wrap the get_dataset() function with try-catch as it will give an additional error due to how the dataset is loaded
# The error is one thrown by the model itself and don't want to surround the model with try-catch
# Instead a check if the file exists on disk and is not empty
# Terminate the script using 'sys.exit()' as you are unable to continue anyway
# Prints the files the script tried
def check_if_exists(dataset_loc, dataset_type):
    file_path_x = pathlib.Path(f"{dataset_loc}/camelyonpatch_level_2_split_{dataset_type}_x.h5")
    file_path_y = pathlib.Path(f"{dataset_loc}/camelyonpatch_level_2_split_{dataset_type}_y.h5")
    # Order matters as .stat().st_size gives an error if the file doesn't exist
    if not (file_path_x.exists() and file_path_x.stat().st_size > 0) or not (file_path_y.exists() and file_path_y.stat().st_size > 0):
        print(f"""
Could not find one or both of the files needed for dataset creation\n
Inputs:\n
{dataset_loc}/camelyonpatch_level_2_split_{dataset_type}_x.h5
{dataset_loc}/camelyonpatch_level_2_split_{dataset_type}_y.h5
              """)
        sys.exit()
        

# Loads the data, uses a generator to prevent the entire dataset to be loaded into memory
def get_dataset(dataset_loc, dataset_type, batch_size, img_height, img_width): 
    dataset = tf.data.Dataset.from_generator(
      _load_as_generator,
      (tf.uint8, tf.uint8),
      (tf.TensorShape((batch_size, img_height, img_width, 3)), tf.TensorShape((batch_size, 1, 1, 1))),
      args=(dataset_loc, dataset_type))
    return _dataset_config(dataset)
     
        
# Get the labels. Uses np.ravel to go from [[[[0]]] [[[1]]] [[[0]]]] to [[0] [1] [0]]
def get_labels_h5_file(dataset_loc, set_type):
    try:
        return array(list(map(ravel, h5File(f"{dataset_loc}/camelyonpatch_level_2_split_{set_type}_y.h5", 'r')["y"])))
    except FileNotFoundError as exc:
        print(f"Unable to find file for test labels\nInput: {parse_FNFE(exc)}")
        sys.exit()
        
        
# Parse the expection string to only return the file path
# Returns what's between 'name = ' and the first comma after.
# Should not ever go here as the script could not have passed check_if_exists() if this gets called
def parse_FNFE(exc_str):
    return re.search("(?<=name = ).*?(?=,)", str(exc_str)).group()
    

# @profile
# Ran into memory issues running this, this loads the entire dataset into memory, the gpu doesnt have enough
def load_h5_files_to_tfDataset(dataset_loc, set_type):
    images_file = h5File(f"{dataset_loc}/camelyonpatch_level_2_split_{set_type}_x.h5", 'r')
    labels_file = h5File(f"{dataset_loc}/camelyonpatch_level_2_split_{set_type}_y.h5", 'r')
    return tf.data.Dataset.from_tensor_slices((images_file['x'], labels_file['y']))
