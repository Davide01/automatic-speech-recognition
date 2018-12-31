# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:19:02 2018

@author: Kamil
"""

debug = True
"""
#########################################
#               Step 0
# Utility functions and library importing 
#########################################
"""

#import resource
#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

import pip
#from pip._internal import main

def install_library(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])
        
#install_library('librosa')
#install_library('jiwer')




import os
import numpy as np
#os.system("sudo pip install librosa")
import librosa
#import librosa.display
#import matplotlib.pyplot as plt


#import torch
#from warpctc_pytorch import CTCLoss
#from ctcdecode import CTCBeamDecoder
#import itertools
#import jiwer
#import random
#import scipy



"""
Data load and pre-processing

"""

def load_samples(file_path):
    
    """
    ys - input array of all samples [num_samples] x [sample_length]
    sr - sampling rate
    """
    
    ys, srs = [],[]
    
    #loads .wav files
    for filename in os.listdir(file_path):
        if filename.endswith(".wav"):
            y, sr = librosa.load(file_path+filename, sr=16000)
            ys.append(y)
            srs.append(sr)
            
    return (ys, srs)


def load_labels(file_path):
    labels = []
    
    for filename in os.listdir(file_path):
        if filename.endswith(".txt"):
            file = open(file_path+filename, "r") 
            labels.append(file.read())
            
    return labels



"""
#######################################
#               Step 1
# Load the samples and labels
#######################################
"""
print("STEP 1 START:")

training_dataset_path = "./an4_dataset/train/"

ys, srs = load_samples(training_dataset_path)
labels = load_labels(training_dataset_path)

print('Train sample shape', ys[0].shape[0])
print('Train samples count', len(ys))
print('Train labels count', len(labels))

validation_dataset_path = "./an4_dataset/validation/"

ys_valid, srs_valid = load_samples(validation_dataset_path)
labels_valid = load_labels(validation_dataset_path)

print('Validation sample shape', ys_valid[0].shape[0])
print('Validation samples count', len(ys_valid))
print('Validation labels count', len(labels_valid))


print('STEP 1 FINISHED')

"""
#######################################
#               Step 2
# Pad samples to  make them to have equal length
#######################################
"""
print("STEP 2 START:")

def find_max(ys):
    """
    Searches for the longest signal.
    used for padding.
    ys - [num_samples] x [sample_length]
    
    returns
    max(sample_length)
    """
    
    maximum = 0
    for current_sample in ys:
        dim = current_sample.shape[0]
        if dim > maximum:
            maximum = dim
    return maximum



def pad_samples(ys, ys_valid):
    """
    pads the signal
    
    Signals are not of the same length, so samples that are shorter then longest signal are padded
    ys - [num_samples] x [sample_length]
    
    returns
    [num_samples] x [sample_length] - sample_length is now same for each sample
    """
    
    training_max_length= find_max(ys)
    validation_max_length = find_max(ys_valid)
    
    max_length = training_max_length if training_max_length>validation_max_length else validation_max_length

    if debug:
        print('pad_samples:','Max sample length: ', max_length )
        
    new_ys = ys
    for i, current_sample in enumerate(new_ys, start= 0):
        if current_sample.shape[0] < max_length:
            z = np.zeros((max_length - current_sample.shape[0]))
            new_ys[i] = np.append(current_sample, z)
    new_ys_valid = ys_valid
    for i, current_sample in enumerate(new_ys_valid, start= 0):
        if current_sample.shape[0] < max_length:
            z = np.zeros((max_length - current_sample.shape[0]))
            new_ys_valid[i] = np.append(current_sample, z)
            
    return (new_ys, new_ys_valid)


padded_training_ys, padded_validation_ys= pad_samples(ys, ys_valid)

max_signal_length = padded_training_ys[0].shape[0]

if debug:
    print('main:','Training dataset length: ', len(padded_training_ys) )
    print('main:','Validation dataset length: ', len(padded_validation_ys) )
    #for i in padded_validation_ys:
    #    print(i.shape[0])
    print('main', 'Maximum signal length', max_signal_length)
print('STEP 2 FINISHED')

"""
#######################################
#               Step 3
# Pad samples to  make them to have equal length
#######################################
"""
print("STEP 3 Cutting into frames:")

def cut_audio_into_frames(samples, max_len,  sample_rate=16000, frame_size = 0.025, frame_stride = 0.01  ):
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples

    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    rawSamples = []
    
    for sam_no, current_sample in enumerate(samples, start=0):

        signal_length = len(current_sample)
        #print('signal_length ',signal_length)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        #print('num_frames ',num_frames)
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        padded_signal_frames= np.append(current_sample, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
            
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = padded_signal_frames[indices.astype(np.int32, copy=False)]
        
        rawSamples.append(frames)
        #if sam_no > 900:
            #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
         #   print('break at: ',sam_no)
            #break
        
    return rawSamples

validation_frames = cut_audio_into_frames(padded_validation_ys, max_len = max_signal_length, sample_rate= 16000)
print("validation already")
training_frames = cut_audio_into_frames(padded_training_ys, max_len = max_signal_length, sample_rate= 16000)
print("training already")


print(len(training_frames))
print(len(validation_frames))
print("STEP 3 FINISHED")


"""
#######################################
#               Step 4
# Pickle pre-processed data
#######################################
"""
print("STEP 3 Pickle pre-processed data:")

import pickle

training_frames = "raw_wave_training_frames"
validation_frames = "raw_wave_validation_frames"

# open the file for writing
training_fileObject = open(training_frames,'wb') 
validation_fileObject = open(validation_frames,'wb') 

pickle.dump(training_frames,training_fileObject)
pickle.dump(training_frames,validation_fileObject)          

print("STEP 4 FINISHED")