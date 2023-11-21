import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

model_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/classification/network/faceforensics++_models_subset/full/xception/full_c23.p'
data_path = '/exports/eddie/scratch/s2017377/code/ff/FaceForensics/dataset/manipulated_sequences/DeepFakeDetection/c23/videos'

test = ...
train = ...

model = torch.load(model_path)

base_instance = ...
target_instance = ...

CREATE POISON INSTANCE

RETRAIN NETWORK


def create_poison():
    feature_space = 