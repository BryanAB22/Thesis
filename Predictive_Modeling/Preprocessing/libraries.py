
import pandas as pd
import os, glob, shutil, zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import optuna


main_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset"
avg_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_avg"
linear_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_linear"
subset_dir  = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/subset"
rest_dir  = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/rest"
mice_dir= "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/mice" 
main_rest_impute_dir  = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/2_5_rest_impute"
imputed_data_dir  = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/impute_data"
BRITS_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_BRITS"
normalize_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_BRITS_Normalize"
kf_data_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_BRITS_Normalize_KF"



feat = ["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp","MAP"]



