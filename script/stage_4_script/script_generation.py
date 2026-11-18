from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Method_RNN_Classifier import Method_RNN_Classifier
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Val_Test import Setting_Train_Val_Test
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib
matplotlib.use('Agg')            # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import csv
import time