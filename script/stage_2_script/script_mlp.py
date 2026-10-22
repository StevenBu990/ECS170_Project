import matplotlib

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test

matplotlib.use('Agg')            # save plots to files instead of opening a window
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---- experiment settings: change these for each run ----


# open the config file given on the command line, e.g. python script_mlp.py ../../configs/run1.json
with open(sys.argv[1]) as f:
    cfg = json.load(f)


HIDDEN_DIMS = cfg['hidden_layers']
EPOCHS = cfg['epochs']
LEARNING_RATE = cfg['lr']
OPTIMIZER = cfg['optimizer']
LOSS_FN = cfg['loss']
RUN_NAME = f'{LOSS_FN}-loss_{OPTIMIZER}-opt_{LEARNING_RATE}-lr_{HIDDEN_DIMS}-hidden_{EPOCHS}-epoch'

results_dir = '../../result/stage_2_result/'

# save experiment settings to .txt file
os.makedirs(results_dir + f'{RUN_NAME}', exist_ok=True) # make dirs
with open(f"{results_dir}{RUN_NAME}/config.txt", "w") as f:
    f.write(f"name: {RUN_NAME}\nhidden dimensions: {HIDDEN_DIMS}\nepochs: {EPOCHS}\nlearning_rate: {LEARNING_RATE}\noptimizer: {OPTIMIZER}\nloss function: {LOSS_FN}")

# ---------------------------------------------------------

# ---- parameter section ----
np.random.seed(2)
torch.manual_seed(2)

# ---- object initialization section ----
train_obj = Dataset_Loader('MNIST train', '')
train_obj.dataset_source_folder_path = '../../data/stage_2_data/'
train_obj.dataset_source_file_name = 'train.csv'

test_obj = Dataset_Loader('MNIST test', '')
test_obj.dataset_source_folder_path = '../../data/stage_2_data/'
test_obj.dataset_source_file_name = 'test.csv'

method_obj = Method_MLP('multi-layer perceptron', '', hidden_dims=HIDDEN_DIMS, loss_fn=LOSS_FN)
method_obj.max_epoch = EPOCHS
method_obj.learning_rate = LEARNING_RATE
method_obj.optimizer_name = OPTIMIZER

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'

setting_obj = Setting_Train_Test('pre-split train/test', '')
setting_obj.test_dataset = test_obj

evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')

# ---- running section ----
print('************ Start ************')
setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
scores = setting_obj.load_run_save_evaluate()

print('************ Test Performance ************')
for name, value in scores.items():
    print(f'{name:20s} {value:.4f}')

# ---- convergence plot (2-3) ----
history = method_obj.history

plt.plot(history['epoch'], history['loss'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title(f'Training loss vs. Epoch ({RUN_NAME})')
plt.grid(True)
plt.tight_layout(pad=1)
plt.savefig(f'../../result/stage_2_result/{RUN_NAME}/convergence_{RUN_NAME}.png', dpi=150)

# ---- record this run (2-4) ----

# 1. save the loss history, for the comparison plot later
pd.DataFrame(history).to_csv(f'{results_dir}{RUN_NAME}/history_{RUN_NAME}.csv', index=False)

# 2. add a row to the experiments table
row = {'run': RUN_NAME, 'hidden_dims': '-'.join(str(h) for h in HIDDEN_DIMS), 'epochs': EPOCHS, 'lr': LEARNING_RATE, 'optimizer': OPTIMIZER, 'loss_fn': LOSS_FN,
       'train_acc': round(history['acc'][-1], 4)}
row.update({name: round(value, 4) for name, value in scores.items()})

table_path = results_dir + 'experiments.csv'
is_new = not os.path.exists(table_path)
with open(table_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if is_new:
        writer.writeheader()      # column names, only the first time
    writer.writerow(row)
print('recorded run', RUN_NAME, 'in experiments.csv')

print('************ Finish ************')