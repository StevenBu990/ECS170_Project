from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib
matplotlib.use('Agg')            # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv
import pandas as pd

# ---- experiment settings: change these for each run ----
RUN_NAME = 'arch_deeper'
HIDDEN_DIMS = (512, 256, 128, 64)
EPOCHS = 20
LEARNING_RATE = 1e-3
OPTIMIZER = 'adam'          # 'adam' or 'sgd'
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

method_obj = Method_MLP('multi-layer perceptron', '', hidden_dims=HIDDEN_DIMS)
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
plt.savefig(f'../../result/stage_2_result/convergence_{RUN_NAME}.png', dpi=150)

# ---- record this run (2-4) ----
results_dir = '../../result/stage_2_result/'

# 1. save the loss history, for the comparison plot later
pd.DataFrame(history).to_csv(f'{results_dir}history_{RUN_NAME}.csv', index=False)

# 2. add a row to the experiments table
row = {'run': RUN_NAME, 'hidden_dims': '-'.join(str(h) for h in HIDDEN_DIMS), 'epochs': EPOCHS, 'lr': LEARNING_RATE, 'optimizer': OPTIMIZER,
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