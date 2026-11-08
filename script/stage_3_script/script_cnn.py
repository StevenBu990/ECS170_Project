from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Val_Test import Setting_Train_Val_Test
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib
matplotlib.use('Agg')            # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import csv
import time

# ---- experiment settings: change these for each run ----
DATASET = 'MNIST'
RUN_NAME = 'baseline'
EPOCHS = 10
LEARNING_RATE = 1e-3
OPTIMIZER = 'adam'         # 'adam' or 'sgd'
BATCH_SIZE = 64
DEVICE = 'auto'
# ---------------------------------------------------------

MODELS = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}
VAL_SIZE = {'MNIST': 0.1, 'ORL': 40, 'CIFAR': 0.1}   # ORL: one image per person
RESULTS_DIR = '../../result/stage_3_result/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- parameter section ----
np.random.seed(2)
torch.manual_seed(2)

# ---- object initialization section ----
data_obj = Dataset_Loader(DATASET, '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = DATASET

method_obj = MODELS[DATASET](f'CNN ({DATASET})', '', device=DEVICE)
method_obj.max_epoch = EPOCHS
method_obj.learning_rate = LEARNING_RATE
method_obj.optimizer_name = OPTIMIZER
method_obj.batch_size = BATCH_SIZE

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = f'{RESULTS_DIR}{DATASET}_'
result_obj.result_destination_file_name = RUN_NAME

setting_obj = Setting_Train_Val_Test('train / validation / test', '')
setting_obj.val_size = VAL_SIZE[DATASET]

evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')

# ---- running section ----
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
start = time.time()
scores = setting_obj.load_run_save_evaluate()
train_time = time.time() - start

print('************ Test Performance ************')
for name, value in scores.items():
    print(f'{name:20s} {value:.4f}')
print(f'training + testing time: {train_time:.1f} s on {method_obj.device}')

# ---- learning curves (3-3, 3-4) ----
h = method_obj.history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(h['epoch'], h['loss'], marker='o', color='#2a78d6')
ax1.set_title('Training loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-entropy loss')
ax1.grid(True, color='#e4e4e0')
ax2.plot(h['epoch'], h['acc'], marker='o', color='#2a78d6', label='train')
ax2.plot(h['epoch'], h['val_acc'], marker='s', color='#eb6834', label='validation')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True, color='#e4e4e0')
ax2.legend(frameon=False)
fig.suptitle(f'CNN learning curves: {DATASET} ({RUN_NAME})')
fig.tight_layout()
plt.savefig(f'{RESULTS_DIR}{DATASET}_{RUN_NAME}_curves.png', dpi=150)

# ---- record this run ----
pd.DataFrame(h).to_csv(f'{RESULTS_DIR}{DATASET}_{RUN_NAME}_history.csv', index=False)
row = {'dataset': DATASET, 'run': RUN_NAME, 'epochs': EPOCHS, 'lr': LEARNING_RATE,
       'optimizer': OPTIMIZER, 'batch_size': BATCH_SIZE, 'device': str(method_obj.device),
       'time_s': round(train_time, 1), 'train_acc': round(h['acc'][-1], 4),
       'val_acc': round(float(h['val_acc'][-1]), 4)}
row.update({f'test_{k}': round(v, 4) for k, v in scores.items()})
table_path = RESULTS_DIR + 'experiments.csv'
is_new = not os.path.exists(table_path)
with open(table_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if is_new:
        writer.writeheader()
    writer.writerow(row)
print('recorded run', RUN_NAME, 'in experiments.csv')
print('************ Finish ************')
