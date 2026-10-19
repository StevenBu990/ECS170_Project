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

method_obj = Method_MLP('multi-layer perceptron', '')

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
plt.title('Training loss vs. Epoch')
plt.grid(True)
plt.savefig('../../result/stage_2_result/convergence.png', dpi=150)

print('************ Finish ************')