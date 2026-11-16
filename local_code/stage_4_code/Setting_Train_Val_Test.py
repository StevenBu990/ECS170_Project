
'''
Stage 3 setting: one file holds train + test. Hold out a stratified validation set from
the training data, train the method, then evaluate once on the test set.
'''

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split

class Setting_Train_Val_Test(setting):
    val_size = 0.1      # fraction of training images (or a whole number of images) to hold out

    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X, y = data['train']['X'], data['train']['y']

        # split X, y into training and validation parts
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=self.val_size, stratify=y, random_state=2)

        # give the method 3 splits
        self.method.data = {'train': {'X': X_tr, 'y': y_tr}, 'val': {'X': X_val, 'y': y_val}, 'test': data['test']}

        # run, save, evaluate, and return the scores
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result

        return self.evaluate.evaluate()