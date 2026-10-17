'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None
    
    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        scores = {}

        # store accuracy in scores['accuracy]
        accuracy = accuracy_score(y_true, y_pred)
        scores['accuracy'] = accuracy

        # store weighted precision, recall and F1
        precision_weighted = precision_score(y_true, y_pred, average="weighted")
        scores['precision_weighted'] = precision_weighted
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        scores['recall_weighted'] = recall_weighted
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        scores['f1_weighted'] = f1_weighted

        # store macro precision, recall, and F1
        precision_macro = precision_score(y_true, y_pred, average="macro")
        scores['precision_macro'] = precision_macro
        recall_macro = recall_score(y_true, y_pred, average="macro")
        scores['recall_macro'] = recall_macro
        f1_macro = f1_score(y_true, y_pred, average="macro")
        scores['f1_macro'] = f1_macro

        # store micro precision, recall, and F1
        precision_micro = precision_score(y_true, y_pred, average="micro")
        scores['precision_micro'] = precision_micro
        recall_micro = recall_score(y_true, y_pred, average="micro")
        scores['recall_micro'] = recall_micro
        f1_micro = f1_score(y_true, y_pred, average="micro")
        scores['f1_micro'] = f1_micro

        return scores