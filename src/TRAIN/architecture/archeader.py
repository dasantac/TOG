# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
import TRAINheader as train

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


###############################################################################
############################# Generic Architecture ############################

class Arch():
  def __init__(self, data_path, label_col, batch_size=32, test_split=0.2):
    self.data_path = data_path
    self.label_col = label_col
    self.batch_size = batch_size
    self.test_split = test_split

    self.me = None

    # data
    self._prepare_data()

    # score
    self.accuracy = None
    self.f1 = None


  def _prepare_data(self):
    df = pd.read_csv(self.data_path)

    y = torch.tensor(df[self.label_col].values, dtype=torch.long)
    X = torch.tensor(df.drop(columns=[self.label_col]).values, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    test_size = int(len(dataset) * self.test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

  def fit(self):
    pass

  def predict(self, X):
    pass

  def score(self):
    y_pred = self.predict(self.X_test)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    y_true = self.y_test.numpy()

    self.accuracy = accuracy_score(y_true, y_pred)
    self.f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {self.accuracy}")
    print(f"F1 score: {self.f1}")
    return self.accuracy, self.f1
  
############################# Generic Architecture ############################
###############################################################################
##################################### KNN #####################################

from .KNN.knn import KNN

##################################### KNN #####################################
###############################################################################