# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
from ..archeader import Arch

from sklearn.neighbors import KNeighborsClassifier
from torch import vstack, cat

class KNN(Arch):
  def __init__(self, data_path, label_col, k, **kwargs):
    super().__init__(data_path, label_col, **kwargs)
    self.me = KNeighborsClassifier(n_neighbors=k)

  def fit(self):
    self.me.fit(self.X_train.numpy(), self.y_train.numpy())

  def predict(self, X):
    return self.me.predict(X.numpy())
    