# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
from ..archeader import Arch

import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class KNN(Arch):
  def __init__(self, data_config, model_path_dir, k):
    # Dataset and scoring
    super().__init__(data_config, model_path_dir)
    self.set_datasets()
    if not self.PH3:
      self.standardize_data()

    # Model
    self.k = k
    self.me = KNeighborsClassifier(n_neighbors=self.k)
      
  def set_datasets(self):
    X = self.df.drop(columns=[self.label_col])
    y = self.df[self.label_col]

    self.X_train, self.X_test, self.y_train, self.y_test = \
      train_test_split(X, y, test_size=0.2, random_state=42)
    
  def standardize_data(self):
    self.scaler = StandardScaler()
    data_cols = self.df.columns.difference([self.label_col])
    self.df[data_cols] = self.scaler.fit_transform(self.df[data_cols])
    

  def fit(self):
    self.me.fit(self.X_train, self.y_train)

  def predict(self, X):
    return self.me.predict(X)
  
  def keep(self):
    model_path = os.path.join(self.model_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"k{self.k}.pkl"
    )
    with open(model_path, 'wb') as f:
            pickle.dump(self.me, f)


# Record keeping functions
def keep_scores_knn(score, data_config, k):
  sup.knn_score_tracker.append(data_config["class_list"]+[score]+
                               data_config["data_unit"]+data_config["PH2"]+
                               data_config["PH3"]+data_config["reducer"]+
                               data_config["kernel"]+data_config["n"]+[k])

def update_best_knn(score, data_config, k, model):
  if score > sup.best_knn_scores[data_config["data_unit"]]["score"]:
      print(f"updating best... {score}")

      model.keep()

      sup.best_knn_scores[data_config["data_unit"]].update({
          "score": score,
          "data_config": data_config,
          "k": k
      })

def print_best_knn(data_unit):
  best = sup.best_knn_scores[data_unit]
  print(f"Data Unit: {data_unit}")
  print(f"Best score: {best['score']}")
  print(f"Best k: {best['k']}")
  print(f"Best data config: {best['data_config']}")


# Training functions
def try_all_k(data_config, model_path_dir):
  for k in sup.TRAIN_KNN_K_CANDIDATES:
    model = KNN(data_config=data_config, model_path_dir=model_path_dir, k=k)
    model.fit()
    score = model.score()
    #print(f"n={n}; k={k}; score: {score}")

    keep_scores_knn(score, data_config, k)
    update_best_knn(score, data_config, k, model)

def best_KNN(data_unit, label_col, class_list=sup.NUM_CLASSES, test_ratio=0.2):
  model_path_dir = os.path.join(sup.TRAIN_BINGEN_ROOT, sup.TRAIN_KNN_CODE, data_config["data_unit"])
  sup.create_dir_if_not_exists(model_path_dir)

  data_config = {
  "PH2" : None,
  "PH3" : None,
  "reducer": '',
  "kernel": '',
  "n": -1,
  "data_unit": data_unit,
  "label_col": label_col,
  "class_list": class_list,
  "test_ratio": test_ratio,
  }



  for PH2 in [True, False]:
    data_config["PH2"] = PH2
    for PH3 in [True, False]:
      data_config["PH3"] = PH3
      if PH3:
        for n in sup.PH3_N_CANDIDATES:
          data_config["n"] = n
          for reducer in sup.PH3_REDUCER_NAMES:
            data_config["reducer"] = reducer
            if reducer == sup.PH3_REDUCER_NAME_KPCA:
              for kernel in sup.PH3_REDUCER_KERNEL_NAMES:
                data_config["kernel"] = kernel
                try_all_k(data_config, model_path_dir)
            else:
              data_config["kernel"] = ''
              try_all_k(data_config, model_path_dir)
      else:
        data_config["n"] = -1
        data_config["reducer"] = ''
        data_config["kernel"] = ''
        try_all_k(data_config, model_path_dir)
  
  print_best_knn(data_unit)