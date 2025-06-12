# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
from ..archeader import Arch
from ..archeader import print_best
from ..archeader import update_best

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import gc

###############################################################################
####################### K Nearest Neighbors architecture ######################
class KNN(Arch):
  def __init__(self, data_config, df, train_config):
    # Dataset and scoring
    super().__init__(data_config, df, train_config)
    self.set_datasets()

    # Model
    self.k = train_config["k"]
    self.me = KNeighborsClassifier(n_neighbors=self.k)
      
  def set_datasets(self):
    X = self.df.drop(columns=[self.label_col])
    y = self.df[self.label_col]

    self.X_train, self.X_test, self.y_train, self.y_test = \
      train_test_split(X, y, test_size=0.2, random_state=42)

  def fit(self, verbose=False):
    self.me.fit(self.X_train, self.y_train)
  
  def test(self):
    self.y_true = self.y_test.values
    self.y_logits = self.me.kneighbors(self.X_test, 
                                       n_neighbors=self.num_classes)[0]
    self.y_pred = self.me.predict(self.X_test)

  def keep_confusion_matrix(self):
    confusion_path_dir = os.path.join(sup.TRAIN_MEDIAGEN_ROOT, self.class_list,
                                       str(len(self.class_numeric_list)),  
                                       sup.TRAIN_KNN_CODE, self.data_unit,
                                       "confusion")
    sup.create_dir_if_not_exists(confusion_path_dir)

    confusion_path = os.path.join(confusion_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"k{self.k}-"\
                              f"s{self.accuracy}.jpg"
    )

    self.plot_confusion_matrix()
    self.confusion_fig.savefig(confusion_path, dpi=300, bbox_inches='tight')
  
  def keep(self):
    model_path_dir = os.path.join(sup.TRAIN_BINGEN_ROOT, self.class_list, 
                                       str(len(self.class_numeric_list)), 
                                       sup.TRAIN_KNN_CODE, self.data_unit)
    sup.create_dir_if_not_exists(model_path_dir)
    model_path = os.path.join(model_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"k{self.k}-"\
                              f"s{self.accuracy}.pkl"
    )
    with open(model_path, 'wb') as f:
            pickle.dump(self.me, f)
  
    self.keep_confusion_matrix()
####################### K Nearest Neighbors architecture ######################
###############################################################################

# Record keeping functions
def keep_scores_knn(model:KNN):
  reducer = model.reducer if model.reducer else 'None'
  kernel = model.kernel if model.kernel else 'None'
  sup.knn_score_tracker.append([model.class_list, model.accuracy,
                               model.data_unit, model.PH2,
                               model.PH3, reducer, kernel, model.n, model.k])

# Training functions
TRAIN_KNN_K_CANDIDATES = [k for k in range(1,16)]

def try_train_configs(data_config):
  for k in TRAIN_KNN_K_CANDIDATES:
    train_config = {"arch" : sup.TRAIN_KNN_CODE, "k" : k}
    if k == 1:
      model = KNN(data_config=data_config, df=None, train_config=train_config)
      save_df = model.df
    else:
      model = KNN(data_config=data_config, df=save_df, train_config=train_config)

    model.fit()
    model.score()

    keep_scores_knn(model)
    update_best(model)

    del model
    gc.collect()

def try_data_configs(data_unit, label_col, class_list):
  data_config = {
    "PH2" : None,
    "PH3" : None,
    "reducer": '',
    "kernel": '',
    "n": -1,
    "data_unit": data_unit,
    "label_col": label_col,
    "class_list": class_list
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
                try_train_configs(data_config)
            else:
              data_config["kernel"] = ''
              try_train_configs(data_config)
      else:
        data_config["n"] = -1
        data_config["reducer"] = ''
        data_config["kernel"] = ''
        try_train_configs(data_config)

def find_best(data_unit, label_col, class_list):
  try_data_configs(data_unit, label_col, class_list)
  
  print_best(sup.TRAIN_KNN_CODE, data_unit)