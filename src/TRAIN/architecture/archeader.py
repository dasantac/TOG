# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup

import pandas as pd
from sklearn.preprocessing import StandardScaler

sup.report_dir_if_not_exists(sup.PH1_DATA_ROOT)
sup.report_dir_if_not_exists(sup.PH2_DATA_ROOT)
sup.report_dir_if_not_exists(sup.PH3_DATA_ROOT)

###############################################################################
############################# Generic Architecture ############################

class Arch():
  def __init__(self, data_config, df, train_config):
    # Dataset
    self.data_config = data_config
    self.PH2 = data_config["PH2"]
    self.PH3 = data_config["PH3"]
    self.reducer = data_config["reducer"]
    self.kernel = data_config["kernel"]
    self.n = data_config["n"]
    self.data_unit = data_config["data_unit"]
    self.label_col = data_config["label_col"]
    self.class_list = data_config["class_list"]
    self.class_numeric_list = sup.get_class_numeric_list(
                                sup.get_class_list(data_config["class_list"]))
    self.test_ratio = 0.2

    if df is None:
      self.set_datapath()
      self.set_dataframe()
      if not self.PH3:
        self.standardize_data()
    else:
      self.df = df

    # Model
    self.train_config = train_config
    self.me = None
    self.arch = train_config["arch"]

    # score
    self.accuracy = None

  ### Dataset helper functions start here ###
  def set_datapath(self):
    if not self.PH2 and not self.PH3:
      self.datapath = os.path.join(sup.PH1_DATA_ROOT, f"{sup.DATA_AH_PF}.csv")
    elif self.PH2 and not self.PH3:
      self.datapath = os.path.join(sup.PH2_DATA_ROOT, f"{sup.DATA_AH_PF}.csv")
    elif not self.PH2 and self.PH3:
      self.datapath = os.path.join(sup.PH3_DATA_ROOT,
                                   sup.PH3_WO2_CODE,
                                   self.reducer,
                                   self.kernel,
                                   f"{sup.DATA_AH_PF}_{self.n}.csv")
    elif self.PH2 and self.PH3:
      self.datapath = os.path.join(sup.PH3_DATA_ROOT,
                                   sup.PH3_W2_CODE,
                                   self.reducer,
                                   self.kernel,
                                   f"{sup.DATA_AH_PF}_{self.n}.csv")
    else:
      print("bad data_config")

  def __flatten_group(self, group):
    data_cols = [col for col in group.columns if col not in 
                  sup.tag_columns + sup.class_columns + [sup.current_frame_col]]

    flattened_dict = {}

    for _, row in group.iterrows():
        frame_num = int(row[sup.current_frame_col])
        prefix = f"f{frame_num}_"
        for col in data_cols:
            flattened_dict[prefix + col] = row[col]

    # Add the group keys (the tags)
    group_keys = group.iloc[0][sup.tag_columns + sup.class_columns].to_dict()
    group_keys.update(flattened_dict)
    return pd.Series(group_keys)

  def __filter_data_unit(self):
    if self.data_unit == sup.DATA_AH_PF:
      return self.df
    
    spf_df = self.df[self.df[sup.active_hand_col] == 1]
    
    if self.data_unit == sup.DATA_S_PF:
      spf_df = spf_df.drop(columns=[sup.active_hand_col])
      return spf_df
    
    spv_df = spf_df.groupby(sup.tag_columns+sup.class_columns).apply(self.__flatten_group).reset_index(drop=True)
    spf_df = spf_df.drop(columns=[sup.active_hand_col])
    
    if self.data_unit == sup.DATA_S_PV:
      return spv_df

  def __filter_class_list(self):
    self.df = self.df[self.df['class_numeric'].isin(self.class_numeric_list)]
    return self.df
  
  def set_dataframe(self):
    full_df = pd.read_csv(self.datapath)
    nonlabel = sup.class_numeric_column \
                  if self.label_col == sup.active_hand_col \
                  else sup.active_hand_col
    
    self.df = full_df.copy()
    self.df = self.__filter_class_list()
    self.df = self.__filter_data_unit()
    self.df = self.df.drop(columns=sup.tag_columns+[nonlabel, sup.current_frame_col], errors='ignore')

  def standardize_data(self):
    self.scaler = StandardScaler()
    data_cols = self.df.columns.difference([self.label_col])
    self.df[data_cols] = self.scaler.fit_transform(self.df[data_cols])
  ### Dataset helper functions end here###

  def fit(self):
    pass

  def score(self):
    pass
  
  def keep(self):
    pass

# Score tracking
def print_best(arch, data_unit):
  best = sup.best_scores[arch][data_unit]
  print(f"Data Unit: {data_unit}")
  print(f"Best score: {best['accuracy']}")
  print(f"Best data config: {best['data_config']}")
  print(f"Best train config: {best['train_config']}")

def update_best(model:Arch):
  if model.accuracy > sup.best_scores[model.arch][model.data_unit]["accuracy"]:
      print(f"updating best... {model.accuracy}")
      print(f"\t{model.data_config}")
      print(f"\t{model.train_config}")

      model.keep()

      sup.best_scores[model.arch][model.data_unit].update({
          "accuracy": model.accuracy,
          "data_config": model.data_config.copy(),
          "train_config": model.train_config.copy()
      })
############################# Generic Architecture ############################
###############################################################################
############################ Specific Architectures ###########################

from .KNN import knn
from .KNN.knn import KNN
from .BERT.bert import BERT
from .BERT import bert

# Taining
def find_best(data_unit, label_col, class_list, arch):
  if arch == sup.TRAIN_KNN_CODE:
    knn.try_data_configs(data_unit, label_col, class_list)
  elif arch == sup.TRAIN_BERT_CODE:
    bert.try_data_configs(data_unit, label_col, class_list)

  print_best(arch, data_unit)

############################ Specific Architectures ###########################
###############################################################################

