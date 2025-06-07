# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


###############################################################################
############################# Generic Architecture ############################

class Arch():
  def __init__(self, data_config, df, train_config, model_path_dir):
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
    self.test_ratio = data_config["test_ratio"]

    if df == None:
      self.set_datapath()
      self.set_dataframe()
      if not self.PH3:
        self.standardize_data()
    else:
      self.df = df

    # Model
    self.train_config = train_config
    self.me = None
    self.model_path_dir = model_path_dir

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

  def set_dataframe(self):
    full_df = pd.read_csv(self.datapath)
    nonlabel = sup.class_numeric_column \
                  if self.label_col == sup.active_hand_col \
                  else sup.active_hand_col
    
    self.df = full_df.copy()
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

############################# Generic Architecture ############################
###############################################################################
##################################### KNN #####################################

from .KNN import knn
from .KNN.knn import KNN
from .BERT.bert import BERT
from .BERT import bert

##################################### KNN #####################################
###############################################################################