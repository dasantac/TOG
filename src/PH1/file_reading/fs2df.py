"""
Filesystem (fs) to (2) DataFrame (df)

This file contains helper functions in the process of passing from a
filesystem-like organization of the projects data to data stored in pandas
dataframes.
"""

# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup

###############################################################################
####################### nltk corpus to pandas dataframe #######################

import unicodedata

def get_person_id(fileid:str):
  person_id = fileid.split('/')[0]
  return person_id

def get_cycle_num(fileid):
  second_part = fileid.split('/')[1]
  cycle_num = second_part.split('_')[1]
  return cycle_num

def get_handedness(fileid):
  second_part = fileid.split('/')[1]
  handedess_name = second_part.split('_')[-1]
  handedness = -1
  if handedess_name in ['Derecha']:
    handedness = 0
  elif handedess_name in ['Izquierda']:
    handedness = 1
  else: 
    print(f" bad handedness in file: {fileid}")
  return handedness

def get_class_name(fileid):
  last_part = fileid.split('/')[-1]
  class_dot_mp4 = last_part.split('_')[-1]
  class_name = class_dot_mp4.split('.')[0]
  norm_class_name = unicodedata.normalize("NFC", class_name)
  return norm_class_name


def get_tags(fileid):
  person_id = get_person_id(fileid)
  cycle_num = get_cycle_num(fileid)
  handedness = get_handedness(fileid)
  class_name = get_class_name(fileid)
  class_numeric = sup.CLASSES_TO_NUMBERS[class_name]
  return person_id, cycle_num, handedness, class_name, class_numeric, fileid

####################### nltk corpus to pandas dataframe #######################
###############################################################################