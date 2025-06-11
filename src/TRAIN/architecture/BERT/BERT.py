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

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"device: {device}")

from tqdm import tqdm

### Helper classes ###
# --- Custom BERT Embeddings ---

class CustomBertEmbeddings(nn.Module):
  def __init__(self, input_dim, hidden_size, max_position_embeddings=512, dropout=0.1):
    super().__init__()
    self.linear = nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size)
    )
    self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
    self.token_type_embeddings = nn.Embedding(2, hidden_size)
    self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
    self.dropout = nn.Dropout(dropout)

  def forward(
    self,
    input_ids=None,             # Needed for BERT compatibility
    token_type_ids=None,
    position_ids=None,
    inputs_embeds=None,
    past_key_values_length=0
):
    if inputs_embeds is None:
        raise ValueError("inputs_embeds must be provided")

    B, L, _ = inputs_embeds.size()

    if position_ids is None:
        position_ids = torch.arange(L, device=inputs_embeds.device).unsqueeze(0).expand(B, L)
    if token_type_ids is None:
        token_type_ids = torch.zeros((B, L), dtype=torch.long, device=inputs_embeds.device)

    inputs_proj = self.linear(inputs_embeds)
    pos_embed = self.position_embeddings(position_ids + past_key_values_length)
    token_type_embed = self.token_type_embeddings(token_type_ids)
    embeddings = inputs_proj + pos_embed + token_type_embed
    embeddings = self.LayerNorm(embeddings)
    return self.dropout(embeddings)
    
# --- Model ---

class BertWithCustomInput(nn.Module):
  def __init__(self, loadable, input_dim, output_dim):
    super().__init__()
    loadable_path = os.path.join(sup.TRAIN_BINLOAD_ROOT, loadable)
    self.bert = BertModel.from_pretrained(loadable_path)
    self.bert.embeddings = CustomBertEmbeddings(
        input_dim=input_dim,
        hidden_size=self.bert.config.hidden_size
    )

    self.classifier = nn.Linear(self.bert.config.hidden_size, output_dim)

  def forward(self, x):
    B, L, _ = x.shape
    position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    token_type_ids = torch.zeros((B, L), dtype=torch.long, device=x.device)

    outputs = self.bert(
        inputs_embeds=x,
        position_ids=position_ids,
        token_type_ids=token_type_ids
    )
    return self.classifier(outputs.pooler_output)

###############################################################################
############################## BERT architecture ##############################

class BERT(Arch):
  def __init__(self, data_config, df, train_config):
    # Dataset and scoring
    super().__init__(data_config, df, train_config)
    self.seq_len = data_config["seq_len"]
    self.input_dim = data_config["input_dim"]
    self.output_dim = len(self.class_numeric_list)
    self.batch_size = data_config["batch_size"]
    self.set_dataloaders()

    # Model
    self.device = train_config["device"]
    self.loadable = train_config["loadable"]
    self.me = BertWithCustomInput(self.loadable, self.input_dim, self.output_dim)
    self.me.to(self.device)
    self.lr = train_config["lr"]
    self.weight_decay = train_config["weight_decay"]
    self.optimizer = train_config["optimizer"](self.me.parameters(), 
                                               lr=self.lr,
                                               weight_decay=self.weight_decay)
    self.loss_fn = train_config["loss_fn"]()
    self.num_epochs = train_config["num_epochs"]

  def set_dataloaders(self):
    y = torch.tensor(self.df[self.label_col].values, dtype=torch.long)
    X_flat = self.df.drop(columns=[self.label_col]).values
    X = torch.tensor(X_flat.reshape(-1, self.seq_len, self.input_dim), 
                     dtype=torch.float32)
    
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
      X, y, test_size=self.test_ratio, stratify=y)
    
    self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), 
                                    batch_size=self.batch_size, shuffle=True)
    self.test_loader = DataLoader(TensorDataset(self.X_test, self.y_test), 
                                    batch_size=self.batch_size)
  
  def fit(self, verbose=False):
    self.me.train()

    epoch_iter = range(self.num_epochs)
    if verbose:
      epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch", leave=False)

    for epoch in epoch_iter:
      total_loss = 0

      for xb, yb in self.train_loader:
        xb, yb = xb.to(self.device), yb.to(self.device)
        self.optimizer.zero_grad()
        logits = self.me(xb)
        loss = self.loss_fn(logits, yb)
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()

      self.loss_list.append(total_loss)

      if verbose:
        epoch_iter.set_postfix(epoch=epoch+1, loss=total_loss)

  def test(self):
    self.me.eval()
    all_preds, all_labels = [], []
    all_logits = []
    with torch.no_grad():
      for xb, yb in self.test_loader:
          xb = xb.to(self.device)
          logits = self.me(xb)
          preds = logits.argmax(dim=1).cpu()
          all_preds.append(preds)
          all_labels.append(yb)
          all_logits.append(logits.cpu())

    self.y_true = torch.cat(all_labels)
    self.y_logits = torch.cat(all_logits)
    self.y_pred = torch.cat(all_preds)
      
  def keep_loss(self):
    loss_path_dir = os.path.join(sup.TRAIN_MEDIAGEN_ROOT, self.class_list,
                                       str(len(self.class_numeric_list)), 
                                       sup.TRAIN_BERT_CODE, self.data_unit,
                                       self.loadable, "loss")
    sup.create_dir_if_not_exists(loss_path_dir)

    loss_path = os.path.join(loss_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"lr{self.lr}-"\
                              f"ep{len(self.loss_list)}"\
                              f"s{self.accuracy}.jpg"
    )

    self.plot_loss()
    self.loss_fig.savefig(loss_path, dpi=300, bbox_inches='tight')

  def keep_confusion_matrix(self):
    confusion_path_dir = os.path.join(sup.TRAIN_MEDIAGEN_ROOT, self.class_list,
                                       str(len(self.class_numeric_list)),  
                                       sup.TRAIN_BERT_CODE, self.data_unit,
                                       self.loadable, "confusion")
    sup.create_dir_if_not_exists(confusion_path_dir)

    confusion_path = os.path.join(confusion_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"lr{self.lr}-"\
                              f"ep{len(self.loss_list)}"\
                              f"s{self.accuracy}.jpg"
    )

    self.plot_confusion_matrix()
    self.confusion_fig.savefig(confusion_path, dpi=300, bbox_inches='tight')

  def keep(self):
    model_path_dir = os.path.join(sup.TRAIN_BINGEN_ROOT, self.class_list,
                                       str(len(self.class_numeric_list)),  
                                       sup.TRAIN_BERT_CODE, self.data_unit,
                                       self.loadable)
    sup.create_dir_if_not_exists(model_path_dir)

    model_path = os.path.join(model_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"lr{self.lr}-"\
                              f"ep{len(self.loss_list)}"\
                              f"s{self.accuracy}.pth"
    )


    torch.save(self.me.state_dict(), model_path)
    
    # Also save training loss information and confusion matrix
    self.keep_loss()
    self.keep_confusion_matrix()
     

############################## BERT architecture ##############################
###############################################################################

# Making sure the loadables are there
def download_if_not_exists(model_name):
  model_dir = os.path.join(sup.TRAIN_BINLOAD_ROOT,model_name)
  if os.path.exists(model_dir)==False:
    print(f"Directory {model_dir} does not exist. Downloading the model and "
           "continuing with the execution")
    model = BertModel.from_pretrained(model_name)
    model.save_pretrained(model_dir)
  else:
    print(f"Directory {model_dir} exists. Continuing with execution")

DISTILBERT= "distilbert-base-uncased"
BERT_TINY="prajjwal1/bert-tiny"
BERT_BASE="bert-base-uncased"
BERT_LARGE="bert-large-uncased"
BERT_LOADABLE_CANDIDATES = [BERT_TINY]

download_if_not_exists(DISTILBERT)
download_if_not_exists(BERT_TINY)
download_if_not_exists(BERT_BASE)
download_if_not_exists(BERT_LARGE)

# Record keeping functions
def keep_scores_bert(model:BERT):
  reducer = model.reducer if model.reducer else 'None'
  kernel = model.kernel if model.kernel else 'None'
  sup.bert_score_tracker.append([model.class_list, model.accuracy,
                               model.data_unit, model.PH2,
                               model.PH3, reducer, kernel, model.n, 
                               model.loadable, model.lr, model.optimizer, 
                               model.loss_fn, model.num_epochs])

# Training parameters
BERT_PH2_CANDIDATES = [True, False]
BERT_PH3_CANDIDATES = [True, False]
BERT_N_CANDIDATES = [3, 7, 11, 15]
BERT_REDUCER_CANDIDATES = [sup.PH3_REDUCER_NAME_UMAP]
BERT_REDUCER_KERNEL_CANDIDATES = []
BERT_lr_CANDIDATES = [1e-5]
BERT_optimizer_CANDIDATES = [optim.AdamW]
BERT_loss_fn_CANDIDATES = [nn.CrossEntropyLoss]
BERT_num_epochs_CANDIDATES = [1000]

# Training functions
def try_train_configs(data_config,
                      LOADABLE_CANDIDATES=BERT_LOADABLE_CANDIDATES,
                      lr_CANDIDATES=BERT_lr_CANDIDATES,
                      optimizer_CANDIDATES=BERT_optimizer_CANDIDATES,
                      loss_fn_CANDIDATES=BERT_loss_fn_CANDIDATES,
                      num_epochs_CANDIDATES=BERT_num_epochs_CANDIDATES):
  first = True

  for load_name in LOADABLE_CANDIDATES:
    for lr in lr_CANDIDATES:
      for optimizer in optimizer_CANDIDATES:
        for loss_fn in loss_fn_CANDIDATES:
          for num_epochs in num_epochs_CANDIDATES:

            train_config = {
              "arch" : sup.TRAIN_BERT_CODE,
              "device" : device,
              "loadable" : load_name,
              "optimizer" : optimizer,
              "lr" : lr,
              "loss_fn" : loss_fn,
              "num_epochs" : num_epochs
            }

            if first:
              model = BERT(data_config=data_config, df=None, 
                           train_config=train_config)
              save_df = model.df
              first = False
            else:
              model = BERT(data_config=data_config, df=save_df, 
                           train_config=train_config)
              
            model.fit()
            model.score()

            keep_scores_bert(model)
            update_best(model)

def try_data_configs(data_unit, label_col, class_list,
                     PH2_CANDIDATES=BERT_PH2_CANDIDATES,
                     PH3_CANDIDATES=BERT_PH3_CANDIDATES,
                     N_CANDIDATES=BERT_N_CANDIDATES,
                     REDUCER_CANDIDATES=BERT_REDUCER_CANDIDATES,
                     KERNEL_CANDIDATES=BERT_REDUCER_KERNEL_CANDIDATES,
                     LOADABLE_CANDIDATES=BERT_LOADABLE_CANDIDATES,
                     lr_CANDIDATES=BERT_lr_CANDIDATES,
                     optimizer_CANDIDATES=BERT_optimizer_CANDIDATES,
                     loss_fn_CANDIDATES=BERT_loss_fn_CANDIDATES,
                     num_epochs_CANDIDATES=BERT_num_epochs_CANDIDATES):
  data_config = {
    "PH2" : None,
    "PH3" : None,
    "reducer": '',
    "kernel": '',
    "n": -1,
    "data_unit": data_unit,
    "label_col": label_col,
    "class_list": class_list,
    "batch_size": 256
    }
  
  if data_unit == sup.DATA_S_PV:
    data_config["seq_len"] = sup.NUM_FRAMES_PER_VIDEO
  else:
    data_config["seq_len"] = 1

  for PH2 in PH2_CANDIDATES:
    data_config["PH2"] = PH2
    for PH3 in PH3_CANDIDATES:
      data_config["PH3"] = PH3
      if PH3:
        for n in N_CANDIDATES:
          data_config["n"] = n
          data_config["input_dim"] = n
          for reducer in REDUCER_CANDIDATES:
            data_config["reducer"] = reducer
            if reducer == sup.PH3_REDUCER_NAME_KPCA:
              for kernel in KERNEL_CANDIDATES:
                data_config["kernel"] = kernel
                try_train_configs(data_config=data_config,
                      LOADABLE_CANDIDATES=LOADABLE_CANDIDATES,
                      lr_CANDIDATES=lr_CANDIDATES,
                      optimizer_CANDIDATES=optimizer_CANDIDATES,
                      loss_fn_CANDIDATES=loss_fn_CANDIDATES,
                      num_epochs_CANDIDATES=num_epochs_CANDIDATES)
            else:
              data_config["kernel"] = ''
              try_train_configs(data_config=data_config,
                      LOADABLE_CANDIDATES=LOADABLE_CANDIDATES,
                      lr_CANDIDATES=lr_CANDIDATES,
                      optimizer_CANDIDATES=optimizer_CANDIDATES,
                      loss_fn_CANDIDATES=loss_fn_CANDIDATES,
                      num_epochs_CANDIDATES=num_epochs_CANDIDATES)
      else:
        data_config["n"] = -1
        if PH2:
          data_config["input_dim"] = 87
        else:
          data_config["input_dim"] = 72
        data_config["reducer"] = ''
        data_config["kernel"] = ''
        try_train_configs(data_config=data_config,
                      LOADABLE_CANDIDATES=LOADABLE_CANDIDATES,
                      lr_CANDIDATES=lr_CANDIDATES,
                      optimizer_CANDIDATES=optimizer_CANDIDATES,
                      loss_fn_CANDIDATES=loss_fn_CANDIDATES,
                      num_epochs_CANDIDATES=num_epochs_CANDIDATES)
        
def find_best(data_unit, label_col, class_list,
                     PH2_CANDIDATES=BERT_PH2_CANDIDATES,
                     PH3_CANDIDATES=BERT_PH3_CANDIDATES,
                     N_CANDIDATES=BERT_N_CANDIDATES,
                     REDUCER_CANDIDATES=BERT_REDUCER_CANDIDATES,
                     KERNEL_CANDIDATES=BERT_REDUCER_KERNEL_CANDIDATES,
                     LOADABLE_CANDIDATES=BERT_LOADABLE_CANDIDATES,
                     lr_CANDIDATES=BERT_lr_CANDIDATES,
                     optimizer_CANDIDATES=BERT_optimizer_CANDIDATES,
                     loss_fn_CANDIDATES=BERT_loss_fn_CANDIDATES,
                     num_epochs_CANDIDATES=BERT_num_epochs_CANDIDATES):
  
  try_data_configs(data_unit, label_col, class_list,
                     PH2_CANDIDATES=PH2_CANDIDATES,
                     PH3_CANDIDATES=PH3_CANDIDATES,
                     N_CANDIDATES=N_CANDIDATES,
                     REDUCER_CANDIDATES=REDUCER_CANDIDATES,
                     KERNEL_CANDIDATES=KERNEL_CANDIDATES,
                     LOADABLE_CANDIDATES=LOADABLE_CANDIDATES,
                     lr_CANDIDATES=lr_CANDIDATES,
                     optimizer_CANDIDATES=optimizer_CANDIDATES,
                     loss_fn_CANDIDATES=loss_fn_CANDIDATES,
                     num_epochs_CANDIDATES=num_epochs_CANDIDATES)
  
  print_best(sup.TRAIN_BERT_CODE, data_unit)