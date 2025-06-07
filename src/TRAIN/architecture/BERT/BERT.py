# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
from ..archeader import Arch
from ..archeader import update_best

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"device: {device}")

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
  def __init__(self, loadable, input_dim):
    super().__init__()
    loadable_path = os.path.join(sup.TRAIN_BINLOAD_ROOT, loadable)
    self.bert = BertModel.from_pretrained(loadable_path)
    self.bert.embeddings = CustomBertEmbeddings(
        input_dim=input_dim,
        hidden_size=self.bert.config.hidden_size
    )

    self.classifier = nn.Linear(self.bert.config.hidden_size, input_dim)

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
    self.seq_len = 12
    self.input_dim = data_config["input_dim"]
    self.batch_size = data_config["batch_size"]
    self.set_dataloaders()

    # Model
    self.device = train_config["device"]
    self.loadable = train_config["loadable"]
    self.me = BertWithCustomInput(self.loadable, self.input_dim)
    self.me.to(self.device)
    self.lr = train_config["lr"]
    self.optimizer = train_config["optimizer"](self.me.parameters(), self.lr)
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
  
  def fit(self):
     self.me.train()

     for epoch in range(self.num_epochs):
        total_loss = 0
        for xb, yb in self.train_loader:
          xb, yb = xb.to(self.device), yb.to(self.device)
          self.optimizer.zero_grad()
          logits = self.me(xb)
          loss = self.loss_fn(logits, yb)
          loss.backward()
          self.optimizer.step()
          total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(self.train_loader):.4f}")
  
  def score(self):
    self.me.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
      for xb, yb in self.test_loader:
          xb = xb.to(self.device)
          logits = self.me(xb)
          preds = logits.argmax(dim=1).cpu()
          all_preds.append(preds)
          all_labels.append(yb)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)

    self.accuracy = accuracy_score(y_true, y_pred)

  def keep(self):
    sup.create_dir_if_not_exists(self.model_path_dir)
    model_path = os.path.join(self.model_path_dir,
                              f"{self.PH2}-"\
                              f"{self.PH3}-"\
                              f"{self.reducer}-"\
                              f"{self.kernel}-"\
                              f"n{self.n}-"\
                              f"{self.loadable}.pth"
    )

    torch.save(self.me.state_dict(), model_path)
     

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
BERT_TINY="prajjwal1-bert-tiny"
BERT_BASE="bert-base-uncased"
BERT_LARGE="bert-large-uncased"
BERT_CANDIDATES = [DISTILBERT, BERT_TINY, BERT_BASE, BERT_LARGE]

download_if_not_exists(DISTILBERT)
download_if_not_exists(BERT_TINY)
download_if_not_exists(BERT_BASE)
download_if_not_exists(BERT_LARGE)

# Record keeping functions
def keep_scores_bert(model:BERT):
  sup.bert_score_tracker.append([model.class_list, model.accuracy,
                               model.data_unit, model.PH2,
                               model.PH3, model.reducer,
                               model.kernel, model.n, model.loadable,
                               model.lr, model.optimizer, model.loss_fn,
                               model.num_epochs])

# Training functions
BERT_lr_CANDIDATES = [2e-5, 2e-6]
BERT_optimizer_CANDIDATES = [optim.AdamW]
BERT_loss_fn_CANDIDATES = [nn.CrossEntropyLoss]
BERT_num_epochs_CANDIDATES = [1000, 5000, 10000]

def try_bert_train_configs(data_config):
  first = True
  for load_name in BERT_CANDIDATES:
    for lr in BERT_lr_CANDIDATES:
      for optimizer in BERT_optimizer_CANDIDATES:
        for loss_fn in BERT_loss_fn_CANDIDATES:
          for num_epochs in BERT_num_epochs_CANDIDATES:

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