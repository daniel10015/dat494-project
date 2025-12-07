import kagglehub
from kagglehub import KaggleDatasetAdapter

import os
import pandas as pd

verbose = True
def print_verbose(*args):
  if verbose:
    print(*args)

dataset_dir = 'dataset/'
train_filename = f'{dataset_dir}QnA-train.csv'
valid_filename = f'{dataset_dir}QnA-valid.csv'
cleaned_train_filename = f'{dataset_dir}train_cleaned_QnA.csv'
cleaned_valid_filename = f'{dataset_dir}valid_cleaned_QnA.csv'
dataAlreadyLoaded = os.path.isfile(train_filename)

os.makedirs(dataset_dir, exist_ok=True)

# next step is to clean the dataset by only extracting the title and body data as well as the label classification
# see https://stackoverflow.com/questions/5002111/how-to-strip-html-tags-from-string-in-javascript
# for removing html tags from the text

from bs4 import BeautifulSoup
import html

def clean_html(s):
  if not s:
    return ''
  soup = BeautifulSoup(s, 'html')
  # strip all the code
  for code_tag in soup.find_all("code"):
    code_tag.replace_with('')
  for tag in soup(['script', 'style']):
    tag.decompose()
  return html.unescape(soup.get_text(separator='\n')).strip()

def clean_df(df):
  cleaned = pd.DataFrame()
  cleaned["Body"] = df["Body"].apply(clean_html)
  cleaned["Y"] = df["Y"]
  return cleaned

def download_dataset(filename: str):
  return kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "imoore/60k-stack-overflow-questions-with-quality-rate",
    filename
  )

def load_dataset(raw_path, clean_path, dataset_filename):
  # first check if clean data exists, then raw path, then download dataset

  if os.path.isfile(clean_path):
    return pd.read_csv(clean_path)
  elif os.path.isfile(raw_path):
    df_raw = pd.read_csv(raw_path)
  else:
    df_raw = download_dataset(filename=dataset_filename)
    df_raw.to_csv(raw_path, index=False)
  
  # clean
  df_clean = clean_df(df_raw)
  df_clean.to_csv(clean_path, index=False)
  return df_clean

cleaned_df_train = load_dataset(train_filename, cleaned_train_filename, 'train.csv')
cleaned_df_valid = load_dataset(valid_filename, cleaned_valid_filename, 'valid.csv')

import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# Test GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

print("BERT model loaded successfully!")

from datasets import Dataset, load_from_disk
# Helper functions to ease loading in dataset  
label_map = {'HQ': 0, 'LQ_EDIT' : 1, 'LQ_CLOSE': 2}

def preprocess_and_save(max_len: int, path: str, type):
  print(f'Preprocessing Stackoverflow dataset with maxlen={max_len}')
  df = cleaned_df_train if type == 'train' else cleaned_df_valid

  df_process = pd.DataFrame()
  df_process['text'] = df['Body'].str.replace("\n", " ", regex=False).astype(str)
  df_process['label'] = df['Y'].map(label_map).astype(int)
  hf_dataset = Dataset.from_pandas(df_process)

  def preprocess_function(examples):
    return tokenizer(
      examples["text"],
      truncation=True,
      padding="max_length",
      max_length=max_len
    )
  
  tokenized = hf_dataset.map(preprocess_function, batched=True)
  tokenized.save_to_disk(dataset_path=path)
  print(f'saved dataset to {path}')

  return tokenized
  
def load_stackoverflow_dataset(max_len: int, type: str):
  path = f'stackoverflow_tokenized_{type}_length{max_len}'

  if os.path.exists(path):
    print('Loading from cached file')
    return load_from_disk(path)
  
  return preprocess_and_save(max_len=max_len, path=path, type=type)


# Look at the first datapoint
seq_len = 128
tokenized_dataset_train = load_stackoverflow_dataset(max_len=seq_len, type='train')
tokenized_dataset_valid = load_stackoverflow_dataset(max_len=seq_len, type='valid')

finetune_type = 'last_layer'
if finetune_type == 'last_layer' or finetune_type == 'gradual_unfreeze':
  # Freeze all layers except the classifier
  for param in model.bert.parameters():
    param.requires_grad = False
  # Keep only the classification head trainable
  for param in model.classifier.parameters():
    param.requires_grad = True
elif finetune_type == 'all_layers':
  for param in model.bert.parameters():
    param.requires_grad = True
  for param in model.classifier.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

training_args = TrainingArguments(
  output_dir = 'output/',
  eval_strategy='epoch',
  learning_rate=1e-4,
  per_device_eval_batch_size=256, # batch size=256 is optimal for training speed on my device
  per_device_train_batch_size=256,
  num_train_epochs=2,
  weight_decay = 1e-2,
  save_strategy='epoch',
  save_total_limit=2,
  load_best_model_at_end=True,
  logging_dir='log/',
  logging_steps=5,
  label_smoothing_factor=0.1
)

from evaluate import load

# Source - https://stackoverflow.com/a
# Posted by dominic
# small modifications
metrics = ["accuracy", "precision", "f1"] #List of metrics to return
metric={}
for met in metrics:
  metric[met] = load(met)

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = logits.argmax(axis=-1)
  metric_res={}
  if met in ["precision", "f1"]:
    metric_res[met] = metric[met].compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )[met]
  else:
    # accuracy
    metric_res[met] = metric[met].compute(
        predictions=predictions,
        references=labels
    )[met]
  return metric_res

# Source - https://stackoverflow.com/a
# Posted by sid8491, modified by community. See post 'Timeline' for change history
# Modified a bit with updated function and 
class CustomTrainer(Trainer):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.i = 0
      self.logging_steps = 5

  # need kwargs because now they pass in num_items_in_batch
  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
      
      labels = inputs.get('labels')
      outputs = model(**inputs) # keyword 

      # compute accuracy
      if labels is not None and self.i != 0 and self.i % self.logging_steps == 0:
        logits = outputs.logits.detach()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item() # mean accuracy across batch
        self.log({'train_accuracy': acc})
      
      # compute loss
      if labels is not None:
        loss = self.label_smoother(outputs, labels)
      else:
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

      self.i += 1
      return (loss, outputs) if return_outputs else loss


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = CustomTrainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset_train,
  eval_dataset=tokenized_dataset_valid,
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(f"output/final-model-{finetune_type}-{seq_len}/")
results = trainer.evaluate()
print(results)