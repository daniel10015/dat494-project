import kagglehub
from kagglehub import KaggleDatasetAdapter

import os
import pandas as pd

verbose = True
def print_verbose(*args):
  if verbose:
    print(*args)

dataset_dir = 'dataset/'
train_filename = f'{dataset_dir}QnA.csv'
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

# get train dataset
if cleaned_train_filename:
  cleaned_df_train = pd.read_csv(cleaned_train_filename)
if cleaned_valid_filename:
  cleaned_df_valid = pd.read_csv(cleaned_valid_filename)

if dataAlreadyLoaded:
  df_train = pd.read_csv(train_filename)
else:
  # Set the path to the file you'd like to load
  file_path = "train.csv"
  # Load the latest version
  df_train = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "imoore/60k-stack-overflow-questions-with-quality-rate",
    file_path,
  )
  os.makedirs(dataset_dir)
  df_train.to_csv(train_filename, index=False)

if not cleaned_train_filename:
  labels = df_train['Y']

  cleaned_df_train = pd.DataFrame()

  cleaned_df_train["Body"] = df_train["Body"].apply(clean_html)
  #cleaned_df_train["Title"] = df_train["Title"].apply(clean_html)
  cleaned_df_train["Y"] = df_train["Y"]
  cleaned_df_train.to_csv('dataset/cleaned_QnA.csv')

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
def preprocess_and_save(max_len: int, path: str):
  print(f'Preprocessing Stackoverflow dataset with maxlen={max_len}')

  df_train = pd.DataFrame()
  df_train['text'] = cleaned_df_train['Body'].str.replace("\n", " ", regex=False).astype(str)
  df_train['label'] = cleaned_df_train['Y'].map(label_map).astype(int)
  hf_dataset = Dataset.from_pandas(df_train)

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
  
def load_stackoverflow_dataset(max_len: int):
  path = f'stackoverflow_tokenized_length{max_len}'

  if os.path.exists(path):
    print('Loading from cached file')
    return load_from_disk(path)
  
  return preprocess_and_save(max_len=max_len, path=path)


# Look at the first datapoint
print(load_stackoverflow_dataset(max_len=128)[0])

# Freeze all layers except the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# Keep only the classification head trainable
for param in model.classifier.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

training_args = TrainingArguments(
  output_dir = 'output/',
  learning_rate=1e-4,
  per_device_eval_batch_size=16,
  per_device_train_batch_size=16,
  num_train_epochs=1,
  save_total_limit=2,
  load_best_model_at_end=True,
  logging_dir='log/'
  logging_steps=100,
)

from evaluate import load

metric = load('f1')

def compute_metrics(logits, labels):
  predictions = logits.argmax(axis=-1)
  return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
  model=model,
  args=training_args,

)