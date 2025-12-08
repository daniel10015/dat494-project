from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT
max_length = 128
finetune_type = 'last_layer'
model_path = f'output/final-model-{finetune_type}-{max_length}/'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

size = 0
bert_size = 0
for param in model.bert.parameters():
  param.requires_grad = False
  size += 1
  bert_size += 1
# Keep only the classification head trainable
for param in model.classifier.parameters():
  param.requires_grad = True
  size += 1
print(f'model layer count: {size}, bert layer count: {bert_size}')