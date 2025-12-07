import torch 
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT
max_length = 128
finetune_type = 'last_layer'
model_path = f'output/final-model-{finetune_type}-{max_length}/'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device('cuda')
model.to(device)
model.eval()

example = "How come we say division by 0 is undefined?"
label_map = {'HQ': 0, 'LQ_EDIT' : 1, 'LQ_CLOSE': 2}
label_map_reverse = {0: 'HQ', 1: 'LQ_EDIT', 2: 'LQ_CLOSE'}
label_decode = {'HQ': 'High Quality', 'LQ_EDIT': "Okay but is salvagable an edit", "LQ_CLOSE": "Very bad and can't be salvaged"}


# tokenize and run
encoding = tokenizer(
  example,
  return_tensors='pt',
  truncation = True,
  padding = True,
  max_length=max_length
).to(device)

with torch.no_grad():
  outputs = model(**encoding)
  logits = outputs.logits
  probs = torch.softmax(logits, dim=-1)
  pred = probs.argmax(dim=-1).item()
  cls = label_map_reverse[pred]
print(f'input: {example}')
print(f'predicted class: {cls}')
print(f'==verdict== {label_decode.get(cls)}')
print(f'probabilities: {probs.cpu().numpy()}')
