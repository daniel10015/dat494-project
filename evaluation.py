from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)