import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

class LegalTextClassifier(nn.Module):
    def __init__(self, num_labels):
        super(LegalTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_output = self.dropout(pooled_output)
        return self.classifier(dropped_output)

class Model:
    def __init__(self):
        pass

    def train(X, y, batch_size=8, epochs=3, learning_rate=2e-5):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        dataset = LegalDataset(X, y_encoded, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = LegalTextClassifier(num_labels=len(label_encoder.classes_))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
        
        return model, tokenizer, label_encoder

