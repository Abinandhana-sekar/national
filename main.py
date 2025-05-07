# main.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess dataset
def load_data():
    df = pd.read_csv("generated_nic_pairs.csv")
    return df

# Define model class
class SiameseBERT(torch.nn.Module):
    def __init__(self):
        super(SiameseBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = torch.nn.Linear(768, 256)
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output_1 = self.bert(input_ids_1, attention_mask=attention_mask_1).pooler_output
        output_2 = self.bert(input_ids_2, attention_mask=attention_mask_2).pooler_output
        embedding_1 = self.fc(output_1)
        embedding_2 = self.fc(output_2)
        return self.cos(embedding_1, embedding_2)

# Run the app
if __name__ == "__main__":
    print("📦 Loading data...")
    df = load_data()

    print("🧠 Loading model...")
    model = SiameseBERT().cuda()
    model.load_state_dict(torch.load("siamese_nic_bert.pth"))
    model.eval()

    print("✅ Model ready for inference.")
    
    # Example test
    test_samples = [
        ("NIC code 0111 Agriculture", "Organic farming"),
        ("NIC code 6201 Software", "Cybersecurity solutions"),
    ]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for text1, text2 in test_samples:
        encoded_1 = tokenizer(text1, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        encoded_2 = tokenizer(text2, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        for key in encoded_1:
            encoded_1[key] = encoded_1[key].cuda()
            encoded_2[key] = encoded_2[key].cuda()

        with torch.no_grad():
            sim = model(encoded_1["input_ids"], encoded_1["attention_mask"],
                        encoded_2["input_ids"], encoded_2["attention_mask"])

        print(f"Similarity({text1} 🆚 {text2}) → {sim.item():.4f}")
