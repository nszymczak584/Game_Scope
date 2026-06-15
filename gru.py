import sqlite3
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import os
import pickle

torch.manual_seed(42)
np.random.seed(42)

MODEL_PATH = "gru_model.pth"
META_PATH = "gru_meta.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_str_cleaned(str_dirty: str) -> str:
    punctuation = '!?.,;:()[]{}"\''
    if not isinstance(str_dirty, str): return ""
    new_str = str_dirty.lower()
    new_str = re.sub(' +', ' ', new_str)
    for char in punctuation:
        new_str = new_str.replace(char, '')
    return new_str.strip()

def tokenize(text: str) -> list:
    return text.split()

def pad_sequence(sequence: list, max_len: int) -> list:
    if len(sequence) < max_len:
        sequence.extend([0] * (max_len - len(sequence)))
    else:
        sequence = sequence[:max_len]
    return sequence

def load_trained_model(model_path=MODEL_PATH, meta_path=META_PATH):
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None, None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    vocab_size = meta["vocab_size"]
    mlb = meta["mlb"]
    word_to_idx = meta["word_to_idx"]
    num_classes = len(mlb.classes_)

    model = GRUClassifier(vocab_size, 64, 64, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return model, mlb, word_to_idx


def predict_top_labels(text: str, model, mlb, word_to_idx, top_k: int = 5, threshold: float = 0.01):
    if model is None:
        return []

    cleaned = get_str_cleaned(text)
    encoded = [word_to_idx.get(token, word_to_idx.get("<UNK>", 1)) for token in tokenize(cleaned)]
    padded = pad_sequence(encoded, 100)

    input_tensor = torch.tensor([padded], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0]

    valid_indices = (probs >= threshold).nonzero(as_tuple=True)[0]
    
   
    if len(valid_indices) == 0:
        _, top_idx = torch.topk(probs, k=1)
        return [mlb.classes_[top_idx.item()]]

    valid_probs = probs[valid_indices]

    sorted_indices = valid_indices[torch.argsort(valid_probs, descending=True)]
    top_k = min(top_k, len(sorted_indices))
    
    return [mlb.classes_[i.item()] for i in sorted_indices[:top_k]]


def prepare_dataset(max_vocab_size=10000, max_len=100):
    print("Pobieranie danych z bazy boardgames.db...")
    with sqlite3.connect("boardgames.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT description, domains FROM boardgames WHERE description IS NOT NULL AND domains IS NOT NULL AND domains != 'Brak danych'")
        rows = cursor.fetchall()
        
    texts = [row[0] for row in rows]

    raw_labels = [[domain.strip().lower() for domain in row[1].split(',') if domain.strip()] for row in rows]
    
    clean_texts = [get_str_cleaned(t) for t in texts]
    tokenized_texts = [tokenize(t) for t in clean_texts]
    
    all_tokens = [word for sentence in tokenized_texts for word in sentence]
    most_common = Counter(all_tokens).most_common(max_vocab_size)
    
    vocab = ["<PAD>", "<UNK>"] + [word for word, count in most_common]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    encoded_texts = [[word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens] for tokens in tokenized_texts]
    padded_texts = [pad_sequence(seq, max_len) for seq in encoded_texts]
    
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(raw_labels)
    
    X = torch.tensor(padded_texts, dtype=torch.long)
    y = torch.tensor(encoded_labels, dtype=torch.float32)
    
    return X, y, len(vocab), mlb, word_to_idx



class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, gru_hidden = self.gru(embedded)
        return self.fc(gru_hidden[-1])




def main():
    model, mlb, word_to_idx = load_trained_model()

    if model is not None:
        print("Wczytano model z dysku, pomijam trening.")
    else:
        X_all, y_all, vocab_size, mlb, word_to_idx = prepare_dataset()
        num_classes = y_all.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        
        train_loader = DataLoader(TensorDataset(X_train.to(DEVICE), y_train.to(DEVICE)), batch_size=32, shuffle=True)

        model = GRUClassifier(vocab_size, 64, 64, num_classes).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        print(f"\nRozpoczynamy trening (15 epok) na urządzeniu: {DEVICE}...")
        for epoch in range(15):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoka {epoch+1}/15 zakończona")

        torch.save(model.state_dict(), MODEL_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump({"vocab_size": vocab_size, "mlb": mlb, "word_to_idx": word_to_idx}, f)
        print(f"Zapisano model: {MODEL_PATH} i metadane: {META_PATH}")

    # TEST
    test_desc = "Legend has it that eight mighty Dragons once rose from the mountains as darkness swept across Sinistra and Dextra."
    results = predict_top_labels(test_desc, model, mlb, word_to_idx, top_k=3, threshold=0.2)

    print(f"\n--- TEST ---")
    print(f"Predykcje GRU: {results if results else 'brak motywów'}")

if __name__ == "__main__":
    main()