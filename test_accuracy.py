import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from gru import load_trained_model, prepare_dataset, DEVICE

def evaluate_model(threshold=0.5):
    print("Wczytywanie zapisanego modelu...")
    model, mlb, word_to_idx = load_trained_model()
    
    if model is None:
        print("Błąd: Nie znaleziono modelu lub pliku meta. Najpierw uruchom główny skrypt, aby wytrenować model.")
        return

    X_all, y_all, vocab_size, _, _ = prepare_dataset()
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    test_loader = DataLoader(
        TensorDataset(X_test.to(DEVICE), y_test.to(DEVICE)), 
        batch_size=32, 
        shuffle=False
    )

    print(f"Uruchamianie testów na {len(y_test)} próbkach...")
    model.eval()
    
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch_y.cpu().numpy())

    # poniżej metryki
    exact_match_acc = accuracy_score(all_trues, all_preds)
    f1_micro = f1_score(all_trues, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_trues, all_preds, average='macro', zero_division=0)

    print("\n" + "="*40)
    print("WYNIKI EWALUACJI MODELU GRU")
    print("="*40)
    print(f"Próg decyzyjny (threshold): {threshold}")
    print(f"Exact Match Accuracy:       {exact_match_acc:.4f}")
    print(f"F1-Score (Micro):           {f1_micro:.4f}")
    print(f"F1-Score (Macro):           {f1_macro:.4f}")
    print("="*40)

    report_dict = classification_report(all_trues, all_preds, target_names=mlb.classes_, zero_division=0, output_dict=True)
    
    class_metrics = []
    for label, metrics in report_dict.items():
        if label not in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            class_metrics.append({
                'name': label,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1-score'],
                'support': int(metrics['support'])
            })

    # malejąco
    class_metrics_sorted = sorted(class_metrics, key=lambda x: x['f1'], reverse=True)

    print("\nSzczegółowy raport dla każdej z klas (posortowany malejąco po F1-score):")
    print(f"{'Motyw (Domena)':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'Support'}")
    print("-" * 75)
    
    for item in class_metrics_sorted:
        print(f"{item['name']:<25} | {item['precision']:<10.4f} | {item['recall']:<10.4f} | {item['f1']:<10.4f} | {item['support']}")

if __name__ == "__main__":
    evaluate_model(threshold=0.2)