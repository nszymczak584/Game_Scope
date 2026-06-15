import sqlite3
import numpy as np
from sense2vec import Sense2Vec
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_vector
from gru import predict_top_labels, load_trained_model

s2v = Sense2Vec().from_disk("data/s2v_old")


def load_precomputed_games():

    with sqlite3.connect("boardgames.db") as conn:

        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description, description_vector, domains FROM boardgames")
        rows = cursor.fetchall()

    processed_games = []
    for game_id, title, description, vector_blob, domains in rows:
        if vector_blob:

            vector = np.frombuffer(vector_blob, dtype=np.float32)
            processed_games.append((game_id, title, description, vector, domains or ""))

    return processed_games

def find_similar_games(query_desc, loaded_games, s2v_model, top_n=3):

    query_vector = get_vector(query_desc, s2v_model)

    if query_vector is None:
        return []

    similarities = []
    for game_id, title, description, desc_vector, _domains in loaded_games:
        similarity = cosine_similarity([query_vector], [desc_vector])[0][0]
        similarities.append((game_id, title, description, similarity))

    return sorted(similarities, key=lambda x: x[3], reverse=True)[:top_n]

def filter_games_by_domains(loaded_games, target_domains):
    target = {domain.strip().lower() for domain in target_domains if domain.strip()}

    if not target:
        return loaded_games

    filtered_games = []
    for game_id, title, description, vector, domains in loaded_games:
        game_domains = {domain.strip().lower() for domain in domains.split(',') if domain.strip()}
        if game_domains & target:
            filtered_games.append((game_id, title, description, vector, domains))
    return filtered_games

def main():

    print("loading data...")
    cached_games = load_precomputed_games()
    print(f"found {len(cached_games)} games")
    gru_model, mlb, word_to_idx = load_trained_model()

    query = "A game about sewing a dress or tailoring and materials."
    print(f"\nZapytanie: '{query}'")

    predicted_domains = predict_top_labels(query, gru_model, mlb, word_to_idx, top_k=5, threshold=0.000001)
    print(f"GRU wytypowało kategorie: {predicted_domains if predicted_domains else 'Brak pewnych predykcji'}")

    filtered_games = filter_games_by_domains(cached_games, predicted_domains)

    if len(filtered_games) == 0:
        filtered_games = cached_games

    top_games = find_similar_games(query, filtered_games, s2v, top_n=3)
    print("\nTop Recommendations:")
    for game in top_games:
        print(f"[{game[3]:.4f}] {game[1]}")

if __name__ == "__main__":
    main()