import sqlite3
import numpy as np
from sense2vec import Sense2Vec
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_vector

s2v = Sense2Vec().from_disk("data/s2v_old")


def load_precomputed_games():

    with sqlite3.connect("boardgames.db") as conn:

        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description, description_vector FROM boardgames")
        rows = cursor.fetchall()

    processed_games = []

    for game_id, title, description, vector_blob in rows:

        if vector_blob:

            vector = np.frombuffer(vector_blob, dtype=np.float32)
            processed_games.append((game_id, title, description, vector))

    return processed_games

def find_similar_games(query_desc, loaded_games, s2v_model, top_n=3):

    query_vector = get_vector(query_desc, s2v_model)

    if query_vector is None:
        return []

    similarities = []

    for game_id, title, description, desc_vector in loaded_games:

        similarity = cosine_similarity([query_vector], [desc_vector])[0][0]
        similarities.append((game_id, title, description, similarity))

    return sorted(similarities, key=lambda x: x[3], reverse=True)[:top_n]


def main():

    print("loading data...")
    cached_games = load_precomputed_games()
    print(f"found {len(cached_games)} games")

    query = "A game about building a medieval city with unique buildings."
    print("comparing...")
    top_games = find_similar_games(query, cached_games, s2v, top_n=3)

    print("\nTop Recommendations:")
    for game in top_games:
        print(f"[{game[3]:.4f}] {game[1]}")


if __name__ == "__main__":
    main()