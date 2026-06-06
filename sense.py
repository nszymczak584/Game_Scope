import sqlite3
import numpy as np
from sense2vec import Sense2Vec

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


processed_games = load_precomputed_games()
print(len(processed_games))
print(processed_games[:5])  # test