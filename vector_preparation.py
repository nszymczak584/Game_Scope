import sqlite3
import spacy
import numpy as np
from sense2vec import Sense2Vec

nlp = spacy.load("en_core_web_lg")
s2v = Sense2Vec().from_disk("data/s2v_old")


def setup_database():

    with sqlite3.connect("boardgames.db") as conn:
        cursor = conn.cursor()

        try:

            cursor.execute("ALTER TABLE boardgames ADD COLUMN description_vector BLOB")
            conn.commit()
            print("'boardgames' table updated with 'description_vector' column")

        except sqlite3.OperationalError:

            print("'description_vector' column already exists or some other error")


def get_vector(text, s2v_model):

    doc = nlp(text.lower())
    vectors = []

    for token in doc:

        if token.is_punct or token.is_space:
            continue
        key = f"{token.text}|{token.pos_}"

        if key in s2v_model:
            vectors.append(s2v_model[key])

    if vectors:

        return np.mean(vectors, axis=0).astype(np.float32)

    return None


def precompute_and_store_vectors():

    with sqlite3.connect("boardgames.db") as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id, description FROM boardgames WHERE description_vector IS NULL")
        games = cursor.fetchall()

        print(f"processing {len(games)} games...")

        for game_id, description in games:
            if not description:
                continue

            vector = get_vector(description, s2v)

            if vector is not None:
                vector_blob = vector.tobytes()

                cursor.execute(
                    "UPDATE boardgames SET description_vector = ? WHERE id = ?",
                    (vector_blob, game_id)
                )

        conn.commit()
        print("precomputation completed")


if __name__ == "__main__":
    setup_database()
    precompute_and_store_vectors()