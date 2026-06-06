import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")


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