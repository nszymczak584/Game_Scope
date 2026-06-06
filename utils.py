import string

import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")

tens = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

word_to_num = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
    **tens
}

def words_to_numbers(text):
    result = []
    words = text.split()
    i = 0

    while i < len(words):
        clean_word = words[i].strip(string.punctuation).lower()

        if clean_word in tens and i + 1 < len(words):
            next_word = words[i + 1].strip(string.punctuation).lower()
            if next_word in word_to_num:
                number = tens[clean_word] + word_to_num[next_word]
                result.append(str(number))
                i += 2
                continue

        if clean_word in word_to_num:
            result.append(str(word_to_num[clean_word]))

        elif "-" in clean_word:
            parts = clean_word.split("-")
            if len(parts) == 2 and parts[0] in tens and parts[1] in word_to_num:
                number = tens[parts[0]] + word_to_num[parts[1]]
                result.append(str(number))
            else:
                result.append(words[i])

        else:
            result.append(words[i])
        i += 1

    return " ".join(result)

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