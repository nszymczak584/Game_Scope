import spacy
import re
import string


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



nlp = spacy.load("en_core_web_lg")

player_keywords = {"people", "guy", "friend", "player", "mate", "person", "participant", "individual", "enthusiast",
                   "gamer"}


def extract_game_info(text):
    text = text.replace("\n", " ").strip()
    text = words_to_numbers(text)

    doc = nlp(text)

    year = None
    game_time = None
    age = None
    num_players = None

    for ent in doc.ents:
        if ent.label_ == "DATE":
            if re.search(r"\b(19|20)\d{2}\b", ent.text):
                year = ent.text

                if ent.start > 0:
                    prev_token = doc[ent.start - 1].lemma_.lower()
                    if prev_token == "before":
                        year += "-"
                    elif prev_token == "after":
                        year += "+"

            elif "year" in ent.text.lower() and "old" in ent.text.lower():
                age_match = re.search(r"\d+", ent.text)
                if age_match:
                    age = age_match.group()

        elif ent.label_ == "TIME":
            game_time = ent.text

    if not age:
        lower_text = text.lower()
        age_match = re.search(r"\b(?:age|aged|ages)\s*(\d+)", lower_text)
        if age_match:
            age = age_match.group(1)
        elif "young" in lower_text:
            age = "young"


    return {
        "year": year,
        "game_time": game_time,
        "age": age,
        "num_players": num_players
    }

print(extract_game_info("I want a game for 4 people, around 30 minutes, suitable for ages 10 and up, released after 2015."))