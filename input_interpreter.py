import spacy
import re

from utils import words_to_numbers

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

    for token in doc:
        if token.text.isdigit():
            if token.head.lemma_.lower() in player_keywords:
                num_players = int(token.text)
                break

            found_player = False
            for child in token.head.children:
                if child.lemma_.lower() in player_keywords:
                    num_players = int(token.text)
                    found_player = True
                    break

            if found_player:
                break

    if not num_players:
        player_match = re.search(r"(group of) (\d+)", text.lower())
        if player_match:
            num_players = int(player_match.group(2))

    return {
        "year": year,
        "game_time": game_time,
        "age": age,
        "num_players": num_players
    }

print(extract_game_info("I want a game for 4 people, around 30 minutes, suitable for ages 10 and up, released after 2015."))