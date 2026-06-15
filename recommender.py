import sqlite3
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from input_interpreter import extract_game_info
from utils import get_vector
from sense import s2v
from gru import predict_top_labels

def build_sql_and_params(extracted_info):
    query = "SELECT id, title, description, description_vector, quality_score, domains FROM boardgames WHERE 1=1"
    params = []
    
    num_players = extracted_info.get("num_players")
    if num_players:
        query += " AND minplayers <= ? AND maxplayers >= ?"
        params.extend([num_players, num_players])
        
    age = extracted_info.get("age")
    if age and str(age).isdigit():
        query += " AND age <= ?"
        params.append(int(age))
        
    game_time = extracted_info.get("game_time")
    if game_time:
        time_match = re.search(r'\d+', str(game_time))
        if time_match:
            time_int = int(time_match.group())
            query += " AND minplaytime <= ? AND maxplaytime >= ?"
            params.extend([time_int + 30, max(0, time_int - 30)])
            
    return query, params


def recommend_best_games(user_query, s2v_model, gru_model, mlb, word_to_idx, db_path="boardgames.db", top_n=5):
    extracted_info = extract_game_info(user_query)
    sql_query, params = build_sql_and_params(extracted_info)
    
    filtered_games = []
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query, params)
        rows = cursor.fetchall()
        

        for game_id, title, description, vector_blob, quality_score, domains in rows:
            if vector_blob:
                vector = np.frombuffer(vector_blob, dtype=np.float32)
                
                #jeśli gra nie ma oceny w bazie (jest NULL), dajemy jej przeciętne 5.0
                if quality_score is None:
                    quality_score = 5.0

                filtered_games.append((game_id, title, description, vector, quality_score, domains or ""))
                
    if not filtered_games:
        return "Brak gier spełniających twarde kryteria liczbowe."
        
    # GRU wytypowuje kategorie
    predicted_domains = predict_top_labels(user_query, gru_model, mlb, word_to_idx, top_k=3, threshold=0.000001)
    target_domains = {domain.strip().lower() for domain in predicted_domains if domain.strip()} if predicted_domains else set()

    query_vector = get_vector(user_query, s2v_model)
    if query_vector is None:
        return "System nie był w stanie stworzyć wektora dla podanego zapytania."
        
    recommendations = []
    
    for game_id, title, description, desc_vector, quality_score, domains in filtered_games:
        
        #Obliczamy podobieństwo tekstu z Sense2Vec
        similarity = cosine_similarity([query_vector], [desc_vector])[0][0]
        
        #Sprawdzamy, czy kategoria gry pasuje do tego, co odgadło GRU
        domain_score = 0.0
        game_domains = {d.strip().lower() for d in domains.split(',') if d.strip()}
        if target_domains and (game_domains & target_domains):
            domain_score = 1.0 # Dajemy pełen punkt bonusowy
            
        #Normalizujemy jakość
        normalized_quality = quality_score / 10.0  
        
        # 60% wagi dla podobieństwa NLP, 30% dla jakości, 10% dla dopasowania kategorii (jeśli GRU coś przewidziało) 
        nlp_weight = 0.60
        quality_weight = 0.30
        domain_weight = 0.10
        
        # Ostateczny wzór uwzględniający wszystkie 3 sztuczne inteligencje/statystyki
        final_score = (similarity * nlp_weight) + (normalized_quality * quality_weight) + (domain_score * domain_weight)
        
        # Dodajemy wszystko do listy
        recommendations.append({
            "title": title,
            "final_score": final_score,
            "similarity": similarity,
            "quality": quality_score,
            "description": description
        })
        
    # Sortujemy malejąco
    recommendations.sort(key=lambda x: x["final_score"], reverse=True)
    
    return recommendations[:top_n]

if __name__ == "__main__":
    print("Uruchamianie testowego zapytania do systemu rekomendacji")
    
    test_query = "I want a cooperative game for 4 players, about fighting monsters in a dungeon, around 60 minutes."
    print(f"Wprowadzone zapytanie: '{test_query}'\n")
    
    print("Inicjalizacja i uruchamianie algorytmu")
    wyniki = recommend_best_games(test_query, s2v, db_path="boardgames.db", top_n=5)
    
    print("\ntop 5 rekomendacji:")
    if isinstance(wyniki, list):
        for i, game in enumerate(wyniki, 1):
            print(f"{i}. [{game['final_score']:.4f}] - {game['title']}")
    else:
        print(wyniki)