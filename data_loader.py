import csv
import sqlite3
import os

def safe_int(value, default=0):
    if not value or value.strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default

# Funkcja przyjmuje teraz pełną ścieżkę bezwzględną
def load_binary_csv(filepath):
    print(f"Przetwarzam plik: {filepath}...")
    feature_dict = {}
    
    if not os.path.exists(filepath):
        print(f"Ostrzeżenie: Nie znaleziono pliku {filepath}. Te dane zostaną pominięte.")
        return feature_dict

    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        for row in reader:
            if not row:
                continue
            
            game_id = row[0]
            active_features = []
            
            for i in range(1, len(row)):
                if row[i] == '1' or row[i] == '1.0':
                    active_features.append(headers[i])
            
            feature_dict[game_id] = ", ".join(active_features) if active_features else "Brak danych"
            
    return feature_dict

def build_full_database():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    games_path = os.path.join(script_dir, "data", "games.csv")
    mechanics_path = os.path.join(script_dir, "data", "mechanics.csv")
    themes_path = os.path.join(script_dir, "data", "themes.csv")
    db_path = os.path.join(script_dir, "boardgames.db")

    print(f"Lokalizacja skryptu: {script_dir}")
    print(f"Szukam plików w: {os.path.join(script_dir, 'data')}\n")
    
    # 1. Wczytujemy mechaniki i motywy przy użyciu pełnych ścieżek
    mechanics_lookup = load_binary_csv(mechanics_path)
    themes_lookup = load_binary_csv(themes_path)
    
    # 2. Tworzymy / otwieramy bazę danych SQLite w tym samym folderze co skrypt
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS boardgames (
        id INTEGER PRIMARY KEY,
        title TEXT,
        year INTEGER,
        description TEXT,
        minplayers INTEGER,
        maxplayers INTEGER,
        minplaytime INTEGER,
        maxplaytime INTEGER,
        age INTEGER,
        mechanics TEXT,
        domains TEXT
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON boardgames(title)")
    
    # 3. Czytamy główny plik games.csv
    print(f"Otwieram główny plik: {games_path}")
    
    if not os.path.exists(games_path):
        print(f"\n[BŁĄD DALEJ WYSTĘPUJE!]")
        print(f"Python szukał pliku tutaj: {games_path}")
        print("Upewnij się, czy plik na pewno nazywa się dokładnie 'games.csv' a nie np. 'games.csv.zip'.")
        conn.close()
        return

    with open(games_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        games_to_insert = []
        
        for row in reader:
            game_id_str = row.get('BGGId', '').strip()
            game_id = safe_int(game_id_str)
            
            if not game_id:
                continue
            
            title = row.get('Name', 'Brak tytułu')
            year = safe_int(row.get('YearPublished'), None)
            description = row.get('Description', 'Brak opisu')
            minplayers = safe_int(row.get('MinPlayers'), None)
            maxplayers = safe_int(row.get('MaxPlayers'), None)
            minplaytime = safe_int(row.get('MinPlaytime') or row.get('ComMinPlaytime'), 0)
            maxplaytime = safe_int(row.get('MaxPlaytime') or row.get('ComMaxPlaytime'), 0)
            age = safe_int(row.get('MinAge'), 0)
            
            mechanics = mechanics_lookup.get(game_id_str, "Brak danych")
            domains = themes_lookup.get(game_id_str, "Brak danych")
            
            games_to_insert.append((
                game_id, title, year, description, 
                minplayers, maxplayers, minplaytime, maxplaytime, 
                age, mechanics, domains
            ))
            
            if len(games_to_insert) >= 2000:
                cursor.executemany("""
                INSERT OR REPLACE INTO boardgames (
                    id, title, year, description, minplayers, maxplayers, minplaytime, maxplaytime, age, mechanics, domains
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, games_to_insert)
                conn.commit()
                print(f"Zaimportowano kolejną paczkę danych...")
                games_to_insert = []
                
        if games_to_insert:
            cursor.executemany("""
            INSERT OR REPLACE INTO boardgames (
                id, title, year, description, minplayers, maxplayers, minplaytime, maxplaytime, age, mechanics, domains
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, games_to_insert)
            conn.commit()
            
    conn.close()
    print(f"\nSukces! Pełna baza danych została stworzona w: {db_path}")

if __name__ == "__main__":
    build_full_database()