import sqlite3
import pandas as pd
import os

def apply_distribution_ratings(csv_path="data/ratings_distribution.csv", db_path="boardgames.db"):
    if not os.path.exists(csv_path):
        print(f"Błąd: Nie znaleziono pliku '{csv_path}'")
        return

    print(f"Wczytywanie danych z {csv_path}")
    df = pd.read_csv(csv_path)

    initial_len = len(df)
    df = df.dropna(subset=['BGGId'])
    df = df[df['total_ratings'] > 0]
    print(f"Odrzucono {initial_len - len(df)} gier bez oddanych głosów.")

    print("Obliczanie średniej ważonej")
    
    rating_columns = [col for col in df.columns if col.replace('.', '', 1).isdigit() and col != 'BGGId']
    
    weighted_sum = pd.Series(0.0, index=df.index)
    
    for col in rating_columns:
        rating_value = float(col)
        weighted_sum += rating_value * df[col].fillna(0)
        

    df['quality_score'] = (weighted_sum / df['total_ratings']).round(2)

    print(f"Wyliczono jakość dla {len(df)} unikalnych gier.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE boardgames ADD COLUMN quality_score REAL")
        print("Dodano nową kolumnę 'quality_score' do bazy danych.")
    except sqlite3.OperationalError:
        print("Kolumna 'quality_score' już istnieje w bazie. Nadpisuję dane")

    print("Zapisywanie obliczonych ocen do bazy danych")
    

    updates = []
    for _, row in df.iterrows():
        updates.append((row['quality_score'], int(row['BGGId'])))

    cursor.executemany("UPDATE boardgames SET quality_score = ? WHERE id = ?", updates)
    conn.commit()
    conn.close()

    print("Wszystkie gry w bazie otrzymały ocenę na podstawie rozkładu głosów.")

if __name__ == "__main__":
    apply_distribution_ratings()