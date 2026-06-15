import sqlite3
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "boardgames.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM boardgames")
total_games = cursor.fetchone()[0]
print(f"Całkowita liczba gier w bazie: {total_games}\n")

cursor.execute("SELECT * FROM boardgames LIMIT 5")
rows = cursor.fetchall()

print("Przykładowe gry w bazie:")
for row in rows:
    print(f"ID: {row[0]}, Tytuł: {row[1]}, Rok: {row[2]}, Mechaniki: {row[9]}")

conn.close()