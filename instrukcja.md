# GameScope - instrukcja uruchomienia

Ta instrukcja prowadzi od zera do uruchomienia aplikacji `GameScope`.

## 1) Wymagania

- Python 3.10+ (zalecane środowisko wirtualne)(testowane na 3.13)
- Konto Kaggle (do pobrania datasetu)

## 2) Utwórz środowisko wirtualne (venv)


## 3) Utwórz folder `data`



## 4) Zainstaluj zależności Pythona

```powershell
pip install -r requirements.txt
```

## 5) Zainstaluj korpus/model `large` do spaCy

Projekt używa modelu `en_core_web_lg`.

```powershell
python -m spacy download en_core_web_lg
```

## 6) Pobierz i wypakuj `s2v_reddit_2015_md.tar`

Kod oczekuje modelu Sense2Vec w katalogu: `data/s2v_old`.

https://github.com/explosion/sense2vec

1. Pobierz archiwum `s2v_reddit_2015_md.tar` (lub `.tar.gz`) i zapisz je w katalogu projektu.
2. Wypakuj je do folderu `data`.
3. Zmień nazwę wypakowanego katalogu na `s2v_old`.


Po tym kroku powinny istnieć pliki podobne do:
- `data/s2v_old/cfg`
- `data/s2v_old/vectors`
- `data/s2v_old/strings.json`

## 7) Pobierz dataset z Kaggle (Board Games Geek)

https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek

Wymagane są pliki CSV z danymi BGG (m.in. `games.csv`, `mechanics.csv`, `themes.csv`, `ratings_distribution.csv`).



### 7a) Pobierz i wypakuj dataset
1. Pobierz dataset z Kaggle (może być w formacie `.zip`).
2. Wypakuj go do katalogu `data/` w projekcie.

Następnie upewnij się, że w `data/` znajdują się co najmniej:
- `games.csv`
- `mechanics.csv`
- `themes.csv`
- `ratings_distribution.csv`

## 8) Zbuduj lokalną bazę SQLite i wektory

```powershell
python .\data_loader.py
python .\apply_csv_ratings.py
python .\vector_preparation.py
```

To utworzy/uzupełni plik `boardgames.db`.

## 9) Uruchom aplikację

```powershell
streamlit run .\app.py
```

Po uruchomieniu otwórz adres pokazany przez Streamlit (zwykle `http://localhost:8501`).

---

## Szybka diagnostyka

- Błąd `Can't find model 'en_core_web_lg'`:
  - uruchom ponownie `python -m spacy download en_core_web_lg`
- Błąd ładowania Sense2Vec:
  - sprawdź, czy ścieżka `data/s2v_old` istnieje i zawiera pliki modelu
- Brak danych w rekomendacjach:
  - sprawdź, czy `boardgames.db` został utworzony oraz czy w `data/` są wymagane CSV

