import streamlit as st
from sense2vec import Sense2Vec
from recommender import recommend_best_games

# Konfiguracja wyglądu strony
st.set_page_config(
    page_title="GameScope - Wyszukiwarka",
    page_icon="🎲",
    layout="centered"
)

# cachowanie modelu Sense2Vec, aby nie ładować go za każdym razem
@st.cache_resource
def load_s2v_model():
    return Sense2Vec().from_disk("data/s2v_old")

# Wczytywanie modelu
with st.spinner("Loading Sense2Vec model..."):
    s2v = load_s2v_model()

st.title("🎲 GameScope")

with st.form(key="search_form"):
    user_query = st.text_input(
        "Your query:", 
        placeholder="I want a cooperative game for 4 players, around 60 minutes..."
    )
    
    top_n_results = st.slider("How many games to recommend?", min_value=1, max_value=10, value=5)
    
    submit_button = st.form_submit_button(label="Find Games")

if submit_button:
    if not user_query.strip():
        st.warning("Please enter a query to search for games.")
    else:
        with st.spinner("Searching for the best games..."):
            
            wyniki = recommend_best_games(user_query, s2v, top_n=top_n_results)
            
            st.divider() 
            
            if isinstance(wyniki, str):
                st.error(wyniki)
            elif isinstance(wyniki, list) and wyniki:
                st.success("We found the perfect matches!")
                
                # Ulepszone wyświetlanie wyników hybrydowych (z poprzedniego kroku)
                for i, game in enumerate(wyniki, 1):
                    with st.container(border=True):
                        st.markdown(f"### {i}. {game['title']}")
                        st.write(f"⭐ **Quality:** {game['quality']}/10")
                        st.write(f"🧠 **Query Similarity:** {game['similarity']:.2f}")
                        st.info(f"📝 **Description:** {game['description']}")
            else:
                st.info("No games found matching your criteria.")