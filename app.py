import streamlit as st
from sense2vec import Sense2Vec
from recommender import recommend_best_games
from gru import load_trained_model

st.set_page_config(
    page_title="GameScope - Wyszukiwarka",
    page_icon="🎲",
    layout="centered"
)

# cachowanie modelu sense2Vec
@st.cache_resource
def load_ai_models():
    s2v = Sense2Vec().from_disk("data/s2v_old")
    gru_model, mlb, word_to_idx = load_trained_model()
    return s2v, gru_model, mlb, word_to_idx

with st.spinner("Loading AI models (Sense2Vec & GRU)..."):
    s2v, gru_model, mlb, word_to_idx = load_ai_models()

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
            
   
            wyniki = recommend_best_games(user_query, s2v, gru_model, mlb, word_to_idx, top_n=top_n_results)
            
            st.divider() 
            
            if isinstance(wyniki, str):
                st.error(wyniki)
            elif isinstance(wyniki, list) and wyniki:
                st.success("We found the perfect matches!")
                
                for i, game in enumerate(wyniki, 1):
                    with st.container(border=True):
                        st.markdown(f"### {i}. {game['title']}")
                        st.write(f"⭐ **Quality:** {game['quality']}/10")
                        st.write(f"🧠 **Query Similarity:** {game['similarity']:.2f}")
                        
                        with st.expander("📝 Read full description"):
                            st.write(game['description'])
            else:
                st.info("No games found matching your criteria.")