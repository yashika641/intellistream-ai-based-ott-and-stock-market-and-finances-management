import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load saved model components (adjust path accordingly)
@st.cache_resource
def load_model_components():
    with open("saved_model/recommender_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    from keras.models import load_model
    fnn_model = load_model("saved_model/fnn_model.h5")
    return metadata, fnn_model

metadata, fnn_model = load_model_components()
df = metadata['dataframe']
cosine_sim_matrix = metadata['cosine_similarity_matrix']

st.set_page_config(page_title="OTT Movie Recommender", layout="wide")

st.title("🎬 OTT Movie Recommender System")

# Sidebar user selection
st.sidebar.header("User Selection")
user_ids = df['user_id'].unique()
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

# Filter movies watched by the user
user_movies = df[df['user_id'] == selected_user]

st.sidebar.markdown(f"**Movies watched:** {user_movies['title'].nunique()}")

def recommend_for_user(user_id, top_n=10):
    # Find user vector (user embedding input)
    user_idx = df[df['user_id'] == user_id].index[0]  # or any user indexing logic
    
    # Content-based recommendation (FNN predictions)
    user_movie_ids = df[df['user_id'] == user_id]['movie_id'].unique()
    all_movie_ids = df['movie_id'].unique()

    # Predict watch times for all movies for this user
    user_input = np.array([user_id] * len(all_movie_ids))
    movie_input = all_movie_ids

    preds = fnn_model.predict([user_input, movie_input], verbose=0).flatten()
    
    recommendations_idx = np.argsort(preds)[::-1]
    recommended_movie_ids = all_movie_ids[recommendations_idx][:top_n]
    
    # Get movie titles and metadata
    rec_movies_df = df[df['movie_id'].isin(recommended_movie_ids)].drop_duplicates(subset=['movie_id'])
    rec_movies_df = rec_movies_df[['movie_id', 'title', 'genre']].head(top_n)
    
    return rec_movies_df

st.subheader(f"Recommendations for User ID: {selected_user}")

recommendations = recommend_for_user(selected_user, top_n=10)

cols = st.columns(2)
for idx, (_, row) in enumerate(recommendations.iterrows()):
    with cols[idx % 2]:
        st.markdown(
            f"""
            <div style='background:#fafafa;padding:15px;margin:10px 0;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.1);'>
                <h4>{row['title']}</h4>
                <p><b>Genre:</b> {row['genre']}</p>
                <p><i>Movie ID: {row['movie_id']}</i></p>
            </div>
            """, unsafe_allow_html=True
        )

st.markdown("---")

# Search box to find movies info
st.subheader("Search for a movie")

search_term = st.text_input("Enter movie title keyword")

if search_term:
    filtered = df[df['title'].str.contains(search_term, case=False, na=False)].drop_duplicates(subset=['movie_id'])
    if filtered.empty:
        st.warning("No movies found with that title.")
    else:
        for _, row in filtered.iterrows():
            st.markdown(
                f"""
                <div style='background:#e3f2fd;padding:10px;margin-bottom:10px;border-radius:8px;'>
                    <h5>{row['title']}</h5>
                    <p><b>Genre:</b> {row['genre']}</p>
                    <p><b>User Watched:</b> {row['user_name'] if 'user_name' in row else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True
            )
else:
    st.info("Search for movies by title.")

# Footer with info
st.markdown(
    """
    <div style='text-align:center; margin-top:50px; color:gray;'>
        Built with ❤️ using Streamlit | Powered by your pretrained recommender model
    </div>
    """, unsafe_allow_html=True
)
