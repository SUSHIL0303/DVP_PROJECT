# import streamlit as st
# import joblib
# import pandas as pd
# import os
# import scipy.sparse as sp
# from tmdb_utils import get_movie_details

# # =====================
# # Paths
# # =====================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# # =====================
# # OMDB API Key
# # =====================
# TMDB_API_KEY = "d16975117eda6b867b3206cac865b128"  # <-- replace with your OMDB key

# # =====================
# # Load Data
# # =====================
# movies = pd.read_csv(os.path.join(MODELS_DIR, "movies.csv"))

# # Load final_dataset and ensure movieId is a column
# final_dataset = pd.read_csv(os.path.join(MODELS_DIR, "final_dataset.csv"))
# if "movieId" not in final_dataset.columns:
#     final_dataset.reset_index(inplace=True)

# # Load csr_data
# csr_data = sp.load_npz(os.path.join(MODELS_DIR, "csr_data.npz"))
# print("CSR data shape before fix:", csr_data.shape)

# # =====================
# # Load Models & Vectorizer
# # =====================
# recommender_model = joblib.load(os.path.join(MODELS_DIR, "recommender_model.joblib"))
# sentiment_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))  # classifier
# tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

# # =====================
# # Fix feature mismatch
# # =====================
# expected_features = recommender_model._fit_X.shape[1]  # number of features KNN expects
# if csr_data.shape[1] < expected_features:
#     missing = expected_features - csr_data.shape[1]
#     print(f"Adding {missing} missing feature(s) to csr_data")
#     csr_data = sp.hstack([csr_data, sp.csr_matrix((csr_data.shape[0], missing))])
# print("CSR data shape after fix:", csr_data.shape)

# # =====================
# # Helper Functions
# # =====================
# def recommend_movies(movie_name, top_n=5):
#     """Use recommender model (KNN) to suggest similar movies with poster & plot info"""
#     try:
#         # Find movieId from title
#         movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
#         if len(movie_list) == 0:
#             return []

#         movie_id = movie_list.iloc[0]['movieId']

#         # Make sure movieId exists in training dataset
#         if movie_id not in final_dataset['movieId'].values:
#             return []

#         # Get row index for csr_data
#         movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index[0]

#         # Query recommender
#         distances, indices = recommender_model.kneighbors(
#             csr_data[movie_idx],
#             n_neighbors=top_n + 1
#         )

#         recommendations = []
#         for idx in indices.squeeze().tolist()[1:]:  # skip itself
#             rec_movie_id = final_dataset.iloc[idx]['movieId']
#             title = movies[movies['movieId'] == rec_movie_id]['title'].values[0]

#             # Fetch poster & plot from OMDB
#             plot, poster = get_movie_details(title, TMDB_API_KEY)
#             recommendations.append({
#                 "title": title,
#                 "plot": plot,
#                 "poster": poster
#             })

#         return recommendations

#     except Exception as e:
#         return []

# def analyze_sentiment(user_review):
#     """Use sentiment model to classify user review"""
#     cleaned = user_review.lower()
#     X = tfidf_vectorizer.transform([cleaned])
#     prediction = sentiment_model.predict(X)[0]
#     prob = sentiment_model.predict_proba(X)[0]
#     sentiment = "Positive 😀" if prediction == 1 else "Negative 😡"
#     confidence = round(max(prob) * 100, 2)
#     return sentiment, confidence

# # =====================
# # Streamlit UI
# # =====================
# st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

# st.title("🎬 Hybrid Movie Recommender + Sentiment Analyzer")

# tab1, tab2 = st.tabs(["🔎 Movie Recommendation", "💬 Sentiment Analysis"])

# # --- Movie Recommendation ---
# with tab1:
#     st.subheader("Find Similar Movies")
#     movie_input = st.text_input("Enter a movie name:")
#     if st.button("Recommend"):
#         if movie_input.strip():
#             results = recommend_movies(movie_input)
#             if not results:
#                 st.warning("No recommendations found.")
#             else:
#                 for i, movie in enumerate(results, start=1):
#                     st.write(f"### {i}. {movie['title']}")
#                     if movie['poster'] != "N/A":
#                         st.image(movie['poster'], width=200)
#                     st.write(movie['plot'])
#         else:
#             st.warning("Please enter a movie name.")

# # --- Sentiment Analysis ---
# with tab2:
#     st.subheader("Analyze Your Review")
#     review_input = st.text_area("Enter your movie review:")
#     if st.button("Analyze"):
#         if review_input.strip():
#             sentiment, confidence = analyze_sentiment(review_input)
#             st.success(f"Prediction: {sentiment} (Confidence: {confidence}%)")
#         else:
#             st.warning("Please enter a review text.")
import streamlit as st
import joblib
import pandas as pd
import os
import scipy.sparse as sp
from tmdb_utils import get_movie_details

# =====================
# Paths
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =====================
# TMDB API Key
# =====================
TMDB_API_KEY = "d16975117eda6b867b3206cac865b128"  # Replace with your key

# =====================
# Load MovieLens Data
# =====================
movies = pd.read_csv(os.path.join(MODELS_DIR, "movies.csv"))
final_dataset = pd.read_csv(os.path.join(MODELS_DIR, "final_dataset.csv"))
if "movieId" not in final_dataset.columns:
    final_dataset.reset_index(inplace=True)

csr_data = sp.load_npz(os.path.join(MODELS_DIR, "csr_data.npz"))

# =====================
# Load Models
# =====================
recommender_model = joblib.load(os.path.join(MODELS_DIR, "recommender_model.joblib"))
sentiment_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))
tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

# =====================
# Fix feature mismatch
# =====================
expected_features = recommender_model._fit_X.shape[1]
if csr_data.shape[1] < expected_features:
    missing = expected_features - csr_data.shape[1]
    csr_data = sp.hstack([csr_data, sp.csr_matrix((csr_data.shape[0], missing))])

# =====================
# Helper Functions
# =====================
def get_user_sentiment(review_text):
    """Compute sentiment score (0-1) from user review"""
    cleaned = review_text.lower()
    X = tfidf_vectorizer.transform([cleaned])
    pred = sentiment_model.predict(X)[0]  # 1=positive, 0=negative
    return float(pred)

def recommend_movies_hybrid(movie_name, user_review=None, top_n=5, alpha=0.7):
    """Hybrid recommendation: CF + user review sentiment"""
    try:
        # Find movie
        movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
        if len(movie_list) == 0:
            return []

        movie_id = movie_list.iloc[0]['movieId']
        if movie_id not in final_dataset['movieId'].values:
            return []

        movie_idx = final_dataset[final_dataset['movieId'] == movie_id].index[0]
        distances, indices = recommender_model.kneighbors(
            csr_data[movie_idx],
            n_neighbors=top_n*2
        )

        # Compute sentiment score from user review (0-1)
        user_sentiment = get_user_sentiment(user_review) if user_review else 0.5

        hybrid_list = []
        for idx, distance in zip(indices.squeeze().tolist()[1:], distances.squeeze().tolist()[1:]):
            rec_movie_id = final_dataset.iloc[idx]['movieId']
            title = movies[movies['movieId'] == rec_movie_id]['title'].values[0]

            # Fetch plot and poster
            plot, poster = get_movie_details(title, TMDB_API_KEY)

            # CF score (inverse distance)
            cf_score = 1 / (distance + 1e-5)

            # Hybrid score
            hybrid_score = alpha * cf_score + (1 - alpha) * user_sentiment

            hybrid_list.append({
                "title": title,
                "plot": plot,
                "poster": poster,
                "hybrid_score": hybrid_score
            })

        hybrid_list = sorted(hybrid_list, key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_list[:top_n]

    except Exception as e:
        return []

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender + Sentiment Analyzer")

tab1, tab2 = st.tabs(["🔎 Movie Recommendation", "💬 Sentiment Analysis"])

# --- Movie Recommendation ---
with tab1:
    st.subheader("Find Similar Movies (Hybrid CF + Sentiment)")
    movie_input = st.text_input("Enter a movie name:")
    review_input = st.text_area("Enter your review for this movie (optional):")

    if st.button("Recommend"):
        if movie_input.strip():
            results = recommend_movies_hybrid(movie_input, user_review=review_input)
            if not results:
                st.warning("No recommendations found.")
            else:
                for i, movie in enumerate(results, start=1):
                    st.write(f"### {i}. {movie['title']}")
                    if movie['poster'] != "N/A":
                        st.image(movie['poster'], width=200)
                    st.write(movie['plot'])
        else:
            st.warning("Please enter a movie name.")

# --- Sentiment Analysis ---
with tab2:
    st.subheader("Analyze Your Review")
    review_input2 = st.text_area("Enter your movie review:")
    if st.button("Analyze Sentiment"):
        if review_input2.strip():
            cleaned = review_input2.lower()
            X = tfidf_vectorizer.transform([cleaned])
            prediction = sentiment_model.predict(X)[0]
            prob = sentiment_model.predict_proba(X)[0]
            sentiment = "Positive 😀" if prediction == 1 else "Negative 😡"
            confidence = round(max(prob) * 100, 2)
            st.success(f"Prediction: {sentiment} (Confidence: {confidence}%)")
        else:
            st.warning("Please enter a review text.")
