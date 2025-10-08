import requests
import urllib.parse
import certifi

def get_movie_details(title, api_key):
    """Fetch movie plot and poster from TMDB API"""
    encoded_title = urllib.parse.quote(title)
    search_url = f"https://api.themoviedb.org/3/search/movie?query={encoded_title}&api_key={api_key}"

    try:
        # Use updated certificates for SSL handshake
        response = requests.get(search_url, verify=certifi.where(), timeout=10)
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            movie = data["results"][0]
            plot = movie.get("overview", "N/A")
            poster_path = movie.get("poster_path")

            poster_url = (
                f"https://image.tmdb.org/t/p/w500{poster_path}"
                if poster_path else "N/A"
            )

            return plot, poster_url

        print("❌ No movie found for:", title)
        return "N/A", "N/A"

    except Exception as e:
        print("⚠️ TMDB request failed:", e)
        return "N/A", "N/A"
