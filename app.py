import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. TMDB CONFIGURATION
# =====================================================

API_KEY = st.secrets.get("TMDB_API_KEY", "")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# =====================================================
# 3. NETFLIX-INSPIRED CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #101010;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background: #101010;
}

[data-testid="stSidebar"] {
    background: #181818;
    border-right: 1px solid #333;
}

[data-testid="stSidebar"] * {
    color: #eeeeee;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

p, label, span {
    color: #eeeeee;
}

.cine-logo {
    color: #e50914;
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -2px;
    margin-bottom: 0;
}

.cine-tagline {
    color: #bbbbbb;
    font-size: 12px;
    letter-spacing: 2px;
    margin-top: -5px;
    margin-bottom: 20px;
}

.hero {
    background: linear-gradient(110deg, #080808, #301010, #171717);
    border: 1px solid #333;
    border-radius: 14px;
    padding: 35px;
    margin: 15px 0 30px 0;
}

.hero h1 {
    font-size: 40px;
    margin-bottom: 8px;
}

.hero p {
    color: #dddddd;
    font-size: 16px;
}

.section-heading {
    color: white;
    font-size: 24px;
    font-weight: 800;
    margin: 25px 0 15px 0;
}

.movie-title {
    color: white;
    font-size: 15px;
    font-weight: 800;
    line-height: 1.4;
    min-height: 43px;
    margin-top: 10px;
}

.movie-year {
    color: #bbbbbb;
    font-size: 12px;
    margin-top: 5px;
}

.movie-rating {
    color: #f5c518;
    font-size: 13px;
    font-weight: 700;
    margin: 5px 0;
}

[data-testid="stImage"] img {
    height: 285px !important;
    width: 100% !important;
    object-fit: cover !important;
    border-radius: 7px;
}

.stButton > button {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    min-height: 40px;
}

.stButton > button:hover {
    background: #b20710 !important;
    color: white !important;
}

.stTextInput input {
    background: #252525 !important;
    color: white !important;
    border: 1px solid #555 !important;
}

.stTextInput input::placeholder {
    color: #aaaaaa !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: #252525 !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: white !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #191919;
    border-color: #333333 !important;
    border-radius: 10px;
}

hr {
    border-color: #333333;
}

.footer {
    color: #999999;
    text-align: center;
    font-size: 12px;
    padding: 25px;
}

.stCaption {
    color: #bbbbbb !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. TMDB API FUNCTION
# =====================================================

@st.cache_data(ttl=1800, show_spinner=False)
def tmdb_get(endpoint, params=None):

    if not API_KEY:
        return {}

    request_params = dict(params or {})
    request_params["api_key"] = API_KEY

    try:
        response = requests.get(
            BASE_URL + endpoint,
            params=request_params,
            timeout=20
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return {}


# =====================================================
# 5. MOVIE POSTER LOOKUP
# =====================================================

@st.cache_data(ttl=86400, show_spinner=False)
def search_tmdb_movie(title):

    data = tmdb_get(
        "/search/movie",
        {
            "query": title,
            "language": "en-US",
            "include_adult": False,
            "page": 1
        }
    )

    results = data.get("results", [])

    if not results:
        return None

    # Prefer a result that has a poster.
    for movie in results:
        if movie.get("poster_path"):
            return movie

    return results[0]


def get_poster(movie):

    if not movie:
        return None

    path = movie.get("poster_path")

    if path:
        return IMAGE_URL + path

    return None


def get_year(movie):

    date = movie.get("release_date", "")

    if date and len(date) >= 4:
        return date[:4]

    return "N/A"


# =====================================================
# 6. MOVIE CARD
# =====================================================

def display_movie(movie):

    title = movie.get("title", "Unknown Movie")
    movie_id = movie.get("id", title)

    with st.container(border=True):

        poster = get_poster(movie)

        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown("""
                <div style="
                    height:285px;
                    background:#292929;
                    border-radius:7px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#bbbbbb;
                    text-align:center;">
                    Poster unavailable
                </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f"<div class='movie-title'>{title}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='movie-year'>{get_year(movie)}</div>",
            unsafe_allow_html=True
        )

        rating = movie.get("vote_average", 0) or 0

        st.markdown(
            f"<div class='movie-rating'>★ {rating:.1f}/10</div>",
            unsafe_allow_html=True
        )

        if st.button(
            "View Details",
            key=f"movie_{movie_id}",
            use_container_width=True
        ):
            st.session_state.selected_movie = movie_id
            st.session_state.page = "Movie Details"
            st.rerun()


# =====================================================
# 7. MOVIE GRID
# =====================================================

def display_movie_grid(movies, columns=5):

    if not movies:
        st.warning("No movies found.")
        return

    for start in range(0, len(movies), columns):

        row_movies = movies[start:start + columns]
        cols = st.columns(columns)

        for col, movie in zip(cols, row_movies):

            with col:
                display_movie(movie)


# =====================================================
# 8. LOAD AI DATASET
# =====================================================

@st.cache_data
def load_dataset():

    try:
        df = pd.read_csv("tmdb_5000_movies.csv")
        return df

    except Exception:
        return pd.DataFrame()


movies_df = load_dataset()


# =====================================================
# 9. AI FEATURE PROCESSING
# =====================================================

def parse_features(value):

    try:
        data = ast.literal_eval(value)

        if isinstance(data, list):
            return " ".join(
                item.get("name", "")
                for item in data
                if isinstance(item, dict)
            )

    except (ValueError, SyntaxError, TypeError):
        pass

    return ""


# =====================================================
# 10. TF-IDF RECOMMENDATION MODEL
# =====================================================

@st.cache_resource
def build_recommendation_model():

    if movies_df.empty:
        return None, None

    df = movies_df.copy()

    required_columns = ["title", "overview"]

    if not all(col in df.columns for col in required_columns):
        return None, None

    if "genres" not in df.columns:
        df["genres"] = ""

    if "keywords" not in df.columns:
        df["keywords"] = ""

    df["genres_text"] = df["genres"].apply(parse_features)
    df["keywords_text"] = df["keywords"].apply(parse_features)

    df["combined_features"] = (
        df["title"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["genres_text"] + " " +
        df["keywords_text"]
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )

        matrix = vectorizer.fit_transform(
            df["combined_features"].fillna("")
        )

        return df, matrix

    except ValueError:
        return None, None


def recommend_movies(movie_title, count=10):

    df, matrix = build_recommendation_model()

    if df is None:
        return pd.DataFrame()

    matches = df[
        df["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    position = matches.index[0]

    similarity = cosine_similarity(
        matrix[position],
        matrix
    ).flatten()

    similar_indices = similarity.argsort()[::-1][1:count + 1]

    return df.iloc[similar_indices].copy()


# =====================================================
# 11. SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None


# =====================================================
# 12. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        "<div class='cine-logo'>CINEMATCH</div>"
        "<div class='cine-tagline'>MOVIES. DISCOVERY. MORE.</div>",
        unsafe_allow_html=True
    )

    st.divider()

    page_options = [
        "Home",
        "Explore Movies",
        "Search Movies",
        "AI Recommendations",
        "Movie Details"
    ]

    current_index = page_options.index(st.session_state.page)

    selected_page = st.radio(
        "NAVIGATION",
        page_options,
        index=current_index
    )

    st.session_state.page = selected_page

    st.divider()

    st.markdown("### 🎬 About CineMatch")

    st.caption(
        "Your personal movie discovery platform. "
        "Explore movies and find recommendations using machine learning."
    )

    st.divider()

    st.caption("Powered by TMDB API and Python")


# =====================================================
# 13. MAIN HEADER
# =====================================================

st.markdown(
    "<div class='cine-logo'>CINEMATCH</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='cine-tagline'>YOUR NEXT FAVOURITE MOVIE IS HERE</div>",
    unsafe_allow_html=True
)

if not API_KEY:
    st.error(
        "TMDB API key not found. Check your .streamlit/secrets.toml file."
    )


# =====================================================
# 14. HOME PAGE
# =====================================================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Unlimited Movies. Endless Discovery.</h1>
        <p>
            Discover trending films, explore new stories
            and find your next favourite movie.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-heading'>🔥 Trending Movies</div>",
        unsafe_allow_html=True
    )

    trending_data = tmdb_get(
        "/trending/movie/week",
        {"language": "en-US"}
    )

    trending = trending_data.get("results", [])

    display_movie_grid(trending[:10])

    st.markdown(
        "<div class='section-heading'>⭐ Top Rated Movies</div>",
        unsafe_allow_html=True
    )

    top_data = tmdb_get(
        "/movie/top_rated",
        {"language": "en-US", "page": 1}
    )

    top_movies = top_data.get("results", [])

    display_movie_grid(top_movies[:10])


# =====================================================
# 15. EXPLORE MOVIES
# =====================================================

elif st.session_state.page == "Explore Movies":

    st.title("Explore Movies")

    st.caption("Discover films from different genres and languages.")

    genre_data = tmdb_get("/genre/movie/list")

    genres = genre_data.get("genres", [])

    genre_options = {"All Genres": None}

    for genre in genres:
        genre_options[genre["name"]] = genre["id"]

    col1, col2 = st.columns(2)

    with col1:

        selected_genre = st.selectbox(
            "Select Genre",
            list(genre_options.keys())
        )

    with col2:

        language_options = {
            "All Languages": None,
            "Hindi": "hi",
            "English": "en",
            "Telugu": "te",
            "Tamil": "ta",
            "Malayalam": "ml",
            "Kannada": "kn"
        }

        selected_language = st.selectbox(
            "Select Language",
            list(language_options.keys())
        )

    params = {
        "sort_by": "popularity.desc",
        "language": "en-US",
        "page": 1,
        "include_adult": False
    }

    if genre_options[selected_genre]:
        params["with_genres"] = genre_options[selected_genre]

    if language_options[selected_language]:
        params["with_original_language"] = language_options[selected_language]

    data = tmdb_get("/discover/movie", params)

    results = data.get("results", [])

    display_movie_grid(results[:20])


# =====================================================
# 16. SEARCH MOVIES
# =====================================================

elif st.session_state.page == "Search Movies":

    st.title("Search Movies")

    st.caption(
        "Search for Bollywood, Hollywood and regional Indian movies."
    )

    query = st.text_input(
        "Enter Movie Name",
        placeholder="Try Interstellar, Dangal, RRR, 3 Idiots..."
    )

    if query.strip():

        results_data = tmdb_get(
            "/search/movie",
            {
                "query": query,
                "language": "en-US",
                "include_adult": False,
                "page": 1
            }
        )

        results = results_data.get("results", [])

        if results:

            st.success(f"Found {len(results)} results.")

            display_movie_grid(results[:20])

        else:

            st.warning(
                "No movies found. Please try another movie name."
            )


# =====================================================
# 17. AI RECOMMENDATIONS
# =====================================================

elif st.session_state.page == "AI Recommendations":

    st.title("🤖 AI Movie Recommendations")

    st.markdown(
        "Find movies similar to your favourites using "
        "**TF-IDF and Cosine Similarity**."
    )

    if movies_df.empty:

        st.error(
            "Dataset not found. Make sure tmdb_5000_movies.csv "
            "is inside your project folder."
        )

    elif "title" not in movies_df.columns:

        st.error("The dataset does not contain a title column.")

    else:

        available_titles = sorted(
            movies_df["title"].dropna().unique().tolist()
        )

        selected_title = st.selectbox(
            "Choose a movie you like",
            available_titles
        )

        if st.button(
            "Find My Recommendations",
            use_container_width=True
        ):

            with st.spinner("Finding movies similar to your choice..."):

                recommendations = recommend_movies(selected_title)

            if recommendations.empty:

                st.warning("No recommendations found.")

            else:

                st.success(
                    f"Movies similar to {selected_title}"
                )

                # Display five cards per row.
                rows = [
                    recommendations.iloc[i:i + 5]
                    for i in range(0, len(recommendations), 5)
                ]

                for row in rows:

                    cols = st.columns(5)

                    for col, (_, movie_row) in zip(cols, row.iterrows()):

                        with col:

                            movie_title = movie_row["title"]

                            # Search TMDB for the actual movie poster.
                            tmdb_movie = search_tmdb_movie(movie_title)

                            with st.container(border=True):

                                poster = get_poster(tmdb_movie)

                                if poster:

                                    st.image(
                                        poster,
                                        use_container_width=True
                                    )

                                else:

                                    st.markdown("""
                                        <div style="
                                            height:285px;
                                            background:#292929;
                                            border-radius:7px;
                                            display:flex;
                                            align-items:center;
                                            justify-content:center;
                                            color:#bbbbbb;
                                            text-align:center;">
                                            Poster unavailable
                                        </div>
                                    """, unsafe_allow_html=True)

                                st.markdown(
                                    f"<div class='movie-title'>{movie_title}</div>",
                                    unsafe_allow_html=True
                                )

                                rating = movie_row.get("vote_average", 0)

                                st.markdown(
                                    f"<div class='movie-rating'>★ {rating}/10</div>",
                                    unsafe_allow_html=True
                                )

                                if tmdb_movie and tmdb_movie.get("id"):

                                    if st.button(
                                        "View Details",
                                        key=f"ai_{tmdb_movie['id']}",
                                        use_container_width=True
                                    ):

                                        st.session_state.selected_movie = tmdb_movie["id"]
                                        st.session_state.page = "Movie Details"
                                        st.rerun()


# =====================================================
# 18. MOVIE DETAILS
# =====================================================

elif st.session_state.page == "Movie Details":

    movie_id = st.session_state.selected_movie

    if not movie_id:

        st.info("Select a movie from Home, Explore or Search first.")

    else:

        movie = tmdb_get(f"/movie/{movie_id}")

        if movie:

            st.title(movie.get("title", "Movie Details"))

            col1, col2 = st.columns([1, 2])

            with col1:

                poster = get_poster(movie)

                if poster:
                    st.image(poster, use_container_width=True)

            with col2:

                rating = movie.get("vote_average", 0)

                st.subheader(f"⭐ {rating:.1f}/10")

                st.write(
                    "**Release Date:**",
                    movie.get("release_date", "N/A")
                )

                st.write(
                    "**Runtime:**",
                    str(movie.get("runtime", "N/A")) + " minutes"
                )

                genre_names = ", ".join(
                    item["name"]
                    for item in movie.get("genres", [])
                )

                st.write("**Genres:**", genre_names or "N/A")

                st.write("**Overview:**")

                st.write(
                    movie.get("overview", "No description available.")
                )

            if st.button("Back to Home"):
                st.session_state.page = "Home"
                st.rerun()

        else:

            st.error("Unable to load movie details.")


# =====================================================
# 19. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    <b style="color:#e50914;font-size:18px;">CINEMATCH</b>
    <br>
    Discover. Explore. Enjoy.
    <br><br>
    This product uses the TMDB API but is not endorsed or certified by TMDB.
</div>
""", unsafe_allow_html=True)
