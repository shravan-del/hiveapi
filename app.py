from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import re
import datetime
import numpy as np
import spacy
import hdbscan
import praw
from openai import OpenAI
from collections import Counter
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'PH99oWZjM43GimMtYigFvA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g')

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set in environment variables")

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Loading classifier and vectorizer...")
classifier = joblib.load("AutoClassifier.pkl")
vectorizer = joblib.load("AutoVectorizer.pkl")
print("Initializing TF-IDF embedder (lightweight)...")
embedder = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
print("Models loaded successfully")

print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent='Sentivity_Hive_NewsAggregator',
    check_for_async=False
)
print("Reddit client ready")

client = None

def get_openai_client():
    global client
    if client is None:
        print("Initializing OpenAI client...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client ready")
    return client

ENTITY_LABELS_TO_KEEP = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART"}
ENTITY_STOPLIST = {
    "Reddit", "YouTube", "Instagram", "Twitter",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Today", "Yesterday", "Tomorrow"
}

SOCIOPOLITICAL_SUBREDDITS = [
    "politics", "worldnews", "news", "geopolitics",
    "environment", "climate", "technology", "business"
]

def fetch_posts(subreddit_name: str, days_back: int = 7, limit: int = 100):
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)
    posts = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            created = datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc)
            if created >= cutoff:
                posts.append({
                    'title': post.title,
                    'created_utc': post.created_utc,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': subreddit_name
                })
    except Exception as e:
        print(f"Error fetching from r/{subreddit_name}: {e}")
    
    return posts

def simple_preprocess(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def classify_posts(texts: list):
    cleaned = [simple_preprocess(t) for t in texts]
    X = vectorizer.transform(cleaned)
    
    if X.shape[1] < 5000:
        pad_width = 5000 - X.shape[1]
        padding = sp.csr_matrix((X.shape[0], pad_width))
        X = sp.hstack([X, padding])
    
    predictions = classifier.predict(X)
    return predictions

def top_spacy_entities(texts: list, top_k=10):
    counts = Counter()
    
    for txt in texts:
        if not isinstance(txt, str) or not txt.strip():
            continue
        doc = nlp(txt)
        
        for token in doc:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            
            phrase = token.text.strip()
            
            if phrase in ENTITY_STOPLIST:
                continue
            if len(phrase) < 3:
                continue
            
            if token.pos_ == "NOUN":
                phrase = phrase.lower()
            
            counts[phrase] += 1
    
    return [phrase for phrase, _ in counts.most_common(top_k)]

def cluster_texts_with_hdbscan(texts: list, min_cluster_size: int = 20):
    embeddings = embedder.fit_transform(texts).toarray()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    return labels

def generate_header(cluster_texts: list):
    try:
        openai_client = get_openai_client()
        sample = "\n".join(cluster_texts[:10])
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Generate a single, concise AP-style news headline for these posts:\n\n{sample}"
            }],
            max_tokens=50,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating header: {e}")
        return "News Cluster"

def generate_summary(cluster_texts: list):
    try:
        openai_client = get_openai_client()
        sample = "\n".join(cluster_texts[:15])
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize the following news cluster in 2-3 sentences:\n\n{sample}"
            }],
            max_tokens=150,
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Unable to generate summary"

def analyze_news(days: int = 7, limit: int = 100):
    print(f"Fetching posts from {len(SOCIOPOLITICAL_SUBREDDITS)} subreddits...")
    
    all_posts = []
    for sub in SOCIOPOLITICAL_SUBREDDITS:
        posts = fetch_posts(sub, days, limit)
        all_posts.extend(posts)
    
    if not all_posts:
        return {"error": "No posts collected"}
    
    print(f"Collected {len(all_posts)} posts")
    
    df = pd.DataFrame(all_posts)
    texts = df['title'].tolist()
    
    print("Classifying posts...")
    predictions = classify_posts(texts)
    
    df['is_sociopolitical'] = predictions
    df_filtered = df[df['is_sociopolitical'] == 1].copy()
    
    if df_filtered.empty:
        return {"error": "No socio-political posts found"}
    
    print(f"Filtered to {len(df_filtered)} socio-political posts")
    
    filtered_texts = df_filtered['title'].tolist()
    
    print("Clustering posts...")
    labels = cluster_texts_with_hdbscan(filtered_texts, min_cluster_size=20)
    df_filtered['cluster'] = labels
    
    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        
        cluster_posts = df_filtered[df_filtered['cluster'] == label]
        cluster_texts = cluster_posts['title'].tolist()
        
        entities = top_spacy_entities(cluster_texts, top_k=5)
        
        clusters[int(label)] = {
            "size": len(cluster_posts),
            "texts": cluster_texts[:10],
            "entities": entities,
            "header": None,
            "summary": None
        }
    
    print(f"Found {len(clusters)} clusters, generating summaries...")
    
    for cluster_id, cluster_data in clusters.items():
        cluster_data['header'] = generate_header(cluster_data['texts'])
        cluster_data['summary'] = generate_summary(cluster_data['texts'])
    
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    news_stories = []
    for cluster_id, data in sorted_clusters:
        news_stories.append({
            "cluster_id": cluster_id,
            "headline": data['header'],
            "summary": data['summary'],
            "size": data['size'],
            "key_entities": data['entities'],
            "sample_posts": data['texts'][:5]
        })
    
    print(f"Analysis complete. Returning {len(news_stories)} stories")
    
    return {
        "news_stories": news_stories,
        "total_posts_analyzed": len(all_posts),
        "sociopolitical_posts": len(df_filtered),
        "clusters_found": len(clusters)
    }

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Hive - News Aggregation API",
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/analyze": "Run news analysis"
        },
        "note": "Aggregates and clusters socio-political news from Reddit"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 100))
        
        print(f"Starting news analysis (days={days}, limit={limit})")
        
        result = analyze_news(days, limit)
        
        if "error" in result:
            return jsonify(result), 400
        
        response = {
            "data": result,
            "meta": {
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "time_range_days": days,
                "posts_per_subreddit": limit
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
