import requests
import time
import re
import pandas as pd
import nltk
from atproto import Client, exceptions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text, inspect
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime

# --- NLTK Setup ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- BLUESKY CLIENT ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type((exceptions.InvokeTimeoutError, exceptions.RequestException)))
def get_bsky_client_with_retry(handle, password):
    client = Client()
    client.login(handle, password)
    return client

def resolve_handle_to_did(client, handle):
    try:
        if not handle:
            return None
        if handle.startswith("did:"):
            return handle
        response = client.resolve_handle(handle)
        return response.did
    except Exception as e:
        print(f"Error resolving handle: {e}")
        return None

def parse_bsky_url(url):
    """
    Parses a Bluesky post URL to extract handle/DID and rkey.
    Supports:
    - https://bsky.app/profile/user.bsky.social/post/3l6m7v6n2k22b
    - https://bsky.app/profile/did:plc:123/post/3l6m7v6n2k22b
    - URLs with query params or trailing slashes
    """
    try:
        # Remove query params
        path = url.split('?')[0].strip('/')
        parts = path.split('/')
        
        if 'profile' in parts and 'post' in parts:
            handle_idx = parts.index('profile') + 1
            rkey_idx = parts.index('post') + 1
            
            if handle_idx < len(parts) and rkey_idx < len(parts):
                return parts[handle_idx], parts[rkey_idx]
        return None, None
    except Exception:
        return None, None

def get_post_thread_with_retry(client, uri, depth=100):
    try:
        # Increase default depth to fetch more comments (API limit is usually high)
        data = client.app.bsky.feed.get_post_thread({'uri': uri, 'depth': depth, 'parentHeight': 0})
        return data
    except Exception as e:
        print(f"Error fetching thread: {e}")
        return None

def process_thread(thread_view, post_author_did=None, max_comments=1000, _collected=None):
    """
    Recursively extract comments from a thread view.
    Returns a list of dictionaries.
    """
    if _collected is None:
        _collected = []
        
    # Stop recursion if we hit the limit
    if len(_collected) >= max_comments:
        return _collected
    
    # The 'thread' object can be 'ThreadViewPost', 'NotFoundPost', 'BlockedPost'
    if not thread_view or not hasattr(thread_view, 'post'):
        return _collected
        
    # Check for replies
    if hasattr(thread_view, 'replies') and thread_view.replies:
        for reply in thread_view.replies:
            if len(_collected) >= max_comments:
                break
                
            if hasattr(reply, 'post'):
                text = reply.post.record.text if hasattr(reply.post.record, 'text') else ""
                author = reply.post.author.handle
                created_at = reply.post.record.created_at if hasattr(reply.post.record, 'created_at') else ""
                
                comment_data = {
                    'text': text,
                    'author': author,
                    'created_at': created_at,
                    'uri': reply.post.uri,
                    'is_op': (reply.post.author.did == post_author_did) if post_author_did else False
                }
                
                _collected.append(comment_data)
                
                # Recurse
                process_thread(reply, post_author_did, max_comments, _collected)
            
    return _collected

# --- DATABASE FUNCTIONS ---
def initialize_database(db_conn_str):
    """Initialize database - create tables if they don't exist"""
    try:
        engine = create_engine(db_conn_str)
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        tables_to_create = {
            "keyword_scans": """
                CREATE TABLE IF NOT EXISTS keyword_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    handle VARCHAR(255),
                    display_name VARCHAR(255),
                    post_text TEXT,
                    sentiment FLOAT,
                    positive_score FLOAT,
                    negative_score FLOAT,
                    neutral_score FLOAT,
                    toxicity FLOAT,
                    classification VARCHAR(50),
                    created_at VARCHAR(100),
                    keywords VARCHAR(500),
                    keywords_found VARCHAR(500),
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_handle (handle),
                    INDEX idx_sentiment (sentiment),
                    INDEX idx_scan_date (scan_date)
                )
            """,
            "image_scans": """
                CREATE TABLE IF NOT EXISTS image_scans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    author_handle VARCHAR(255),
                    image_url TEXT,
                    decision VARCHAR(10),
                    confidence FLOAT,
                    categories TEXT,
                    status VARCHAR(50),
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_author (author_handle),
                    INDEX idx_decision (decision),
                    INDEX idx_scan_date (scan_date)
                )
            """,
            "comment_analysis": """
                CREATE TABLE IF NOT EXISTS comment_analysis (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    author_handle VARCHAR(255),
                    display_name VARCHAR(255),
                    comment_text TEXT,
                    sentiment FLOAT,
                    positive_score FLOAT,
                    negative_score FLOAT,
                    toxicity FLOAT,
                    depth INT,
                    parent_uri VARCHAR(500),
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_author (author_handle),
                    INDEX idx_parent (parent_uri(100)),
                    INDEX idx_scan_date (scan_date)
                )
            """,
            "scan_history": """
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    scan_type VARCHAR(50),
                    keywords TEXT,
                    results_count INT,
                    threats_detected INT,
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "user_analysis": """
                CREATE TABLE IF NOT EXISTS user_analysis (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    handle VARCHAR(255),
                    display_name VARCHAR(255),
                    avg_sentiment FLOAT,
                    threat_ratio FLOAT,
                    top_keywords TEXT,
                    post_count INT,
                    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_handle (handle),
                    INDEX idx_scan_date (scan_date)
                )
            """
        }
        
        created_count = 0
        with engine.connect() as conn:
            for table_name, create_sql in tables_to_create.items():
                if table_name not in existing_tables:
                    conn.execute(text(create_sql))
                    conn.commit()
                    created_count += 1
        
        if created_count > 0:
            return f"✅ Database initialized successfully! Created {created_count} new tables."
        else:
            return "✅ Database already initialized with correct schema."
    except Exception as e:
        return f"❌ Database Error: {str(e)}"

def drop_and_recreate_database(db_conn_str):
    """DROP all tables and recreate with correct schema"""
    try:
        engine = create_engine(db_conn_str)
        
        # Drop all existing tables
        with engine.connect() as conn:
            # Get all tables
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            
            # Drop all tables
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                conn.commit()
            
        # Re-initialize
        return initialize_database(db_conn_str)
    except Exception as e:
        return f"❌ Database Error: {str(e)}"

def save_to_database(df, table_name, db_conn_str):
    """Save DataFrame to MySQL database with column validation"""
    try:
        # Create a copy for database
        db_df = df.copy()
        
        # Define valid columns for each table
        valid_columns = {
            'keyword_scans': [
                'handle', 'display_name', 'post_text', 'sentiment',
                'positive_score', 'negative_score', 'neutral_score', 'toxicity',
                'classification', 'created_at', 'keywords', 'keywords_found', 'scan_date'
            ],
            'image_scans': [
                'author_handle', 'image_url', 'decision', 'confidence',
                'categories', 'status', 'scan_date'
            ],
            'comment_analysis': [
                'author_handle', 'display_name', 'comment_text', 'sentiment',
                'positive_score', 'negative_score', 'toxicity', 'depth', 'parent_uri', 'scan_date'
            ],
            'scan_history': [
                'scan_type', 'keywords', 'results_count', 'threats_detected', 'scan_date'
            ],
            'user_analysis': [
                'handle', 'display_name', 'avg_sentiment', 'threat_ratio', 'top_keywords', 'post_count', 'scan_date'
            ]
        }
        
        # Get valid columns for this table
        valid_cols = valid_columns.get(table_name, [])
        
        # Remove columns that don't belong in this table
        columns_to_drop = [col for col in db_df.columns if col not in valid_cols]
        if columns_to_drop:
            db_df = db_df.drop(columns=columns_to_drop)
        
        # Add scan date if not present
        if 'scan_date' not in db_df.columns:
            db_df['scan_date'] = datetime.now()
        
        # Ensure image_url is not None for image_scans
        if table_name == 'image_scans' and 'image_url' in db_df.columns:
            db_df['image_url'] = db_df['image_url'].fillna('')
        
        # Save to database
        engine = create_engine(db_conn_str)
        db_df.to_sql(table_name, con=engine, if_exists='append', index=False)
        return True
    except Exception as e:
        print(f"Failed to save to database: {str(e)}")
        return False

def save_scan_history(scan_type, keywords, results_count, threats_count, db_conn_str):
    """Save scan metadata to history table"""
    try:
        engine = create_engine(db_conn_str)
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO scan_history (scan_type, keywords, results_count, threats_detected)
                    VALUES (:scan_type, :keywords, :results_count, :threats_detected)
                """),
                {
                    "scan_type": scan_type,
                    "keywords": keywords,
                    "results_count": results_count,
                    "threats_detected": threats_count
                }
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"Failed to save scan history: {str(e)}")
        return False

def get_dashboard_stats(db_conn_str):
    try:
        engine = create_engine(db_conn_str)
        with engine.connect() as conn:
            stats = {}
            
            # Get counts from each table
            tables = ['keyword_scans', 'image_scans', 'comment_analysis']
            for table in tables:
                try:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                    stats[f'total_{table}'] = count
                except:
                    stats[f'total_{table}'] = 0
            
            # Get threat counts
            stats['total_threats'] = conn.execute(
                text("SELECT COUNT(*) FROM keyword_scans WHERE classification LIKE '%THREAT%'")
            ).scalar() or 0
            
            # Get flagged images
            stats['flagged_images'] = conn.execute(
                text("SELECT COUNT(*) FROM image_scans WHERE decision='KO'")
            ).scalar() or 0
            
            # Recent activity
            stats['recent_scans'] = conn.execute(
                text("SELECT COUNT(*) FROM keyword_scans WHERE scan_date > NOW() - INTERVAL 1 DAY")
            ).scalar() or 0
            
            return stats
    except Exception as e:
        print(f"Failed to get dashboard stats: {str(e)}")
        print(f"Failed to get dashboard stats: {str(e)}")
        return {}

def get_dashboard_timeseries(db_conn_str):
    """Get time-series data for charts"""
    try:
        engine = create_engine(db_conn_str)
        with engine.connect() as conn:
            # 1. Scans over time (by day)
            scans_query = text("""
                SELECT DATE(scan_date) as date, COUNT(*) as count 
                FROM keyword_scans 
                GROUP BY DATE(scan_date) 
                ORDER BY date ASC
            """)
            scans_df = pd.read_sql(scans_query, conn)
            
            # 2. Sentiment distribution
            sentiment_query = text("""
                SELECT classification, COUNT(*) as count 
                FROM keyword_scans 
                GROUP BY classification
            """)
            sentiment_df = pd.read_sql(sentiment_query, conn)
            
            # 3. Top Keywords
            # This is a bit tricky since keywords_found is a string. 
            # Ideally we'd normalize db, but for now we aggregate the 'keywords' column (search term)
            keywords_query = text("""
                SELECT keywords, COUNT(*) as count 
                FROM keyword_scans 
                GROUP BY keywords 
                ORDER BY count DESC 
                LIMIT 10
            """)
            keywords_df = pd.read_sql(keywords_query, conn)

            # 4. Heatmap Data (Day vs Hour)
            # MySQL: DAYNAME(scan_date), HOUR(scan_date)
            heatmap_query = text("""
                SELECT DAYNAME(scan_date) as day_of_week, HOUR(scan_date) as hour_of_day, COUNT(*) as count
                FROM keyword_scans
                GROUP BY day_of_week, hour_of_day
            """)
            heatmap_df = pd.read_sql(heatmap_query, conn)
            
            # 5. Sunburst Data (Classification -> Keyword)
            sunburst_query = text("""
                SELECT classification, keywords, COUNT(*) as count
                FROM keyword_scans
                WHERE classification != '⚪ NEUTRAL'
                GROUP BY classification, keywords
                LIMIT 50
            """)
            sunburst_df = pd.read_sql(sunburst_query, conn)

            return {
                'scans_over_time': scans_df,
                'sentiment_dist': sentiment_df,
                'top_keywords': keywords_df,
                'heatmap_data': heatmap_df,
                'threat_sunburst': sunburst_df
            }
    except Exception as e:
        print(f"Failed to get timeseries: {str(e)}")
        return None

# --- TELEGRAM ---
def send_telegram(msg, bot_token, chat_id):
    try:
        formatted_msg = f"📡 BlueSky Observatory Alert\n{msg}"
        response = requests.get(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            params={
                "chat_id": chat_id,
                "text": formatted_msg,
                "parse_mode": "HTML"
            },
            timeout=5
        )
        if response.status_code != 200:
            print(f"Telegram API Error: {response.text}")
            return False
        return True
    except Exception as e:
        print(f"Telegram connection failed: {str(e)}")
        return False

# --- ANALYSIS TOOLS ---
def extract_keywords(text, keywords_list):
    found = {}
    text_lower = text.lower()
    for kw in keywords_list:
        count = len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', text_lower))
        if count:
            found[kw] = count
    return found

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    return compound, scores

def analyze_toxicity_perspective(text, api_key):
    """Analyze text using Google Perspective API for professional-grade toxicity scores."""
    if not api_key:
        return None
    
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    params = {"key": api_key}
    
    payload = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {
            "TOXICITY": {},
            "SEVERE_TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {},
            "THREAT": {}
        }
    }
    
    try:
        response = requests.post(url, params=params, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            scores = {}
            for attr, attr_data in data.get("attributeScores", {}).items():
                scores[attr] = attr_data.get("summaryScore", {}).get("value", 0)
            return scores
        else:
            print(f"Perspective API Error: {response.text}")
            return None
    except Exception as e:
        print(f"Perspective Connection Failed: {str(e)}")
        return None

# --- IMAGE CHECK ---
def check_image_picpurify(image_url, api_key):
    """Check image using PicPurify API"""
    if not api_key:
        return "OK", 0.99, [], "API key not configured"
    
    if not image_url or image_url == 'None':
        return "Error", 0, [], "Invalid image URL"
    
    try:
        # Download the image first to send as a file
        # This is more robust than providing a URL as it avoids CDN blocking issues
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        img_response = requests.get(image_url, headers=headers, timeout=10)
        if img_response.status_code != 200:
            return "Error", 0, [], f"Could not download image from Bluesky (HTTP {img_response.status_code})"
        
        image_data = img_response.content
        
        # PicPurify uses a POST request with specific tasks
        tasks = 'porn_moderation,suggestive_nudity_moderation,gore_moderation,weapon_moderation,hate_sign_moderation'
        
        response = requests.post(
            'https://www.picpurify.com/analyse/1.1',
            data={
                'API_KEY': api_key,
                'task': tasks
            },
            files={
                'file_image': ('image.jpg', image_data, 'image/jpeg')
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'failed')
            
            if status == 'success':
                # PicPurify sometimes uses confidence_score_decision or confidence_score
                decision = data.get('final_decision', 'OK')
                confidence = data.get('confidence_score_decision', data.get('confidence_score', 0))
                
                # Extract flagged categories
                categories = []
                for task in tasks.split(','):
                    task_data = data.get(task, {})
                    if isinstance(task_data, dict) and task_data.get('decision') == 'KO':
                        categories.append(task.replace('_moderation', ''))
                
                return decision, confidence, categories, "Success"
            else:
                # PicPurify uses 'error' object with 'errorCode' and 'errorMsg'
                error_info = data.get('error', {})
                error_code = error_info.get('errorCode', 'N/A')
                error_msg = error_info.get('errorMsg', 'Unknown error')
                return "Error", 0, [], f"PicPurify Error {error_code}: {error_msg}"
        else:
            return "Error", 0, [], f"API HTTP Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Error", 0, [], "Request timeout"
    except requests.exceptions.RequestException as e:
        return "Error", 0, [], f"Request failed: {str(e)}"
    except Exception as e:
        return "Error", 0, [], f"Unexpected error: {str(e)}"

def extract_images_from_post(post):
    """Extract image URLs from a Bluesky post (using hydrated View)"""
    images = []
    
    # We use post.embed (the View) because it contains fully resolved URLs (fullsize/thumb)
    # post.record (the Record) only contains Blobs (CIDs).
    if not hasattr(post, 'embed') or not post.embed:
        return images

    embed = post.embed
    
    # 1. Standard Images
    if hasattr(embed, 'images') and embed.images:
        for img in embed.images:
            # ViewImage has 'fullsize', 'thumb', 'alt'
            if hasattr(img, 'fullsize') and img.fullsize:
                images.append({
                    'author_handle': post.author.handle,
                    'image_url': img.fullsize,
                    'alt': getattr(img, 'alt', '')
                })

    # 2. External Link with Thumbnail
    elif hasattr(embed, 'external') and embed.external:
        if hasattr(embed.external, 'thumb') and embed.external.thumb:
            images.append({
                'author_handle': post.author.handle,
                'image_url': embed.external.thumb,
                'alt': getattr(embed.external, 'description', '')[:100]
            })

    # 3. Record With Media (Quote Post + Media)
    elif hasattr(embed, 'media') and embed.media:
        # The media part can be images or external
        media = embed.media
        if hasattr(media, 'images') and media.images:
            for img in media.images:
                if hasattr(img, 'fullsize') and img.fullsize:
                    images.append({
                        'author_handle': post.author.handle,
                        'image_url': img.fullsize,
                        'alt': getattr(img, 'alt', '')
                    })
        elif hasattr(media, 'external') and media.external:
            if hasattr(media.external, 'thumb') and media.external.thumb:
                images.append({
                    'author_handle': post.author.handle,
                    'image_url': media.external.thumb,
                    'alt': getattr(media.external, 'description', '')[:100]
                })

    return images


def get_author_feed(client, handle, limit=50):
    """Fetch recent posts from a specific user"""
    try:
        did = resolve_handle_to_did(client, handle)
        if not did:
            return None
        
        feed = client.app.bsky.feed.get_author_feed({'actor': did, 'limit': limit})
        return feed.feed # List of FeedViewPost
    except Exception as e:
        print(f"Error fetching author feed: {e}")
        return None

def analyze_user_posts(feed_posts):
    """
    Analyze a list of FeedViewPost for sentiment and behavior.
    """
    if not feed_posts:
        return None
        
    results = []
    total_sentiment = 0
    threat_count = 0
    all_text = ""
    
    for item in feed_posts:
        post = item.post
        text = post.record.text if hasattr(post.record, 'text') else ""
        all_text += " " + text
        
        compound, scores = analyze_sentiment(text)
        total_sentiment += compound
        
        # Consistent with Keyword Scanner: < -0.7 is THREAT
        is_threat = compound < -0.7
        if is_threat:
            threat_count += 1
            
        results.append({
            'created_at': post.record.created_at[:10] if hasattr(post.record, 'created_at') and post.record.created_at else "N/A",
            'post_text': text[:500],
            'sentiment': round(compound, 3),
            'classification': "⚠️ THREAT" if is_threat else ("⚪ NEUTRAL" if compound >= -0.3 else "🔵 NEGATIVE")
        })
        
    avg_sentiment = total_sentiment / len(feed_posts)
    threat_ratio = (threat_count / len(feed_posts)) * 100
    
    # Extract top keywords (simple frequency)
    import re
    from collections import Counter
    words = re.findall(r'\w+', all_text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'for', 'in', 'on', 'at', 'by', 'this', 'that', 'with', 'from', 'as', 'it', 'be', 'you', 'your', 'my', 'me', 'i', 'we', 'us', 'they', 'them', 'he', 'she', 'it'}
    filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
    
    top_kws = ", ".join([w for w, _ in Counter(filtered_words).most_common(10)])
    
    return {
        'avg_sentiment': avg_sentiment,
        'threat_ratio': threat_ratio,
        'top_keywords': top_kws,
        'post_count': len(feed_posts),
        'results_df': pd.DataFrame(results)
    }
