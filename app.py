# app.py
import streamlit as st
import pandas as pd
import time
import toml
import os
from atproto import exceptions
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
# Import refactored utilities
import utils


# salam cv kolchi bi5ir = {"salam","cv","kolchi "} 
# --- LOAD SECRETS FROM TOML ---
try:
    secrets = toml.load("secrets.toml")
    os.environ.update(secrets)
except FileNotFoundError:
    st.error("secrets.toml not found! Please check configuration.")
    st.stop()

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="BlueSky Discourse Observatory", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon=None
)

# --- CONSTANTS ---
DB_CONN_STR = os.environ.get("DB_CONN_STR")
BSKY_HANDLE = os.environ.get("BSKY_HANDLE")
BSKY_PASSWORD = os.environ.get("BSKY_PASSWORD")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PICPURIFY_API_KEY = os.environ.get("PICPURIFY_API_KEY", "")
PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY", "")

# --- SESSION STATE ---
for key in ['total_scans', 'total_threats', 'total_images', 'db_initialized', 'client_connected', 'bluesky_status']:
    if key not in st.session_state:
        if key == 'client_connected':
            st.session_state[key] = False
        elif key == 'bluesky_status':
            st.session_state[key] = "🔴 Not connected"
        else:
            st.session_state[key] = 0

if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'last_scan_data' not in st.session_state:
    st.session_state.last_scan_data = pd.DataFrame()
if 'client' not in st.session_state:
    st.session_state.client = None
if 'image_scan_results' not in st.session_state:
    st.session_state.image_scan_results = pd.DataFrame()
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'


# --- BLUESKY CONNECTION HELPER ---
def connect_bluesky():
    try:
        if not st.session_state.client_connected:
            st.session_state.client = utils.get_bsky_client_with_retry(BSKY_HANDLE, BSKY_PASSWORD)
            st.session_state.client_connected = True
            st.session_state.bluesky_status = "🟢 Connected"
            return True
        
        # Test connection
        st.session_state.client.get_timeline(limit=1)
        return True
    except exceptions.InvokeTimeoutError:
        st.session_state.client_connected = False
        st.session_state.bluesky_status = "🔴 Timeout - Retrying..."
        st.error("Bluesky connection timeout. Please check your internet connection.")
        return False
    except exceptions.UnauthorizedError:
        st.session_state.client_connected = False
        st.session_state.bluesky_status = "🔴 Auth failed"
        st.error("Authentication failed. Check your Bluesky credentials.")
        return False
    except Exception as e:
        st.session_state.client_connected = False
        st.session_state.bluesky_status = f"🔴 Error: {str(e)[:50]}..."
        st.error(f"Connection error: {str(e)}")
        return False

# --- UI VISUALS ---
# --- UI VISUALS ---
def get_custom_css(theme):
    if theme == 'dark':
        bg_gradient = "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"
        card_bg = "rgba(255, 255, 255, 0.1)"
        text_color = "white"
        metric_bg = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    else:
        bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"
        card_bg = "white"
        text_color = "#333"
        metric_bg = "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)"

    return f"""
    <style>
    .stApp {{
        background: {bg_gradient};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {card_bg};
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: {text_color};
        border: 1px solid rgba(255,255,255,0.2);
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #4a90e2 !important;
        color: white !important;
    }}
    .metric-card {{
        background: {metric_bg};
        padding: 20px;
        border-radius: 10px;
        color: {text_color};
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .stDataFrame {{
        background-color: {card_bg};
        padding: 1rem;
        border-radius: 10px;
    }}
    h1, h2, h3 {{
        color: {text_color} !important;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    </style>
    """

st.markdown(get_custom_css(st.session_state.theme), unsafe_allow_html=True)

# --- HEADER ---
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
with col1:
    st.title("BlueSky Discourse Observatory")
    st.caption("Advanced Bluesky Network Dynamics & Sentiment Analysis System")
with col2:
    status_color = "🟢" if st.session_state.client_connected else "🔴"
    st.metric("Bluesky Status", status_color)
with col3:
    db_status = "Connected" if st.session_state.db_initialized else "Disconnected"
    st.metric("Database", db_status)
with col4:
    if st.button("🌓 Theme"):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

# --- DATABASE INITIALIZATION ---
if not st.session_state.db_initialized:
    with st.spinner("Initializing database..."):
        result = utils.initialize_database(DB_CONN_STR)
        if result.startswith("✅"):
            st.success(result)
            st.session_state.db_initialized = True
        else:
            st.error(result)
            if st.button("🔄 Drop and Recreate Database", type="primary"):
                with st.spinner("Dropping and recreating database..."):
                    result = utils.drop_and_recreate_database(DB_CONN_STR)
                    if result.startswith("✅"):
                        st.success(result)
                        st.session_state.db_initialized = True
                        st.rerun()
                    else:
                        st.error(result)

# --- CONNECTION BUTTON ---
if not st.session_state.client_connected:
    if st.button("🔗 Connect to Bluesky", type="primary"):
        with st.spinner("Connecting to Bluesky..."):
            if connect_bluesky():
                st.success("Connected to Bluesky!")
                st.rerun()
else:
    st.success(f"Connected as @{BSKY_HANDLE}")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Keyword Scanner", "💬 Comment Analyzer", "🖼️ Image Guardian", "📊 Dashboard", "👤 User Profiler"])

# --- TAB 1: KEYWORD SCANNER ---
with tab1:
    st.header("Keyword Threat Scanner")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search Keywords", placeholder="e.g., violence hate threat")
        custom_keywords = st.text_area(
            "Custom Threat Keywords (one per line)", 
            placeholder="Add additional keywords to scan for\nviolence\nhate\nthreat\nharassment",
            height=100
        )
    with col2:
        limit = st.number_input("Post Limit", min_value=10, max_value=100, value=25, step=5)
        min_sentiment = st.slider("Min Sentiment for Threat", -1.0, 0.0, -0.7, step=0.1)
    
    if st.button("🚀 Launch Scan", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter keywords to search for")
        else:
            if not st.session_state.client_connected:
                st.error("Please connect to Bluesky first")
            else:
                try:
                    with st.spinner("Scanning Bluesky posts..."):
                        # Search for posts
                        posts = st.session_state.client.app.bsky.feed.search_posts(
                            {'q': query, 'limit': limit}
                        ).posts
                        
                        if not posts:
                            st.info("No posts found for your search query.")
                            st.stop()
                        
                        results = []
                        threats = []
                        alerts = 0
                        
                        # Prepare keyword list
                        custom_list = [k.strip().lower() for k in custom_keywords.split('\n') if k.strip()]
                        search_keywords = [k.strip() for k in query.split() if k.strip()]
                        all_keywords = list(set(search_keywords + custom_list))
                        
                        progress_bar = st.progress(0)
                        
                        for idx, p in enumerate(posts):
                            text_content = p.record.text or ""
                            
                            # Sentiment analysis using utils
                            compound, scores = utils.analyze_sentiment(text_content)
                            
                            # Classification
                            # Extract found keywords
                            found_kws = utils.extract_keywords(text_content, all_keywords)
                            found_str = ", ".join(found_kws.keys()) if found_kws else "None"
                            
                            # Professional Analysis (Perspective API) if available
                            toxicity_score = 0
                            p_scores = {}
                            if PERSPECTIVE_API_KEY:
                                p_scores = utils.analyze_toxicity_perspective(text_content, PERSPECTIVE_API_KEY)
                                if p_scores:
                                    toxicity_score = p_scores.get('TOXICITY', 0)

                            # Classification logic combining VADER and Perspective
                            is_threat = compound < min_sentiment or toxicity_score > 0.7
                            
                            if is_threat:
                                label = "⚠️ THREAT"
                                threats.append(p)
                                alerts += 1
                                
                                # Send Telegram Summary
                                alert_msg = f"""
<b>📢 Observatory Repost Summary</b>
<b>User:</b> @{p.author.handle}
<b>Status:</b> ⚠️ THREAT detected
<b>Toxicity (AI):</b> {toxicity_score:.1%}
<b>Vibe:</b> {compound:.3f}
<b>Content snippet:</b> {text_content[:80]}...
                                """
                                if not utils.send_telegram(alert_msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID):
                                    st.sidebar.error("Telegram Alert Failed! Check your Bot Token and Chat ID.")
                            else:
                                label = "⚪ NEUTRAL" if compound >= -0.3 else "🔵 NEGATIVE"
                            
                            results.append({
                                "handle": f"@{p.author.handle}",
                                "display_name": p.author.display_name or "N/A",
                                "post_text": text_content[:500],
                                "sentiment": round(compound, 3),
                                "positive_score": round(scores['pos'], 3),
                                "negative_score": round(scores['neg'], 3),
                                "neutral_score": round(scores['neu'], 3),
                                "toxicity": round(toxicity_score, 3),
                                "classification": label,
                                "keywords_found": found_str,
                                "created_at": p.record.created_at[:10] if hasattr(p.record, 'created_at') and p.record.created_at else "N/A",
                                "keywords": query
                            })
                            progress_bar.progress((idx + 1) / len(posts))
                        
                        # Create DataFrame
                        df = pd.DataFrame(results)
                        st.session_state.last_scan_data = df
                        
                        # Save to database
                        if not df.empty:
                            if utils.save_to_database(df, 'keyword_scans', DB_CONN_STR):
                                # Save scan history
                                utils.save_scan_history(
                                    scan_type="keyword",
                                    keywords=query,
                                    results_count=len(df),
                                    threats_count=alerts,
                                    db_conn_str=DB_CONN_STR
                                )
                                
                                # Update session state
                                st.session_state.total_scans += len(df)
                                st.session_state.total_threats += alerts
                                
                                st.success(f"✅ Scan completed! Found {len(df)} posts, {alerts} threats detected. Data saved to database.")
                        
                        # Display results
                        st.subheader(f"Results ({len(df)} posts)")
                        st.dataframe(df, use_container_width=True)

                except Exception as e:
                    st.error(f"Scan failed: {str(e)}")

# --- TAB 2: COMMENT ANALYZER ---
with tab2:
    st.header("💬 Deep Thread Analyzer")
    st.caption("Analyze the sentiment and toxicity of comments under any Bluesky post.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        post_url = st.text_input("Bluesky Post URL", placeholder="https://bsky.app/profile/user.bsky.social/post/...")
    with col2:
        max_comments = st.number_input("Max Comments", min_value=1, max_value=5000, value=50, step=1)
        analyze_btn = st.button("🧠 Analyze Comments", type="primary", use_container_width=True)
        
    if analyze_btn:
        if not post_url:
            st.warning("Please enter a valid Post URL.")
        elif not st.session_state.client_connected:
            st.error("Please connect to Bluesky first.")
        else:
            with st.spinner("Fetching thread and analyzing comments..."):
                try:
                    handle, rkey = utils.parse_bsky_url(post_url)
                    
                    if not handle or not rkey:
                        st.error("Invalid URL format. Expected: https://bsky.app/profile/.../post/...")
                        st.stop()
                        
                    did = utils.resolve_handle_to_did(st.session_state.client, handle)
                    if not did:
                        st.error(f"Could not resolve handle: {handle}")
                        st.stop()
                        
                    uri = f"at://{did}/app.bsky.feed.post/{rkey}"
                    thread_data = utils.get_post_thread_with_retry(st.session_state.client, uri, depth=100)
                    
                    if not thread_data or not hasattr(thread_data, 'thread'):
                        st.error("Could not fetch thread data. The post might be deleted or private.")
                    else:
                        comments = utils.process_thread(thread_data.thread, post_author_did=did, max_comments=max_comments)
                        
                        if not comments:
                            st.info("No comments found on this post.")
                        else:
                            st.success(f"Found {len(comments)} comments/replies.")
                            
                            analyzed_comments = []
                            loop_progress = st.progress(0)
                            for idx, c in enumerate(comments):
                                compound, scores = utils.analyze_sentiment(c['text'])
                                c['sentiment'] = compound
                                c['positive'] = scores['pos']
                                c['negative'] = scores['neg']
                                analyzed_comments.append(c)
                                loop_progress.progress((idx+1)/len(comments))
                            
                            c_df = pd.DataFrame(analyzed_comments)
                            
                            avg_sent = c_df['sentiment'].mean()
                            neg_count = len(c_df[c_df['sentiment'] < -0.3])
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Average Sentiment", f"{avg_sent:.3f}")
                            m2.metric("Negative Comments", neg_count)
                            m3.metric("Total Analyzed", len(c_df))
                            
                            db_save_df = c_df.rename(columns={
                                'author': 'author_handle',
                                'text': 'comment_text',
                                'positive': 'positive_score',
                                'negative': 'negative_score'
                            })
                            db_save_df['display_name'] = ""
                            db_save_df['depth'] = 0
                            db_save_df['parent_uri'] = uri
                            
                            utils.save_to_database(db_save_df, 'comment_analysis', DB_CONN_STR)
                            
                            st.dataframe(c_df[['author', 'text', 'sentiment', 'negative']], use_container_width=True)
                            
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

# --- TAB 3: IMAGE GUARDIAN ---
with tab3:
    st.header("Image Threat Scanner")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        scan_source = st.radio("Scan Source", ["My Timeline", "Search Keywords"])
    
    with col2:
        limit_img = st.number_input("Max Posts to Scan", 10, 100, 20)
        
    search_query = ""
    if scan_source == "Search Keywords":
        search_query = st.text_input("Enter keywords to find images", placeholder="e.g., cats, landscape, protest")
    
    if st.button(" Scan Images", type="primary"):
        if not st.session_state.client_connected:
            st.error("Connect to Bluesky first")
        elif scan_source == "Search Keywords" and not search_query:
            st.warning("Please enter a keyword to search.")
        else:
            with st.spinner(f"Scanning images from {scan_source}..."):
                try:
                    posts_to_scan = []
                    
                    if scan_source == "My Timeline":
                        timeline = st.session_state.client.get_timeline(limit=limit_img)
                        posts_to_scan = [feed_view.post for feed_view in timeline.feed]
                    else:
                        # Search posts
                        search_results = st.session_state.client.app.bsky.feed.search_posts(
                            {'q': search_query, 'limit': limit_img}
                        )
                        posts_to_scan = search_results.posts
                    
                    images_found = []
                    
                    for post in posts_to_scan:
                        imgs = utils.extract_images_from_post(post)
                        if imgs:
                            images_found.extend(imgs)
                    
                    if not images_found:
                        st.warning(f"No images found in the last {limit_img} posts for this source.")
                    else:
                        results = []
                        progress = st.progress(0)
                        
                        for idx, img_data in enumerate(images_found):
                            if img_data['image_url']:
                                decision, conf, cats, status = utils.check_image_picpurify(
                                    img_data['image_url'], PICPURIFY_API_KEY
                                )
                                
                                results.append({
                                    'author_handle': img_data['author_handle'],
                                    'image_url': img_data['image_url'],
                                    'decision': decision,
                                    'confidence': conf,
                                    'categories': ', '.join(cats),
                                    'status': status,
                                    'scan_date': datetime.now()
                                })
                                
                                if decision == "KO":
                                    alert_msg = f"""
<b>📢 Observatory Image Repost</b>
<b>User:</b> @{img_data['author_handle']}
<b>Flagged Category:</b> {cats}
<b>Action:</b> Review required
                                    """
                                    if not utils.send_telegram(alert_msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID):
                                        st.sidebar.error("Telegram Alert Failed! Check your Bot Token and Chat ID.")
                            
                            progress.progress((idx + 1) / len(images_found))
                            
                        img_df = pd.DataFrame(results)
                        st.session_state.image_scan_results = img_df
                        utils.save_to_database(img_df, 'image_scans', DB_CONN_STR)
                        
                        # Custom display with images
                        st.success(f"Scanned {len(images_found)} images.")
                        st.dataframe(
                            img_df,
                            column_config={
                                "image_url": st.column_config.ImageColumn("Image Preview")
                            }
                        )
                        
                except Exception as e:
                    import traceback
                    st.error(f"Image scan error: {str(e)}")
                    st.expander("Error Details").code(traceback.format_exc())

# --- TAB 4: DASHBOARD ---
with tab4:
    st.header("CTI Dashboard (Cyber Threat Intelligence)")
    
    if st.button("🔄 Refresh Dashboard Data"):
        with st.spinner("Crunching numbers..."):
            stats = utils.get_dashboard_stats(DB_CONN_STR)
            timeseries = utils.get_dashboard_timeseries(DB_CONN_STR)
            
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scans", stats.get('total_keyword_scans', 0))
                c2.metric("Threats Detected", stats.get('total_threats', 0), delta_color="inverse")
                c3.metric("Flagged Images", stats.get('flagged_images', 0), delta_color="inverse")
                c4.metric("Recent Activity (24h)", stats.get('recent_scans', 0))
                
                st.divider()
                
                if timeseries:
                    col_l, col_r = st.columns([2, 1])
                    with col_l:
                        st.subheader("Threat Activity")
                        if not timeseries['scans_over_time'].empty:
                            fig = px.line(timeseries['scans_over_time'], x='date', y='count', markers=True)
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white" if st.session_state.theme=='dark' else "black")
                            st.plotly_chart(fig, use_container_width=True)
                    with col_r:
                        st.subheader("Sentiment")
                        if not timeseries['sentiment_dist'].empty:
                            # Fix: px.donut does not exist, use px.pie with hole
                            fig = px.pie(timeseries['sentiment_dist'], values='count', names='classification', hole=0.4,
                                color='classification',
                                color_discrete_map={
                                    '⚠️ THREAT': '#ff4b4b',
                                    '🔵 NEGATIVE': '#ffa42be6',
                                    '⚪ NEUTRAL': '#cccccc'
                                })
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white" if st.session_state.theme=='dark' else "black")
                            st.plotly_chart(fig, use_container_width=True)
                
                    # New Visualizations Section
                    st.divider()
                    st.subheader("Deep Dive Analytics")
                    
                    if 'heatmap_data' in timeseries:
                        st.write("### 📅 Activity Heatmap (Day vs Hour)")
                        if not timeseries['heatmap_data'].empty:
                            # Pivot for heatmap
                            heatmap_df = timeseries['heatmap_data'].pivot(index='day_of_week', columns='hour_of_day', values='count').fillna(0)
                            # Ensure all days/hours represented if possible, or just plot what we have
                            fig_heat = px.imshow(heatmap_df, 
                                                labels=dict(x="Hour of Day", y="Day of Week", color="Scan Count"),
                                                x=heatmap_df.columns,
                                                y=heatmap_df.index,
                                                color_continuous_scale="Viridis")
                            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white" if st.session_state.theme=='dark' else "black")
                            st.plotly_chart(fig_heat, use_container_width=True)
                        else:
                            st.info("Not enough data for heatmap.")

                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.write("### 🌩️ Threat Details")
                        if not timeseries['threat_sunburst'].empty:
                            fig_sun = px.sunburst(
                                timeseries['threat_sunburst'], 
                                path=['classification', 'keywords'], 
                                values='count',
                                color='count',
                                color_continuous_scale='RdBu_r'
                            )
                            fig_sun.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white" if st.session_state.theme=='dark' else "black")
                            st.plotly_chart(fig_sun, use_container_width=True)
                    
                    with col_s2:
                         st.write("### ☁️ Keyword Cloud (Top 20)")
                         if not timeseries['top_keywords'].empty:
                             fig_bar = px.bar(timeseries['top_keywords'].head(20), x='count', y='keywords', orientation='h', color='count')
                             fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white" if st.session_state.theme=='dark' else "black", yaxis={'categoryorder':'total ascending'})
                             st.plotly_chart(fig_bar, use_container_width=True)

                else:
                    st.warning("No data for charts")
# --- TAB 5: USER PROFILER ---
with tab5:
    st.header("👤 Advanced User Profiler")
    st.caption("Deep historical analysis of a user's behavior, sentiment patterns, and threat potential.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_handle = st.text_input("Bluesky Handle", placeholder="e.g., user.bsky.social")
    with col2:
        user_limit = st.number_input("Analysis Depth (Posts)", min_value=10, max_value=500, value=50, step=10)
        profile_btn = st.button("📊 Profile User", type="primary", use_container_width=True)
        
    if profile_btn:
        if not user_handle:
            st.warning("Please enter a user handle.")
        elif not st.session_state.client_connected:
            st.error("Please connect to Bluesky first.")
        else:
            with st.spinner(f"Analyzing @{user_handle}..."):
                try:
                    # 1. Fetch author feed
                    feed_posts = utils.get_author_feed(st.session_state.client, user_handle, limit=user_limit)
                    
                    if not feed_posts:
                        st.error(f"Could not find posts for @{user_handle} or user does not exist.")
                    else:
                        # 2. Analyze posts
                        results = utils.analyze_user_posts(feed_posts)
                        
                        if results:
                            # 3. Display Metrics
                            m1, m2, m3 = st.columns(3)
                            
                            # Sentiment color logic
                            sent = results['avg_sentiment']
                            sent_label = "Neutral" if sent >= -0.3 else ("Negative" if sent >= -0.7 else "Very Negative")
                            m1.metric("Average Sentiment", f"{sent:.3f}", sent_label)
                            
                            # Threat color logic
                            tr = results['threat_ratio']
                            m2.metric("Threat Ratio", f"{tr:.1f}%", delta=f"{tr:.1f}%", delta_color="inverse")
                            
                            m3.metric("Posts Analyzed", results['post_count'])
                            
                            st.divider()
                            st.write(f"### ☁️ Top Keywords: {results['top_keywords']}")
                            
                            # 4. Save to database
                            profile_df = pd.DataFrame([{
                                'handle': user_handle,
                                'display_name': feed_posts[0].post.author.display_name if feed_posts else "",
                                'avg_sentiment': results['avg_sentiment'],
                                'threat_ratio': results['threat_ratio'],
                                'top_keywords': results['top_keywords'],
                                'post_count': results['post_count']
                            }])
                            
                            if utils.save_to_database(profile_df, 'user_analysis', DB_CONN_STR):
                                st.success(f"Analysis for @{user_handle} saved to database.")
                            
                            # 5. Show detailed post results
                            st.subheader("Historical Post Breakdown")
                            st.dataframe(results['results_df'], use_container_width=True)
                            
                            # Visualization: Sentiment over time
                            if not results['results_df'].empty:
                                fig = px.line(results['results_df'], x='created_at', y='sentiment', markers=True, title="Sentiment Trend")
                                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white" if st.session_state.theme=='dark' else "black")
                                st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Profiling failed: {str(e)}")
