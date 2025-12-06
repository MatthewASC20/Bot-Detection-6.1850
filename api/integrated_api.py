"""
Integrated API for YouTube Bot Detector Chrome Extension
Connects the extension to the full bot detection pipeline with batch analysis

Place this file in: youtube_botnet_detector/api/integrated_api.py
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add to imports
from discovery.botnet_discovery import BotnetDiscovery
from discovery.expansion_crawler import ExpansionCrawler
from discovery.botnet_identifier import BotnetIdentifier
from visualization.dashboard_viz import generate_botnet_viz_html, generate_dashboard_html

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from features.temporal_features import TemporalFeatures
from features.text_features import TextFeatures
from features.network_features import NetworkFeatures
from features.behavioral_features import BehavioralFeatures
from detection.clustering import ClusteringDetector
from storage.database import DatabaseHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize components
text_features = TextFeatures()
detector = ClusteringDetector()
db = DatabaseHandler()

# In-memory cache
analysis_cache = {}
CACHE_TTL_HOURS = 24


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'features': ['batch_analysis', 'network_detection', 'clustering'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """Main endpoint for batch video comment analysis"""
    try:
        data = request.json
        video_id = data.get('video_id', 'unknown')
        comments = data.get('comments', [])
        include_network = data.get('include_network', True)
        fetch_related = data.get('fetch_related', False)
        
        if not comments:
            return jsonify({'error': 'No comments provided', 'video_id': video_id}), 400
        
        logger.info(f"Analyzing {len(comments)} comments for video {video_id}")
        
        # Check cache
        cache_key = f"{video_id}_{len(comments)}"
        if cache_key in analysis_cache:
            cached = analysis_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_TTL_HOURS):
                return jsonify(cached['result'])
        
        # Convert to DataFrame
        comments_df = pd.DataFrame(comments)
        comments_df = standardize_columns(comments_df)
        
        # Expand with related videos if requested
        if fetch_related:
            comments_df = expand_with_related_videos(comments_df, video_id)
        
        # Run detection pipeline
        result = run_detection_pipeline(comments_df, video_id, include_network)
        
        # Cache result
        analysis_cache[cache_key] = {'result': result, 'timestamp': datetime.now()}
        
        # Store for training
        store_analysis_results(comments_df, result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}", exc_info=True)
        return jsonify({'error': str(e), 'video_id': data.get('video_id', 'unknown')}), 500


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from extension format"""
    if 'content' in df.columns and 'text' not in df.columns:
        df['text'] = df['content']
    if 'video_id' not in df.columns:
        df['video_id'] = 'unknown'
    
    df['is_reply'] = df.get('is_reply', False).fillna(False) if 'is_reply' in df.columns else False
    df['parent_id'] = df.get('parent_id', None)
    df['like_count'] = pd.to_numeric(df.get('like_count', 0), errors='coerce').fillna(0)
    
    return df


def expand_with_related_videos(comments_df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """Expand with related video comments for cross-video detection"""
    try:
        author_ids = comments_df['author_id'].unique().tolist()
        related = db.get_comments_by_author_ids(author_ids)
        
        if related is not None and not related.empty:
            logger.info(f"Found {len(related)} related comments")
            combined = pd.concat([comments_df, related], ignore_index=True)
            return combined.drop_duplicates(subset=['comment_id'])
        return comments_df
    except Exception as e:
        logger.warning(f"Could not expand: {e}")
        return comments_df


def run_detection_pipeline(comments_df: pd.DataFrame, video_id: str, include_network: bool = True) -> Dict:
    """Run the full bot detection pipeline"""
    logger.info("Starting detection pipeline...")
    
    features_df = extract_all_features(comments_df, include_network)
    
    if features_df.empty:
        return format_empty_response(video_id, len(comments_df))
    
    detection_results = detector.detect_bots(features_df)
    
    merged_df = comments_df.merge(detection_results, on='author_id', how='left')
    merged_df['final_bot_probability'] = merged_df['final_bot_probability'].fillna(0)
    merged_df['cluster_id'] = merged_df['cluster_id'].fillna(-1)
    merged_df['classification'] = merged_df['classification'].fillna('likely_human')
    
    merged_df = generate_flags(merged_df, features_df)
    
    return format_analysis_response(merged_df, video_id)


def extract_all_features(comments_df: pd.DataFrame, include_network: bool = True) -> pd.DataFrame:
    """Extract all features from comments"""
    try:
        # Temporal features
        burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
        temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
        temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
        
        sync_groups = TemporalFeatures.detect_synchronized_posting(comments_df)
        sync_authors = set()
        for group in sync_groups:
            sync_authors.update(group)
        temporal_df['in_sync_group'] = temporal_df['author_id'].isin(sync_authors)
        
        # Text features
        template_scores = text_features.detect_template_comments(comments_df)
        spam_scores = text_features.detect_spam_patterns(comments_df)
        diversity_scores = text_features.calculate_comment_diversity(comments_df)
        linguistic_df = text_features.extract_linguistic_features(comments_df)
        
        linguistic_df['template_score'] = linguistic_df['author_id'].map(template_scores)
        linguistic_df['spam_score'] = linguistic_df['author_id'].map(spam_scores)
        linguistic_df['diversity_score'] = linguistic_df['author_id'].map(diversity_scores)
        
        duplicates = text_features.find_duplicate_comments(comments_df)
        dup_authors = set()
        for dup_group in duplicates:
            dup_comment_authors = comments_df[comments_df['comment_id'].isin(dup_group)]['author_id'].unique()
            dup_authors.update(dup_comment_authors)
        linguistic_df['has_duplicates'] = linguistic_df['author_id'].isin(dup_authors)
        
        # Network features
        if include_network and len(comments_df) >= 5:
            network_df = NetworkFeatures.calculate_author_network_features(comments_df)
        else:
            network_df = pd.DataFrame({
                'author_id': comments_df['author_id'].unique(),
                'co_degree': 0, 'community_id': -1, 'community_size': 0, 'in_clique': False
            })
        
        # Behavioral features
        behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
        
        # Merge all
        features_df = temporal_df.copy()
        for df in [linguistic_df, network_df, behavioral_df]:
            features_df = features_df.merge(df, on='author_id', how='outer', suffixes=('', '_dup'))
            features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
        
        return features_df.fillna(0)
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        return pd.DataFrame()


def generate_flags(merged_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate human-readable flags for each comment"""
    author_features = {row['author_id']: row.to_dict() for _, row in features_df.iterrows()}
    
    flags_list = []
    for _, row in merged_df.iterrows():
        flags = []
        af = author_features.get(row.get('author_id'), {})
        
        if af.get('template_score', 0) > 0.5: flags.append('template_match')
        if af.get('burst_score', 0) > 0.5: flags.append('burst_posting')
        if af.get('spam_score', 0) > 0.3: flags.append('spam_keywords')
        if af.get('in_sync_group', False): flags.append('synchronized')
        if af.get('in_clique', False): flags.append('clique_member')
        if af.get('has_duplicates', False): flags.append('duplicate_content')
        if af.get('automation_score', 0) > 0.6: flags.append('automated_behavior')
        if af.get('username_pattern_score', 0) > 0.5: flags.append('suspicious_username')
        if af.get('account_age_score', 1) < 0.3: flags.append('new_account')
        if af.get('co_degree', 0) > 10: flags.append('highly_connected')
        
        flags_list.append(flags)
    
    merged_df['flags'] = flags_list
    return merged_df


def format_analysis_response(merged_df: pd.DataFrame, video_id: str) -> Dict:
    """Format the analysis result for the extension"""
    total = len(merged_df)
    counts = merged_df['classification'].value_counts().to_dict()
    clusters = merged_df['cluster_id'].nunique() - (1 if -1 in merged_df['cluster_id'].values else 0)
    
    # Find coordinated groups
    coordinated = []
    for cid in merged_df['cluster_id'].unique():
        if cid == -1: continue
        cluster = merged_df[merged_df['cluster_id'] == cid]
        if len(cluster) >= 3 and cluster['final_bot_probability'].mean() > 0.6:
            coordinated.append({
                'cluster_id': int(cid),
                'size': len(cluster),
                'avg_bot_probability': float(cluster['final_bot_probability'].mean()),
                'authors': cluster['author_id'].tolist()[:10]
            })
    
    # Format comments
    comments = []
    for _, row in merged_df.iterrows():
        similar = []
        if row.get('cluster_id', -1) != -1:
            similar = merged_df[(merged_df['cluster_id'] == row['cluster_id']) & 
                               (merged_df['comment_id'] != row.get('comment_id'))]['comment_id'].tolist()[:5]
        
        comments.append({
            'comment_id': str(row.get('comment_id', '')),
            'author_id': str(row.get('author_id', '')),
            'bot_probability': float(row.get('final_bot_probability', 0)),
            'confidence': min(0.7 + (0.1 if row.get('cluster_id', -1) != -1 else 0) + 
                            (0.1 if len(row.get('flags', [])) >= 3 else 0.05 if row.get('flags') else 0), 0.95),
            'classification': str(row.get('classification', 'unknown')),
            'cluster_id': int(row.get('cluster_id', -1)),
            'flags': row.get('flags', []),
            'similar_to': [str(s) for s in similar]
        })
    
    return {
        'video_id': video_id,
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_summary': {
            'total_comments': total,
            'clusters_found': clusters,
            'likely_bots': counts.get('likely_bot', 0),
            'suspicious': counts.get('suspicious', 0),
            'likely_humans': counts.get('likely_human', 0),
            'coordinated_groups': coordinated,
            'detection_confidence': 0.85 if total >= 50 else 0.7
        },
        'comments': comments
    }


def format_empty_response(video_id: str, count: int) -> Dict:
    return {
        'video_id': video_id,
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_summary': {
            'total_comments': count, 'clusters_found': 0, 'likely_bots': 0,
            'suspicious': 0, 'likely_humans': count, 'coordinated_groups': [],
            'detection_confidence': 0.3
        },
        'comments': [],
        'error': 'Insufficient data'
    }


def store_analysis_results(comments_df: pd.DataFrame, result: Dict):
    try:
        for c in result.get('comments', []):
            db.update_author_profile(c.get('author_id'), c.get('bot_probability', 0), c.get('flags', []))
    except Exception as e:
        logger.warning(f"Could not store results: {e}")


@app.route('/api/vote', methods=['POST'])
def submit_vote():
    try:
        data = request.json
        db.store_extension_vote(
            data.get('comment_id'), data.get('author_id'), data.get('vote', 0),
            data.get('video_id'), data.get('ml_prediction'), data.get('comment_text')
        )
        return jsonify({'success': True, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify(db.get_extension_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/author/<author_id>', methods=['GET'])
def get_author_history(author_id: str):
    try:
        profile = db.get_author_profile(author_id)
        return jsonify(profile if profile else {'author_id': author_id, 'found': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_single():
    """Legacy single comment endpoint"""
    try:
        import re
        data = request.json
        author = data.get('author', '')
        content = data.get('content', '') or data.get('text', '')
        
        score = 0.0
        if re.search(r'\d{4,}', author): score += 0.2
        if len(author) < 5: score += 0.1
        if author.isupper() and len(author) > 3: score += 0.15
        
        spam_words = ['click', 'subscribe', 'check out', 'visit', 'earn money', 'free', 'winner']
        score += min(sum(1 for w in spam_words if w in content.lower()) * 0.1, 0.3)
        if 'http' in content.lower() or 'www.' in content.lower(): score += 0.25
        if len(re.findall(r'[\U0001F300-\U0001F9FF]', content)) > 5: score += 0.15
        
        return jsonify({
            'bot_probability': min(score, 1.0),
            'confidence': 0.5,
            'method': 'heuristic',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'bot_probability': 0.0, 'error': str(e)}), 500
# ============== BOTNET DISCOVERY MODULE ==============
# Initialize discovery
discovery = BotnetDiscovery(db)

# ============== BOTNET DISCOVERY ENDPOINTS ==============

@app.route('/api/discover', methods=['POST'])
def run_discovery():
    """
    Run full botnet discovery pipeline
    
    POST body:
    {
        "seed_video_ids": ["video1", "video2"],
        "seed_author_ids": ["author1"],
        "max_depth": 3,
        "min_bot_probability": 0.6,
        "min_botnet_size": 5
    }
    """
    try:
        data = request.json or {}
        
        result = discovery.run_full_discovery(
            seed_video_ids=data.get('seed_video_ids'),
            seed_author_ids=data.get('seed_author_ids'),
            max_expansion_depth=data.get('max_depth', 3),
            min_bot_probability=data.get('min_bot_probability', 0.6),
            min_botnet_size=data.get('min_botnet_size', 5)
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/botnets', methods=['GET'])
def list_botnets():
    """List all discovered botnets"""
    try:
        botnets = db.list_botnets()
        return jsonify({
            'count': len(botnets),
            'botnets': botnets
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/botnets/<botnet_id>', methods=['GET'])
def get_botnet(botnet_id: str):
    """Get detailed botnet information"""
    try:
        botnet = db.get_botnet(botnet_id)
        if botnet:
            return jsonify(botnet)
        else:
            return jsonify({'error': 'Botnet not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/investigate/<author_id>', methods=['GET'])
def investigate_author(author_id: str):
    """Deep investigation of a single author"""
    try:
        result = discovery.investigate_author(author_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/botnets/<botnet_id>/visualize', methods=['GET'])
def visualize_botnet(botnet_id: str):
    """Generate visualization for a specific botnet"""
    try:
        botnet = db.get_botnet(botnet_id)
        if not botnet:
            return jsonify({'error': 'Botnet not found'}), 404
        
        # Get all comments from botnet accounts
        accounts = botnet.get('accounts', [])
        comments_df = db.get_comments_by_author_ids(accounts)
        
        if comments_df is None or comments_df.empty:
            return jsonify({'error': 'No comment data available'}), 400
        
        # Build network
        G = NetworkFeatures.build_co_occurrence_network(comments_df)
        
        # Get bot scores from profiles
        bot_scores = {}
        for author_id in accounts:
            profile = db.get_author_profile(author_id)
            bot_scores[author_id] = profile.get('avg_bot_score', 0.5) if profile else 0.5
        
        communities = NetworkFeatures.detect_communities(G) if G.number_of_nodes() > 0 else {}
        
        # Generate HTML
        html = generate_botnet_viz_html(G, bot_scores, communities, botnet)
        
        return Response(html, mimetype='text/html')
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard', methods=['GET'])
def discovery_dashboard():
    """Main discovery dashboard with all botnets"""
    try:
        botnets = db.list_botnets()
        stats = db.get_extension_stats()
        
        html = generate_dashboard_html(botnets, stats)
        return Response(html, mimetype='text/html')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    print("=" * 60)
    print("YouTube Bot Detector - Integrated API v2.0")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /api/analyze_video  - Batch analysis (FULL PIPELINE)")
    print("  POST /api/analyze        - Single comment (legacy)")
    print("  POST /api/vote           - Submit vote")
    print("  GET  /api/stats          - Statistics")
    print("  GET  /api/author/<id>    - Author history")
    print("  GET  /api/health         - Health check")
    print("=" * 60)
    app.run(debug=True, port=5001, host='0.0.0.0')
