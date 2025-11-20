"""
Flask API backend for YouTube Bot Detector Chrome Extension
Integrates with the existing bot detection system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import sqlite3

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

# Database setup for storing votes
DB_PATH = 'bot_detector_votes.db'

def init_db():
    """Initialize the votes database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comment_id TEXT NOT NULL,
            vote INTEGER NOT NULL,
            author TEXT,
            content TEXT,
            timestamp INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_comment_id ON votes(comment_id)
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_comment():
    """
    Analyze a YouTube comment for bot probability
    
    Expected JSON payload:
    {
        "author": "username",
        "content": "comment text",
        "timestamp": "2 hours ago",
        "likes": "5",
        "authorLink": "https://...",
        "isPinned": false,
        "isHeartedByCreator": false
    }
    """
    try:
        data = request.json
        
        # Extract features
        features = extract_features(data)
        
        # Calculate bot probability
        bot_probability = calculate_bot_probability(features)
        
        # Get historical votes for this comment if any
        votes = get_comment_votes(data.get('content', ''))
        
        return jsonify({
            'bot_probability': bot_probability,
            'features': features,
            'community_votes': votes,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'bot_probability': 0.0
        }), 500

def extract_features(comment_data):
    """Extract features from comment data"""
    author = comment_data.get('author', '')
    content = comment_data.get('content', '')
    
    features = {
        'author_length': len(author),
        'content_length': len(content),
        'has_numbers_in_username': bool(__import__('re').search(r'\d{4,}', author)),
        'is_all_caps_username': author.isupper() and len(author) > 3,
        'has_links': 'http' in content.lower() or 'www.' in content.lower(),
        'spam_keywords_count': count_spam_keywords(content),
        'emoji_count': count_emojis(content),
        'has_repetitive_chars': bool(__import__('re').search(r'(.)\1{3,}', content)),
        'is_very_short': len(content) < 10,
        'is_very_long': len(content) > 1000,
        'is_generic': is_generic_comment(content)
    }
    
    return features

def count_spam_keywords(text):
    """Count spam keywords in text"""
    spam_keywords = [
        'click here', 'check out', 'visit', 'subscribe', 
        'earn money', 'free', 'winner', 'congratulations',
        'limited time', 'act now', 'make money'
    ]
    text_lower = text.lower()
    return sum(1 for keyword in spam_keywords if keyword in text_lower)

def count_emojis(text):
    """Count emojis in text"""
    import re
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(text))

def is_generic_comment(text):
    """Check if comment is generic"""
    generic_phrases = [
        'nice video', 'great content', 'awesome', 'cool', 
        'first', 'early', 'good', 'nice', 'love this'
    ]
    text_lower = text.lower().strip()
    return any(phrase == text_lower for phrase in generic_phrases)

def calculate_bot_probability(features):
    """
    Calculate bot probability based on features
    This is a simple heuristic - can be replaced with ML model
    """
    score = 0.0
    
    # Username features
    if features['has_numbers_in_username']:
        score += 0.2
    if features['is_all_caps_username']:
        score += 0.15
    if features['author_length'] < 5:
        score += 0.1
    
    # Content features
    if features['has_links']:
        score += 0.25
    if features['spam_keywords_count'] > 0:
        score += min(features['spam_keywords_count'] * 0.1, 0.3)
    if features['emoji_count'] > 5:
        score += 0.15
    if features['has_repetitive_chars']:
        score += 0.1
    if features['is_very_short'] or features['is_very_long']:
        score += 0.1
    if features['is_generic']:
        score += 0.15
    
    return min(score, 1.0)

@app.route('/api/vote', methods=['POST'])
def submit_vote():
    """
    Submit a user vote on whether a comment is a bot
    
    Expected JSON payload:
    {
        "commentId": "unique-id",
        "vote": 1 or -1 or 0,
        "commentData": { ... },
        "timestamp": 1234567890
    }
    """
    try:
        data = request.json
        comment_id = data.get('commentId')
        vote = data.get('vote', 0)
        comment_data = data.get('commentData', {})
        timestamp = data.get('timestamp', int(datetime.now().timestamp() * 1000))
        
        # Store vote in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete existing vote for this comment (user can change their mind)
        cursor.execute('DELETE FROM votes WHERE comment_id = ?', (comment_id,))
        
        # Insert new vote if not neutral
        if vote != 0:
            cursor.execute('''
                INSERT INTO votes (comment_id, vote, author, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                comment_id,
                vote,
                comment_data.get('author', ''),
                comment_data.get('content', ''),
                timestamp
            ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Vote recorded',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_comment_votes(content):
    """Get community votes for similar comments"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get votes for this exact content
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN vote = 1 THEN 1 ELSE 0 END) as bot_votes,
                SUM(CASE WHEN vote = -1 THEN 1 ELSE 0 END) as human_votes
            FROM votes
            WHERE content = ?
        ''', (content,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'bot_votes': result[0] or 0,
                'human_votes': result[1] or 0
            }
        
        return {'bot_votes': 0, 'human_votes': 0}
    
    except Exception as e:
        print(f"Error getting votes: {e}")
        return {'bot_votes': 0, 'human_votes': 0}

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_votes,
                SUM(CASE WHEN vote = 1 THEN 1 ELSE 0 END) as bot_votes,
                SUM(CASE WHEN vote = -1 THEN 1 ELSE 0 END) as human_votes
            FROM votes
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return jsonify({
            'total_votes': result[0] or 0,
            'bot_votes': result[1] or 0,
            'human_votes': result[2] or 0
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting YouTube Bot Detector API...")
    print("API will be available at: http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')
