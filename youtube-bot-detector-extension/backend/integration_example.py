"""
Integration Example: Connecting Your Bot Detection System to the Chrome Extension

This file shows how to integrate your existing bot detection modules with the Chrome extension API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your existing bot detection modules
# Adjust these imports based on your actual project structure
try:
    from detection.clustering import BotDetector
    from feature_extraction.temporal_features import extract_temporal_features
    from feature_extraction.text_features import extract_text_features
    from feature_extraction.network_features import extract_network_features
    from feature_extraction.behavioral_features import extract_behavioral_features
except ImportError:
    print("Warning: Could not import bot detection modules. Using simple heuristics.")
    BotDetector = None

app = Flask(__name__)
CORS(app)

# Initialize your bot detector
bot_detector = BotDetector() if BotDetector else None

@app.route('/api/analyze', methods=['POST'])
def analyze_with_your_system():
    """
    Integrate your full bot detection system
    """
    try:
        data = request.json
        
        if bot_detector is None:
            # Fall back to simple heuristics
            return jsonify({
                'bot_probability': simple_heuristic_score(data),
                'method': 'heuristic'
            })
        
        # Extract all features using your modules
        features = extract_all_features(data)
        
        # Use your clustering/ML model to predict
        prediction = bot_detector.predict_single(features)
        
        # Get confidence score
        confidence = bot_detector.get_confidence(features)
        
        return jsonify({
            'bot_probability': prediction,
            'confidence': confidence,
            'features': features,
            'method': 'ml_model'
        })
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({
            'bot_probability': 0.0,
            'error': str(e)
        }), 500


def extract_all_features(comment_data):
    """
    Extract features using your existing modules
    
    Args:
        comment_data: Dictionary containing:
            - author: username
            - content: comment text
            - timestamp: when posted
            - likes: like count
            - authorLink: channel URL
    
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # Text features (from your text_features.py)
    if 'extract_text_features' in globals():
        text_feats = extract_text_features(
            comment_data.get('content', ''),
            comment_data.get('author', '')
        )
        features.update(text_feats)
    
    # Behavioral features (from your behavioral_features.py)
    if 'extract_behavioral_features' in globals():
        behavioral_feats = extract_behavioral_features(
            author=comment_data.get('author', ''),
            author_link=comment_data.get('authorLink', ''),
            is_pinned=comment_data.get('isPinned', False),
            is_hearted=comment_data.get('isHeartedByCreator', False)
        )
        features.update(behavioral_feats)
    
    return features


def simple_heuristic_score(data):
    """Simple scoring when ML model isn't available"""
    score = 0.0
    author = data.get('author', '')
    content = data.get('content', '')
    
    # Username checks
    import re
    if re.search(r'\d{4,}', author):
        score += 0.2
    if len(author) < 5:
        score += 0.1
    if author.isupper() and len(author) > 3:
        score += 0.15
    
    # Content checks
    spam_words = ['click', 'subscribe', 'check out', 'visit', 'earn money']
    content_lower = content.lower()
    spam_count = sum(1 for word in spam_words if word in content_lower)
    score += min(spam_count * 0.1, 0.3)
    
    if 'http' in content_lower or 'www.' in content_lower:
        score += 0.25
    
    return min(score, 1.0)


# Example: Advanced integration with your network analysis
@app.route('/api/analyze_network', methods=['POST'])
def analyze_network_context():
    """
    Analyze a comment in the context of a network of comments
    This endpoint can be used for more sophisticated detection
    """
    try:
        data = request.json
        comment = data.get('comment')
        video_id = data.get('video_id')
        
        # Get all comments from this video (you might cache these)
        # comments = get_video_comments(video_id)
        
        # Build network features
        # network_features = extract_network_features(comment, comments)
        
        # Detect communities
        # communities = detect_communities(comments)
        
        # Check if this comment is part of a bot cluster
        # is_in_bot_cluster = check_cluster_membership(comment, communities)
        
        return jsonify({
            'bot_probability': 0.0,  # Your calculation
            'network_analysis': {
                'cluster_id': None,
                'cluster_size': 0,
                'suspicious_cluster': False
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Example: Batch analysis endpoint
@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple comments at once for better network detection
    """
    try:
        comments = request.json.get('comments', [])
        
        # Extract features for all comments
        all_features = [extract_all_features(c) for c in comments]
        
        # Run clustering on this batch
        if bot_detector:
            predictions = bot_detector.predict_batch(all_features)
            
            # Detect coordinated behavior
            # coordinated_groups = detect_coordination(comments, predictions)
            
            return jsonify({
                'predictions': predictions.tolist(),
                'coordinated_groups': []  # Your coordination detection
            })
        
        return jsonify({'error': 'Detector not available'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting integrated bot detection API...")
    print("Endpoints available:")
    print("  POST /api/analyze - Single comment analysis")
    print("  POST /api/analyze_network - Network context analysis")
    print("  POST /api/analyze_batch - Batch comment analysis")
    app.run(debug=True, port=5000, host='0.0.0.0')
