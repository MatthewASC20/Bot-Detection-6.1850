"""
Configuration settings for YouTube Botnet Detector
"""
import os
from dotenv import load_dotenv
from typing import List, Dict
import logging

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class"""
    
    # API Configuration
    YOUTUBE_API_KEYS = [
        os.getenv('YOUTUBE_API_KEY'),
        os.getenv('YOUTUBE_API_KEY_2'),
        os.getenv('YOUTUBE_API_KEY_3')
    ]
    YOUTUBE_API_KEYS = [key for key in YOUTUBE_API_KEYS if key]  # Filter None values
    
    if not YOUTUBE_API_KEYS:
        raise ValueError("No YouTube API keys found in .env file")
    
    # API Quotas and Limits
    MAX_RESULTS_PER_PAGE = 100  # YouTube API maximum
    DEFAULT_MAX_COMMENTS = 1000
    API_RETRY_COUNT = 3
    API_RETRY_DELAY = 2  # seconds
    
    # Data Collection Settings
    POLITICAL_CHANNELS = [
        'UCupvZG-5ko_eiXAupbDfxWw',  # CNN
        'UCaXkIU1QidjPwiAYu6GcHjg',  # MSNBC
        'UCGg45r7uOH90vVgLR_W3ftQ',  # Fox News
        'UC1yBKRuGpC1tSM73A0ZjYjQ',  # The Young Turks
        'UCLXo7UDZvByw2ixzpQqufnA',  # Vox
        'UC52X5wxOL_s5yw0dQk7NtgA',  # The Daily Wire
    ]
    
    POLITICAL_KEYWORDS = [
        'trump', 'biden', 'democrat', 'republican', 'liberal', 'conservative',
        'election', 'vote', 'policy', 'government', 'politics'
    ]
    
    # Feature Extraction Settings
    TIME_WINDOW_SECONDS = 300  # 5 minutes for burst detection
    MIN_COMMENT_LENGTH = 10
    MAX_COMMENT_LENGTH = 5000
    SIMILARITY_THRESHOLD = 0.85  # For text similarity
    
    # Clustering Parameters
    MIN_CLUSTER_SIZE = 3  # Minimum accounts to form a bot cluster
    MIN_SAMPLES = 2  # For HDBSCAN
    CLUSTER_SELECTION_EPSILON = 0.3
    
    # Detection Thresholds
    BOT_PROBABILITY_THRESHOLD = 0.7
    SUSPICIOUS_PROBABILITY_THRESHOLD = 0.5
    
    # Behavioral Indicators
    SUSPICIOUS_ACCOUNT_AGE_DAYS = 30  # Accounts newer than this are suspicious
    SUSPICIOUS_COMMENT_RATE = 10  # Comments per hour
    USERNAME_PATTERN_THRESHOLD = 0.7  # Similarity threshold for usernames
    
    # Network Analysis
    MIN_EDGE_WEIGHT = 2  # Minimum co-occurrences to create edge
    COMMUNITY_RESOLUTION = 1.0  # For Louvain community detection
    
    # Storage Settings
    DATABASE_PATH = 'data/botnet_detection.db'
    CACHE_EXPIRY_HOURS = 24
    
    # Visualization Settings
    GRAPH_LAYOUT = 'spring'  # Options: 'spring', 'circular', 'kamada_kawai'
    NODE_SIZE_MULTIPLIER = 100
    EDGE_WIDTH_MULTIPLIER = 2
    MAX_GRAPH_NODES = 500  # Limit for readability
    
    # Output Settings
    OUTPUT_DIR = 'outputs'
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'graphs')
    REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Feature Weights for Ensemble Detection
    FEATURE_WEIGHTS = {
        'temporal_burst': 0.25,
        'text_similarity': 0.20,
        'network_connectivity': 0.20,
        'account_age': 0.15,
        'username_pattern': 0.10,
        'comment_rate': 0.10
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            'data', 'data/raw', 'data/processed', 'data/labeled', 'data/results',
            cls.OUTPUT_DIR, cls.GRAPHS_DIR, cls.REPORTS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_api_key(cls, index: int = 0) -> str:
        """Get API key by index (for rotation)"""
        return cls.YOUTUBE_API_KEYS[index % len(cls.YOUTUBE_API_KEYS)]
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if not cls.YOUTUBE_API_KEYS:
            raise ValueError("No YouTube API keys configured")
        
        if cls.MIN_CLUSTER_SIZE < 2:
            raise ValueError("MIN_CLUSTER_SIZE must be at least 2")
        
        if not (0 <= cls.BOT_PROBABILITY_THRESHOLD <= 1):
            raise ValueError("BOT_PROBABILITY_THRESHOLD must be between 0 and 1")
        
        return True

# Initialize configuration
Config.create_directories()
Config.validate_config()
