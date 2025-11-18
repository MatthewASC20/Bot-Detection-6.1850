"""
Database handler for storing and retrieving YouTube comment data
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import logging
import json
from datetime import datetime

from config.config import Config

logger = logging.getLogger(__name__)

class DatabaseHandler:
    """Handle SQLite database operations"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Config.DATABASE_PATH
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Comments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    video_id TEXT,
                    text TEXT,
                    author TEXT,
                    author_id TEXT,
                    published_at TEXT,
                    updated_at TEXT,
                    like_count INTEGER,
                    is_reply BOOLEAN,
                    parent_id TEXT,
                    video_title TEXT,
                    channel_title TEXT,
                    channel_created_at TEXT,
                    channel_subscriber_count INTEGER,
                    created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    channel_id TEXT,
                    channel_title TEXT,
                    published_at TEXT,
                    duration TEXT,
                    view_count INTEGER,
                    like_count INTEGER,
                    comment_count INTEGER,
                    tags TEXT,
                    created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Channels table  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    published_at TEXT,
                    subscriber_count INTEGER,
                    video_count INTEGER,
                    view_count INTEGER,
                    country TEXT,
                    custom_url TEXT,
                    created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Detection results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_results (
                    author_id TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_confidence REAL,
                    cluster_bot_probability REAL,
                    individual_bot_probability REAL,
                    final_bot_probability REAL,
                    classification TEXT,
                    detection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_author ON comments(author_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id)")
            
            conn.commit()
            
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_comments(self, comments: List[Dict]):
        """Save comments to database"""
        if not comments:
            return
        
        df = pd.DataFrame(comments)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get existing comment IDs to avoid duplicates
            try:
                existing_ids = pd.read_sql_query(
                    "SELECT comment_id FROM comments", 
                    conn
                )['comment_id'].tolist()
                
                # Filter out duplicates
                df = df[~df['comment_id'].isin(existing_ids)]
                
                if len(df) == 0:
                    logger.info("No new comments to save (all are duplicates)")
                    return
                    
            except Exception:
                # Table doesn't exist yet, all comments are new
                pass
            
            df.to_sql('comments', conn, if_exists='append', index=False, method='multi')
        
        logger.info(f"Saved {len(df)} comments to database")
    
    def save_videos(self, videos: List[Dict]):
        """Save video information to database"""
        if not videos:
            return
        
        df = pd.DataFrame(videos)
        
        # Convert tags list to JSON string
        if 'tags' in df.columns:
            df['tags'] = df['tags'].apply(json.dumps)
        
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('videos', conn, if_exists='replace', index=False, method='multi')
        
        logger.info(f"Saved {len(videos)} videos to database")
    
    def save_channels(self, channels: List[Dict]):
        """Save channel information to database"""
        if not channels:
            return
        
        df = pd.DataFrame(channels)
        
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('channels', conn, if_exists='replace', index=False, method='multi')
        
        logger.info(f"Saved {len(channels)} channels to database")
    
    def save_detection_results(self, results_df: pd.DataFrame):
        """Save bot detection results to database"""
        with sqlite3.connect(self.db_path) as conn:
            results_df.to_sql('detection_results', conn, if_exists='replace', index=False)
        
        logger.info(f"Saved detection results for {len(results_df)} accounts")
    
    def get_all_comments(self) -> pd.DataFrame:
        """Retrieve all comments from database"""
        query = """
            SELECT c.*, ch.published_at as author_channel_created,
                   ch.subscriber_count as author_subscriber_count,
                   ch.video_count as author_video_count,
                   ch.view_count as author_total_views
            FROM comments c
            LEFT JOIN channels ch ON c.author_id = ch.channel_id
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        return df
    
    def get_comments_by_video(self, video_id: str) -> pd.DataFrame:
        """Get comments for a specific video"""
        query = """
            SELECT * FROM comments 
            WHERE video_id = ?
            ORDER BY published_at
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[video_id])
        
        return df
    
    def get_comments_by_author(self, author_id: str) -> pd.DataFrame:
        """Get all comments by a specific author"""
        query = """
            SELECT * FROM comments 
            WHERE author_id = ?
            ORDER BY published_at
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[author_id])
        
        return df
    
    def get_detection_results(self) -> pd.DataFrame:
        """Get latest detection results"""
        query = """
            SELECT * FROM detection_results
            ORDER BY final_bot_probability DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        return df
    
    def get_comments_count(self) -> int:
        """Get total number of comments"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM comments")
            return cursor.fetchone()[0]
    
    def get_videos_count(self) -> int:
        """Get total number of videos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos")
            return cursor.fetchone()[0]
    
    def get_channels_count(self) -> int:
        """Get total number of channels"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM channels")
            return cursor.fetchone()[0]
    
    def get_unique_authors_count(self) -> int:
        """Get number of unique comment authors"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT author_id) FROM comments")
            return cursor.fetchone()[0]
    
    def clear_all_data(self):
        """Clear all data from database (use with caution)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM comments")
            cursor.execute("DELETE FROM videos")
            cursor.execute("DELETE FROM channels")
            cursor.execute("DELETE FROM detection_results")
            conn.commit()
        
        logger.warning("Cleared all data from database")
