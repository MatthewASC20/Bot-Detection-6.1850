"""
Database handler for storing and retrieving YouTube comment data
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import logging
import json
from datetime import datetime
import json

from config.config import Config

logger = logging.getLogger(__name__)

class DatabaseHandler:
    """Handle SQLite database operations"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Config.DATABASE_PATH
        self.db_path = db_path
        self._init_database()
        # Ensure extension/voting/profile tables exist alongside core schema
        self._init_extension_tables()
        self._init_botnet_tables()
    
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
        
        # Deduplicate within the batch first
        original_count = len(df)
        df = df.drop_duplicates(subset=['comment_id'], keep='first')
        if len(df) < original_count:
            logger.info(f"Removed {original_count - len(df)} duplicate comments within batch")
        
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

    # Add to storage/database.py - Phase 2 Database Extensions

    def _init_extension_tables(self):
        """Initialize extension-specific tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Extension votes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extension_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_hash TEXT NOT NULL,
                    author_id TEXT,
                    vote INTEGER NOT NULL,
                    video_id TEXT,
                    ml_prediction REAL,
                    comment_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Author profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS author_profiles (
                    author_id TEXT PRIMARY KEY,
                    total_comments INTEGER DEFAULT 0,
                    avg_bot_score REAL DEFAULT 0,
                    total_analyses INTEGER DEFAULT 0,
                    flags_json TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_votes_author ON extension_votes(author_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_votes_video ON extension_votes(video_id)")
            conn.commit()

    def store_extension_vote(self, comment_id: str, author_id: str, vote: int,
                            video_id: str = None, ml_prediction: float = None,
                            comment_text: str = None):
        """Store a user vote from the extension"""
        import hashlib
        comment_hash = hashlib.md5(f"{comment_text or ''}{author_id or ''}".encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Remove existing vote for this comment
            cursor.execute("DELETE FROM extension_votes WHERE comment_hash = ?", (comment_hash,))

            if vote != 0:
                cursor.execute("""
                    INSERT INTO extension_votes 
                    (comment_hash, author_id, vote, video_id, ml_prediction, comment_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (comment_hash, author_id, vote, video_id, ml_prediction, comment_text))

            conn.commit()

    def update_author_profile(self, author_id: str, bot_probability: float, flags: list):
        """Update author profile with new analysis data"""
        if not author_id:
            return

        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if profile exists
            cursor.execute("SELECT * FROM author_profiles WHERE author_id = ?", (author_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE author_profiles SET
                        total_analyses = total_analyses + 1,
                        avg_bot_score = (avg_bot_score * total_analyses + ?) / (total_analyses + 1),
                        flags_json = ?,
                        last_seen = CURRENT_TIMESTAMP
                    WHERE author_id = ?
                """, (bot_probability, json.dumps(flags), author_id))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO author_profiles 
                    (author_id, total_analyses, avg_bot_score, flags_json, first_seen, last_seen)
                    VALUES (?, 1, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (author_id, bot_probability, json.dumps(flags)))

            conn.commit()

    def get_author_profile(self, author_id: str) -> Optional[Dict]:
        """Get author profile"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM author_profiles WHERE author_id = ?", (author_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'author_id': row[0],
                    'total_comments': row[1],
                    'avg_bot_score': row[2],
                    'total_analyses': row[3],
                    'flags': json.loads(row[4]) if row[4] else [],
                    'first_seen': row[5],
                    'last_seen': row[6],
                    'found': True
                }
            return None

    def get_comments_by_author_ids(self, author_ids: List[str]) -> pd.DataFrame:
        """Get all comments from specified authors"""
        if not author_ids:
            return pd.DataFrame()

        placeholders = ','.join(['?' for _ in author_ids])
        query = f"SELECT * FROM comments WHERE author_id IN ({placeholders})"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=author_ids)

    def get_extension_stats(self) -> Dict:
        """Get extension statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(*) as total_votes,
                    SUM(CASE WHEN vote = 1 THEN 1 ELSE 0 END) as bot_votes,
                    SUM(CASE WHEN vote = -1 THEN 1 ELSE 0 END) as human_votes
                FROM extension_votes
            """)
            result = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM author_profiles")
            profiles = cursor.fetchone()[0]

            return {
                'total_votes': result[0] or 0,
                'bot_votes': result[1] or 0,
                'human_votes': result[2] or 0,
                'tracked_authors': profiles
            }
    def _init_botnet_tables(self):
        """Initialize botnet tracking tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Botnets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS botnets (
                    botnet_id TEXT PRIMARY KEY,
                    auto_name TEXT,
                    size INTEGER,
                    confidence REAL,
                    accounts_json TEXT,
                    targets_json TEXT,
                    metrics_json TEXT,
                    temporal_patterns_json TEXT,
                    text_patterns_json TEXT,
                    role_assignments_json TEXT,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Discovery runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovery_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameters_json TEXT,
                    summary_json TEXT,
                    botnets_found INTEGER,
                    total_accounts INTEGER,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            conn.commit()

    def save_botnet(self, botnet: Dict):
        """Save a discovered botnet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO botnets 
                (botnet_id, auto_name, size, confidence, accounts_json, targets_json, 
                metrics_json, temporal_patterns_json, text_patterns_json, role_assignments_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                botnet['botnet_id'],
                botnet.get('auto_name', ''),
                botnet.get('size', 0),
                botnet.get('confidence', 0),
                json.dumps(botnet.get('accounts', [])),
                json.dumps(botnet.get('targets', {})),
                json.dumps(botnet.get('metrics', {})),
                json.dumps(botnet.get('temporal_patterns', {})),
                json.dumps(botnet.get('text_patterns', {})),
                json.dumps(botnet.get('role_assignments', {}))
            ))
            
            conn.commit()

    def get_botnet(self, botnet_id: str) -> Optional[Dict]:
        """Get a botnet by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM botnets WHERE botnet_id = ?", (botnet_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'botnet_id': row[0],
                    'auto_name': row[1],
                    'size': row[2],
                    'confidence': row[3],
                    'accounts': json.loads(row[4]) if row[4] else [],
                    'targets': json.loads(row[5]) if row[5] else {},
                    'metrics': json.loads(row[6]) if row[6] else {},
                    'temporal_patterns': json.loads(row[7]) if row[7] else {},
                    'text_patterns': json.loads(row[8]) if row[8] else {},
                    'role_assignments': json.loads(row[9]) if row[9] else {},
                    'discovered_at': row[10],
                    'updated_at': row[11]
                }
            return None

    def list_botnets(self) -> List[Dict]:
        """List all discovered botnets"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT botnet_id, auto_name, size, confidence, discovered_at FROM botnets ORDER BY size DESC")
            rows = cursor.fetchall()
            
            return [{
                'botnet_id': row[0],
                'auto_name': row[1],
                'size': row[2],
                'confidence': row[3],
                'discovered_at': row[4]
            } for row in rows]

    def save_discovery_run(self, results: Dict):
        """Save a discovery run"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO discovery_runs 
                (parameters_json, summary_json, botnets_found, total_accounts, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(results.get('parameters', {})),
                json.dumps(results.get('summary', {})),
                len(results.get('phases', {}).get('identification', {}).get('botnets', [])),
                results.get('phases', {}).get('expansion', {}).get('total_accounts', 0),
                results.get('started_at'),
                results.get('completed_at')
            ))
            
            conn.commit()

    def get_high_probability_authors(self, min_probability: float, limit: int = 100) -> List[str]:
        """Get authors with high bot probability"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT author_id FROM author_profiles 
                WHERE avg_bot_score >= ? 
                ORDER BY avg_bot_score DESC 
                LIMIT ?
            """, (min_probability, limit))
            
            return [row[0] for row in cursor.fetchall()]

    def get_comments_by_video(self, video_id: str) -> pd.DataFrame:
        """Get all comments for a video"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM comments WHERE video_id = ?", 
                conn, 
                params=[video_id]
            )

    def get_comments_by_author(self, author_id: str) -> pd.DataFrame:
        """Get all comments by an author"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM comments WHERE author_id = ?", 
                conn, 
                params=[author_id]
            )
