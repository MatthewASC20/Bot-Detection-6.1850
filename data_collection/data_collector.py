"""
Main data collection module for coordinating YouTube data gathering
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import json
from tqdm import tqdm

from data_collection.youtube_api import YouTubeAPI
from storage.database import DatabaseHandler
from config.config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    """Orchestrates data collection from YouTube"""
    
    def __init__(self, use_cache: bool = True):
        self.api = YouTubeAPI()
        self.db = DatabaseHandler()
        self.use_cache = use_cache
        
        if use_cache:
            self.api.load_cache()
    
    def collect_from_urls(self, video_urls: List[str], max_comments_per_video: int = None) -> pd.DataFrame:
        """
        Collect data from list of video URLs
        
        Args:
            video_urls: List of YouTube video URLs or IDs
            max_comments_per_video: Maximum comments to collect per video
            
        Returns:
            DataFrame with all collected data
        """
        if max_comments_per_video is None:
            max_comments_per_video = Config.DEFAULT_MAX_COMMENTS
        
        all_comments = []
        all_videos = []
        all_channels = []
        
        logger.info(f"Starting data collection for {len(video_urls)} videos")
        
        for url in tqdm(video_urls, desc="Collecting video data"):
            video_id = self.api._extract_video_id(url)
            
            # Get video info
            video_info = self.api.get_video_info(video_id)
            if not video_info:
                logger.warning(f"Could not fetch video info for {video_id}")
                continue
            
            all_videos.append(video_info)
            
            # Get channel info
            channel_info = self.api.get_channel_info(video_info['channel_id'])
            if channel_info:
                all_channels.append(channel_info)
            
            # Get comments
            comments = self.api.get_video_comments(video_id, max_comments_per_video)
            
            # Enrich comments with video and channel data
            for comment in comments:
                comment['video_title'] = video_info['title']
                comment['channel_title'] = video_info['channel_title']
                if channel_info:
                    comment['channel_created_at'] = channel_info['published_at']
                    comment['channel_subscriber_count'] = channel_info['subscriber_count']
            
            all_comments.extend(comments)
        
        # Save to database
        self.db.save_comments(all_comments)
        self.db.save_videos(all_videos)
        self.db.save_channels(all_channels)
        
        # Save cache
        if self.use_cache:
            self.api.save_cache()
        
        logger.info(f"Collected {len(all_comments)} comments from {len(all_videos)} videos")
        
        return pd.DataFrame(all_comments)
    
    def collect_political_content(self, max_videos_per_channel: int = 5, 
                                 max_comments_per_video: int = None) -> pd.DataFrame:
        """
        Collect data from predefined political channels
        
        Args:
            max_videos_per_channel: Number of recent videos per channel
            max_comments_per_video: Maximum comments per video
            
        Returns:
            DataFrame with collected data
        """
        if max_comments_per_video is None:
            max_comments_per_video = Config.DEFAULT_MAX_COMMENTS
        
        all_video_ids = []
        
        logger.info(f"Collecting from {len(Config.POLITICAL_CHANNELS)} political channels")
        
        for channel_id in Config.POLITICAL_CHANNELS:
            video_ids = self.api.get_channel_videos(channel_id, max_videos_per_channel)
            all_video_ids.extend(video_ids)
            logger.info(f"Found {len(video_ids)} videos from channel {channel_id}")
        
        return self.collect_from_urls(all_video_ids, max_comments_per_video)
    
    def collect_by_search(self, queries: List[str], max_videos_per_query: int = 10,
                         max_comments_per_video: int = None) -> pd.DataFrame:
        """
        Collect data by searching for videos
        
        Args:
            queries: List of search queries
            max_videos_per_query: Maximum videos per search query
            max_comments_per_video: Maximum comments per video
            
        Returns:
            DataFrame with collected data
        """
        if max_comments_per_video is None:
            max_comments_per_video = Config.DEFAULT_MAX_COMMENTS
        
        all_video_ids = []
        
        for query in queries:
            video_ids = self.api.search_videos(query, max_videos_per_query)
            all_video_ids.extend(video_ids)
            logger.info(f"Found {len(video_ids)} videos for query: {query}")
        
        return self.collect_from_urls(all_video_ids, max_comments_per_video)
    
    def enrich_with_author_data(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich comments with additional author/channel data
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Enriched DataFrame
        """
        unique_authors = comments_df['author_id'].unique()
        author_data = {}
        
        logger.info(f"Enriching data for {len(unique_authors)} unique authors")
        
        for author_id in tqdm(unique_authors[:100], desc="Fetching author data"):  # Limit for API quota
            if not author_id:
                continue
            
            channel_info = self.api.get_channel_info(author_id)
            if channel_info:
                author_data[author_id] = {
                    'author_channel_created': channel_info['published_at'],
                    'author_subscriber_count': channel_info['subscriber_count'],
                    'author_video_count': channel_info['video_count'],
                    'author_total_views': channel_info['view_count']
                }
        
        # Merge author data
        for col in ['author_channel_created', 'author_subscriber_count', 
                   'author_video_count', 'author_total_views']:
            comments_df[col] = comments_df['author_id'].map(
                lambda x: author_data.get(x, {}).get(col) if x else None
            )
        
        return comments_df
    
    def load_labeled_data(self, filepath: str) -> pd.DataFrame:
        """
        Load pre-labeled bot data
        
        Args:
            filepath: Path to CSV with labeled data
            
        Returns:
            DataFrame with labeled data
        """
        try:
            labeled_df = pd.read_csv(filepath)
            required_columns = ['text', 'is_bot']
            
            if not all(col in labeled_df.columns for col in required_columns):
                logger.error(f"Labeled data must contain columns: {required_columns}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(labeled_df)} labeled examples")
            logger.info(f"Bot ratio: {labeled_df['is_bot'].mean():.2%}")
            
            return labeled_df
        except Exception as e:
            logger.error(f"Error loading labeled data: {e}")
            return pd.DataFrame()
    
    def get_collection_summary(self) -> Dict:
        """Get summary of collected data"""
        comments_count = self.db.get_comments_count()
        videos_count = self.db.get_videos_count()
        channels_count = self.db.get_channels_count()
        
        summary = {
            'total_comments': comments_count,
            'total_videos': videos_count,
            'total_channels': channels_count,
            'unique_authors': self.db.get_unique_authors_count(),
            'api_calls_made': self.api.get_api_stats()['api_calls'],
            'cache_size': self.api.get_api_stats()['cache_size']
        }
        
        return summary
    
    def export_collected_data(self, output_path: str = 'data/processed/collected_data.csv'):
        """Export all collected data to CSV"""
        comments_df = self.db.get_all_comments()
        comments_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(comments_df)} comments to {output_path}")
        return comments_df
