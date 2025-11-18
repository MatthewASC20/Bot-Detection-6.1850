"""
YouTube API wrapper for data collection
"""
import time
import logging
from typing import List, Dict, Optional, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import json
import hashlib

from config.config import Config

logger = logging.getLogger(__name__)

class YouTubeAPI:
    """Wrapper for YouTube Data API v3"""
    
    def __init__(self):
        self.api_key_index = 0
        self.api_calls_count = 0
        self.youtube = self._build_youtube_client()
        self.cache = {}
        
    def _build_youtube_client(self):
        """Build YouTube API client with current API key"""
        api_key = Config.get_api_key(self.api_key_index)
        return build('youtube', 'v3', developerKey=api_key)
    
    def _rotate_api_key(self):
        """Rotate to next API key"""
        self.api_key_index = (self.api_key_index + 1) % len(Config.YOUTUBE_API_KEYS)
        logger.info(f"Rotating to API key index {self.api_key_index}")
        self.youtube = self._build_youtube_client()
    
    def _make_request(self, request_func, *args, **kwargs) -> Optional[Dict]:
        """Make API request with retry logic and key rotation"""
        for attempt in range(Config.API_RETRY_COUNT):
            try:
                self.api_calls_count += 1
                result = request_func(*args, **kwargs).execute()
                return result
            except HttpError as e:
                if e.resp.status == 403:  # Quota exceeded
                    logger.warning(f"Quota exceeded, rotating API key")
                    self._rotate_api_key()
                    time.sleep(Config.API_RETRY_DELAY)
                elif e.resp.status == 404:
                    logger.error(f"Resource not found: {e}")
                    return None
                else:
                    logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                    if attempt < Config.API_RETRY_COUNT - 1:
                        time.sleep(Config.API_RETRY_DELAY * (attempt + 1))
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < Config.API_RETRY_COUNT - 1:
                    time.sleep(Config.API_RETRY_DELAY)
        
        return None
    
    def get_video_comments(self, video_id: str, max_comments: int = None) -> List[Dict]:
        """
        Fetch comments for a video
        
        Args:
            video_id: YouTube video ID
            max_comments: Maximum number of comments to fetch
            
        Returns:
            List of comment dictionaries
        """
        if max_comments is None:
            max_comments = Config.DEFAULT_MAX_COMMENTS
            
        comments = []
        next_page_token = None
        
        # Check cache
        cache_key = f"comments_{video_id}_{max_comments}"
        if cache_key in self.cache:
            logger.info(f"Using cached comments for video {video_id}")
            return self.cache[cache_key]
        
        logger.info(f"Fetching comments for video {video_id}")
        
        while len(comments) < max_comments:
            request = self.youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=min(Config.MAX_RESULTS_PER_PAGE, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat='plainText'
            )
            
            response = self._make_request(lambda: request)
            if not response:
                break
            
            for item in response.get('items', []):
                comment_data = self._parse_comment(item)
                comments.append(comment_data)
                
                # Include replies
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_data = self._parse_comment({'snippet': reply['snippet']}, is_reply=True)
                        reply_data['parent_id'] = comment_data['comment_id']
                        comments.append(reply_data)
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            
            # Rate limiting
            time.sleep(0.5)
        
        # Cache results
        self.cache[cache_key] = comments[:max_comments]
        
        logger.info(f"Fetched {len(comments)} comments for video {video_id}")
        return comments[:max_comments]
    
    def _parse_comment(self, item: Dict, is_reply: bool = False) -> Dict:
        """Parse comment data from API response"""
        snippet = item['snippet']['topLevelComment']['snippet'] if not is_reply else item['snippet']
        
        return {
            'comment_id': hashlib.md5(
                f"{snippet.get('textDisplay', '')}_{snippet.get('authorDisplayName', '')}_{snippet.get('publishedAt', '')}".encode()
            ).hexdigest(),
            'video_id': snippet.get('videoId', ''),
            'text': snippet.get('textDisplay', ''),
            'author': snippet.get('authorDisplayName', ''),
            'author_id': snippet.get('authorChannelId', {}).get('value', ''),
            'published_at': snippet.get('publishedAt', ''),
            'updated_at': snippet.get('updatedAt', ''),
            'like_count': snippet.get('likeCount', 0),
            'is_reply': is_reply,
            'parent_id': None
        }
    
    def get_channel_info(self, channel_id: str) -> Optional[Dict]:
        """
        Fetch channel information
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Channel information dictionary
        """
        cache_key = f"channel_{channel_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        request = self.youtube.channels().list(
            part='snippet,statistics,brandingSettings',
            id=channel_id
        )
        
        response = self._make_request(lambda: request)
        if not response or not response.get('items'):
            return None
        
        channel = response['items'][0]
        channel_info = {
            'channel_id': channel_id,
            'title': channel['snippet'].get('title', ''),
            'description': channel['snippet'].get('description', ''),
            'published_at': channel['snippet'].get('publishedAt', ''),
            'subscriber_count': int(channel['statistics'].get('subscriberCount', 0)),
            'video_count': int(channel['statistics'].get('videoCount', 0)),
            'view_count': int(channel['statistics'].get('viewCount', 0)),
            'country': channel['snippet'].get('country', ''),
            'custom_url': channel['snippet'].get('customUrl', '')
        }
        
        self.cache[cache_key] = channel_info
        return channel_info
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """
        Fetch video information
        
        Args:
            video_id: YouTube video ID or URL
            
        Returns:
            Video information dictionary
        """
        # Extract video ID from URL if necessary
        if 'youtube.com' in video_id or 'youtu.be' in video_id:
            video_id = self._extract_video_id(video_id)
        
        cache_key = f"video_{video_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        request = self.youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        )
        
        response = self._make_request(lambda: request)
        if not response or not response.get('items'):
            return None
        
        video = response['items'][0]
        video_info = {
            'video_id': video_id,
            'title': video['snippet'].get('title', ''),
            'description': video['snippet'].get('description', ''),
            'channel_id': video['snippet'].get('channelId', ''),
            'channel_title': video['snippet'].get('channelTitle', ''),
            'published_at': video['snippet'].get('publishedAt', ''),
            'duration': video['contentDetails'].get('duration', ''),
            'view_count': int(video['statistics'].get('viewCount', 0)),
            'like_count': int(video['statistics'].get('likeCount', 0)),
            'comment_count': int(video['statistics'].get('commentCount', 0)),
            'tags': video['snippet'].get('tags', [])
        }
        
        self.cache[cache_key] = video_info
        return video_info
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        if 'v=' in url:
            return url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        return url
    
    def search_videos(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search for videos by query
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of video IDs
        """
        request = self.youtube.search().list(
            part='id',
            q=query,
            type='video',
            maxResults=max_results,
            order='relevance'
        )
        
        response = self._make_request(lambda: request)
        if not response:
            return []
        
        return [item['id']['videoId'] for item in response.get('items', [])]
    
    def get_channel_videos(self, channel_id: str, max_results: int = 10) -> List[str]:
        """
        Get recent videos from a channel
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos
            
        Returns:
            List of video IDs
        """
        request = self.youtube.search().list(
            part='id',
            channelId=channel_id,
            type='video',
            maxResults=max_results,
            order='date'
        )
        
        response = self._make_request(lambda: request)
        if not response:
            return []
        
        return [item['id']['videoId'] for item in response.get('items', [])]
    
    def save_cache(self, filepath: str = 'data/raw/api_cache.json'):
        """Save cache to file"""
        with open(filepath, 'w') as f:
            json.dump(self.cache, f, indent=2, default=str)
        logger.info(f"Saved cache with {len(self.cache)} items to {filepath}")
    
    def load_cache(self, filepath: str = 'data/raw/api_cache.json'):
        """Load cache from file"""
        try:
            with open(filepath, 'r') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded cache with {len(self.cache)} items from {filepath}")
        except FileNotFoundError:
            logger.info("No cache file found, starting with empty cache")
    
    def get_api_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'api_calls': self.api_calls_count,
            'current_key_index': self.api_key_index,
            'cache_size': len(self.cache),
            'estimated_quota_used': self.api_calls_count  # Simplified estimation
        }
