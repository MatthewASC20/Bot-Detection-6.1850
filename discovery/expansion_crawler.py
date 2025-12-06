"""
Expansion Crawler - Follows suspicious accounts to discover full botnets
"""
import logging
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExpansionCrawler:
    """
    Crawls outward from suspicious accounts to discover connected bot networks.
    
    Strategy:
    1. Start with seed accounts (high bot probability)
    2. Find all videos those accounts commented on
    3. Find all OTHER accounts that commented on those videos
    4. Score those accounts for bot probability
    5. Add high-probability accounts to the network
    6. Repeat expansion up to N levels
    """
    
    def __init__(self, db, youtube_api=None):
        self.db = db
        self.youtube_api = youtube_api  # Optional - for fetching new data
        self.discovered_accounts = set()
        self.discovered_videos = set()
        self.expansion_history = []
        
    def expand_from_seeds(self, seed_author_ids: List[str], 
                          max_depth: int = 3,
                          min_bot_probability: float = 0.5,
                          max_accounts: int = 1000) -> Dict:
        """
        Expand network starting from seed accounts.
        
        Args:
            seed_author_ids: Initial suspicious accounts
            max_depth: How many levels to expand
            min_bot_probability: Threshold for including accounts
            max_accounts: Maximum accounts to discover
            
        Returns:
            Dictionary with discovered network
        """
        logger.info(f"Starting expansion from {len(seed_author_ids)} seed accounts")
        
        # Initialize
        current_level = set(seed_author_ids)
        all_accounts = set(seed_author_ids)
        all_videos = set()
        connections = []  # (author1, author2, video_id, strength)
        
        for depth in range(max_depth):
            if len(all_accounts) >= max_accounts:
                logger.info(f"Reached max accounts limit ({max_accounts})")
                break
                
            logger.info(f"Expansion level {depth + 1}: Processing {len(current_level)} accounts")
            
            # Find videos these accounts commented on
            level_videos = self._find_videos_for_accounts(list(current_level))
            new_videos = level_videos - all_videos
            all_videos.update(new_videos)
            
            logger.info(f"  Found {len(new_videos)} new videos")
            
            if not new_videos:
                break
            
            # Find other accounts on those videos
            new_accounts, new_connections = self._find_accounts_on_videos(
                list(new_videos), 
                exclude=all_accounts,
                min_bot_probability=min_bot_probability
            )
            
            connections.extend(new_connections)
            
            logger.info(f"  Found {len(new_accounts)} new suspicious accounts")
            
            # Prepare next level
            current_level = new_accounts
            all_accounts.update(new_accounts)
            
            self.expansion_history.append({
                'depth': depth + 1,
                'accounts_processed': len(current_level),
                'new_videos': len(new_videos),
                'new_accounts': len(new_accounts),
                'total_accounts': len(all_accounts)
            })
        
        # Build result
        result = {
            'seed_accounts': seed_author_ids,
            'total_accounts_discovered': len(all_accounts),
            'total_videos_involved': len(all_videos),
            'accounts': list(all_accounts),
            'videos': list(all_videos),
            'connections': connections,
            'expansion_history': self.expansion_history,
            'depth_reached': len(self.expansion_history)
        }
        
        logger.info(f"Expansion complete: {len(all_accounts)} accounts, {len(all_videos)} videos")
        
        return result
    
    def _find_videos_for_accounts(self, author_ids: List[str]) -> Set[str]:
        """Find all videos that these accounts commented on"""
        videos = set()
        
        # Query database
        comments_df = self.db.get_comments_by_author_ids(author_ids)
        if comments_df is not None and not comments_df.empty:
            videos.update(comments_df['video_id'].unique())
        
        # Optionally fetch from YouTube API
        if self.youtube_api and len(videos) < 10:
            # Could implement channel activity lookup here
            pass
        
        return videos
    
    def _find_accounts_on_videos(self, video_ids: List[str], 
                                  exclude: Set[str],
                                  min_bot_probability: float) -> tuple:
        """Find other suspicious accounts on these videos"""
        new_accounts = set()
        connections = []
        
        for video_id in video_ids:
            # Get all comments on this video
            comments_df = self.db.get_comments_by_video(video_id)
            
            if comments_df is None or comments_df.empty:
                continue
            
            video_authors = set(comments_df['author_id'].unique())
            
            for author_id in video_authors:
                if author_id in exclude:
                    continue
                
                # Check author's bot score
                profile = self.db.get_author_profile(author_id)
                if profile:
                    bot_score = profile.get('avg_bot_score', 0)
                    if bot_score >= min_bot_probability:
                        new_accounts.add(author_id)
                        
                        # Record connections
                        for other_author in video_authors:
                            if other_author != author_id and other_author in exclude:
                                connections.append({
                                    'source': author_id,
                                    'target': other_author,
                                    'video_id': video_id,
                                    'type': 'co_occurrence'
                                })
                else:
                    # Unknown author - add if they appear frequently
                    author_comments = comments_df[comments_df['author_id'] == author_id]
                    if len(author_comments) >= 3:  # Multiple comments
                        new_accounts.add(author_id)
        
        return new_accounts, connections
    
    def crawl_author_history(self, author_id: str) -> Dict:
        """
        Deep crawl of a single author's full history.
        
        Returns all videos they've commented on and when.
        """
        result = {
            'author_id': author_id,
            'videos': [],
            'activity_timeline': [],
            'co_commenters': defaultdict(int)
        }
        
        # Get from database
        comments = self.db.get_comments_by_author(author_id)
        
        if comments is None or comments.empty:
            return result
        
        result['total_comments'] = len(comments)
        result['videos'] = comments['video_id'].unique().tolist()
        result['first_seen'] = comments['published_at'].min()
        result['last_seen'] = comments['published_at'].max()
        
        # Build activity timeline
        comments['date'] = pd.to_datetime(comments['published_at']).dt.date
        daily_activity = comments.groupby('date').size().to_dict()
        result['activity_timeline'] = [
            {'date': str(d), 'count': c} for d, c in sorted(daily_activity.items())
        ]
        
        # Find co-commenters
        for video_id in result['videos']:
            video_comments = self.db.get_comments_by_video(video_id)
            if video_comments is not None:
                for other_author in video_comments['author_id'].unique():
                    if other_author != author_id:
                        result['co_commenters'][other_author] += 1
        
        # Convert to list sorted by frequency
        result['top_co_commenters'] = sorted(
            [{'author_id': k, 'shared_videos': v} for k, v in result['co_commenters'].items()],
            key=lambda x: x['shared_videos'],
            reverse=True
        )[:50]
        
        del result['co_commenters']  # Remove raw dict
        
        return result
    
    def find_campaign_targets(self, author_ids: List[str]) -> Dict:
        """
        Analyze what content a group of accounts is targeting.
        
        Returns channels and topics being targeted.
        """
        targets = {
            'channels': defaultdict(lambda: {'count': 0, 'videos': []}),
            'videos': defaultdict(lambda: {'count': 0, 'authors': []}),
            'timeline': []
        }
        
        for author_id in author_ids:
            comments = self.db.get_comments_by_author(author_id)
            if comments is None or comments.empty:
                continue
            
            for _, row in comments.iterrows():
                video_id = row.get('video_id')
                channel = row.get('channel_title', 'Unknown')
                
                targets['channels'][channel]['count'] += 1
                if video_id not in targets['channels'][channel]['videos']:
                    targets['channels'][channel]['videos'].append(video_id)
                
                targets['videos'][video_id]['count'] += 1
                if author_id not in targets['videos'][video_id]['authors']:
                    targets['videos'][video_id]['authors'].append(author_id)
        
        # Convert to sorted lists
        result = {
            'top_targeted_channels': sorted(
                [{'channel': k, 'comment_count': v['count'], 'video_count': len(v['videos'])} 
                 for k, v in targets['channels'].items()],
                key=lambda x: x['comment_count'],
                reverse=True
            )[:20],
            'top_targeted_videos': sorted(
                [{'video_id': k, 'bot_comments': v['count'], 'bot_accounts': len(v['authors'])} 
                 for k, v in targets['videos'].items()],
                key=lambda x: x['bot_comments'],
                reverse=True
            )[:20]
        }
        
        return result