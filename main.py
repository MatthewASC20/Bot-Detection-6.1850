#!/usr/bin/env python3
"""
YouTube Botnet Detector - Main Execution Script
Educational tool for detecting coordinated bot networks in YouTube comments

Implements BotBuster algorithm for detecting coordinated misinformation botnets.
Based on: "Detecting Coordinated Misinformation Botnets in Social Media"
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data_collection.data_collector import DataCollector
from features.temporal_features import TemporalFeatures
from features.text_features import TextFeatures
from features.network_features import NetworkFeatures
from features.behavioral_features import BehavioralFeatures
from features.semantic_features import SemanticFeatures
from detection.clustering import ClusteringDetector
from detection.botbuster_detector import BotBusterDetector
from visualization.network_viz import NetworkVisualizer
from storage.database import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler('botnet_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class YouTubeBotnetDetector:
    """Main orchestrator for bot detection pipeline"""
    
    def __init__(self, use_cache: bool = True, method: str = 'botbuster'):
        """
        Initialize the detector.
        
        Args:
            use_cache: Whether to use API response caching
            method: Detection method - 'botbuster' (new) or 'original' (legacy)
        """
        self.collector = DataCollector(use_cache=use_cache)
        self.text_features = TextFeatures()
        self.detector = ClusteringDetector()
        self.botbuster_detector = BotBusterDetector()
        self.visualizer = NetworkVisualizer()
        self.db = DatabaseHandler()
        self.method = method
        
        logger.info(f"Initialized detector with method: {method}")
        
    def collect_data(self, mode: str, urls: List[str] = None, 
                    max_comments: int = None) -> pd.DataFrame:
        """
        Collect data based on mode
        
        Args:
            mode: Collection mode ('urls', 'political', 'search')
            urls: List of video URLs (for 'urls' mode)
            max_comments: Maximum comments per video
            
        Returns:
            DataFrame with collected comments
        """
        logger.info(f"Starting data collection in mode: {mode}")
        
        if mode == 'urls' and urls:
            return self.collector.collect_from_urls(urls, max_comments)
        elif mode == 'political':
            return self.collector.collect_political_content(max_comments_per_video=max_comments)
        elif mode == 'search':
            # Search for political content
            queries = Config.POLITICAL_KEYWORDS[:5]  # Use first 5 keywords
            return self.collector.collect_by_search(queries, max_comments_per_video=max_comments)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def extract_all_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from comments
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with all features
        """
        logger.info("Extracting features...")
        
        # Temporal features
        logger.info("Extracting temporal features...")
        burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
        temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
        temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
        
        # Text features
        logger.info("Extracting text features...")
        template_scores = self.text_features.detect_template_comments(comments_df)
        spam_scores = self.text_features.detect_spam_patterns(comments_df)
        diversity_scores = self.text_features.calculate_comment_diversity(comments_df)
        linguistic_df = self.text_features.extract_linguistic_features(comments_df)
        
        linguistic_df['template_score'] = linguistic_df['author_id'].map(template_scores)
        linguistic_df['spam_score'] = linguistic_df['author_id'].map(spam_scores)
        linguistic_df['diversity_score'] = linguistic_df['author_id'].map(diversity_scores)
        
        # Network features
        logger.info("Extracting network features...")
        network_df = NetworkFeatures.calculate_author_network_features(comments_df)
        
        # Behavioral features
        logger.info("Extracting behavioral features...")
        behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
        
        # Merge all features
        logger.info("Merging features...")
        features_df = temporal_df
        
        # Merge other dataframes
        for df in [linguistic_df, network_df, behavioral_df]:
            features_df = features_df.merge(df, on='author_id', how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Extracted {features_df.shape[1]} features for {len(features_df)} authors")
        
        return features_df
    
    def detect_bots(self, features_df: pd.DataFrame, 
                   comments_df: pd.DataFrame = None,
                   labeled_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Detect bots using the selected method
        
        Args:
            features_df: DataFrame with features (used for original method)
            comments_df: DataFrame with raw comments (used for BotBuster method)
            labeled_data: Optional labeled data for semi-supervised learning
            
        Returns:
            DataFrame with detection results
        """
        logger.info(f"Starting bot detection using method: {self.method}")
        
        if self.method == 'botbuster':
            if comments_df is None:
                raise ValueError("BotBuster method requires comments_df parameter")
            
            # Use BotBuster algorithm (semantic + temporal coordination)
            results_df = self.botbuster_detector.detect_bots(comments_df)
            
            # Ensure consistent column names with original method
            if 'cluster_confidence' not in results_df.columns:
                results_df['cluster_confidence'] = 1.0
            if 'cluster_bot_probability' not in results_df.columns:
                results_df['cluster_bot_probability'] = results_df['final_bot_probability']
            if 'individual_bot_probability' not in results_df.columns:
                results_df['individual_bot_probability'] = results_df['avg_comment_bot_prob']
                
        else:
            # Use original unsupervised clustering method
            results_df = self.detector.detect_bots(features_df)
        
        # If labeled data is provided, use it to refine results
        if labeled_data is not None and len(labeled_data) > 0:
            logger.info("Refining results with labeled data...")
            # This is a placeholder for semi-supervised refinement
            # In a full implementation, you would train a classifier on labeled data
            # and use it to adjust the unsupervised results
        
        return results_df
    
    def visualize_results(self, detection_results: pd.DataFrame, 
                         comments_df: pd.DataFrame) -> Dict[str, str]:
        """
        Create visualizations of detection results
        
        Args:
            detection_results: DataFrame with detection results
            comments_df: Original comments DataFrame
            
        Returns:
            Dictionary with paths to visualization files
        """
        logger.info("Creating visualizations...")
        
        visualization_files = {}
        
        # Build network graph
        G = NetworkFeatures.build_co_occurrence_network(comments_df)
        
        # Create bot scores dictionary
        bot_scores = detection_results.set_index('author_id')['final_bot_probability'].to_dict()
        
        # Detect communities
        communities = NetworkFeatures.detect_communities(G) if G.number_of_nodes() > 0 else {}
        
        # Create network visualization
        if G.number_of_nodes() > 0:
            network_path = self.visualizer.visualize_bot_network(
                G, bot_scores, communities,
                title="YouTube Comment Bot Network",
                filename="bot_network.html"
            )
            visualization_files['network'] = network_path
        
        # Create cluster comparison
        cluster_path = self.visualizer.visualize_cluster_comparison(
            detection_results,
            filename="cluster_analysis.html"
        )
        visualization_files['clusters'] = cluster_path
        
        # Create temporal patterns visualization
        temporal_path = self.visualizer.visualize_temporal_patterns(
            comments_df, detection_results,
            filename="temporal_patterns.html"
        )
        visualization_files['temporal'] = temporal_path
        
        return visualization_files
    
    def save_results(self, detection_results: pd.DataFrame, 
                    summary: Dict) -> str:
        """
        Save detection results and summary
        
        Args:
            detection_results: DataFrame with detection results
            summary: Summary statistics dictionary
            
        Returns:
            Path to results file
        """
        # Prepare results for database (convert lists to JSON strings)
        db_results = detection_results.copy()
        if 'comment_ids' in db_results.columns:
            db_results['comment_ids'] = db_results['comment_ids'].apply(json.dumps)
        
        # Save to database
        self.db.save_detection_results(db_results)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(Config.REPORTS_DIR, f"bot_detection_results_{timestamp}.csv")
        detection_results.to_csv(csv_path, index=False)
        
        # Save summary JSON
        json_path = os.path.join(Config.REPORTS_DIR, f"detection_summary_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {csv_path}")
        logger.info(f"Summary saved to {json_path}")
        
        return csv_path
    
    def run_full_pipeline(self, mode: str, urls: List[str] = None,
                         max_comments: int = None,
                         labeled_data_path: str = None) -> Dict:
        """
        Run the complete bot detection pipeline
        
        Args:
            mode: Data collection mode
            urls: Video URLs (for 'urls' mode)
            max_comments: Maximum comments per video
            labeled_data_path: Path to labeled data CSV
            
        Returns:
            Dictionary with results and file paths
        """
        try:
            # Step 1: Collect data
            logger.info("=" * 50)
            logger.info("STEP 1: DATA COLLECTION")
            logger.info("=" * 50)
            
            comments_df = self.collect_data(mode, urls, max_comments)
            
            if comments_df.empty:
                logger.error("No data collected. Exiting.")
                return {}
            
            # Step 2: Extract features (for original method or comparison)
            logger.info("=" * 50)
            logger.info("STEP 2: FEATURE EXTRACTION")
            logger.info("=" * 50)
            
            features_df = None
            if self.method == 'original' or self.method == 'both':
                features_df = self.extract_all_features(comments_df)
            else:
                logger.info("Skipping traditional feature extraction (using BotBuster)")
                # Create minimal features_df for compatibility
                features_df = comments_df[['author_id']].drop_duplicates()
            
            # Step 3: Load labeled data (if provided)
            labeled_data = None
            if labeled_data_path:
                logger.info(f"Loading labeled data from {labeled_data_path}")
                labeled_data = self.collector.load_labeled_data(labeled_data_path)
            
            # Step 4: Detect bots
            logger.info("=" * 50)
            logger.info(f"STEP 3: BOT DETECTION ({self.method.upper()} METHOD)")
            logger.info("=" * 50)
            
            detection_results = self.detect_bots(features_df, comments_df, labeled_data)
            
            # Step 5: Visualize results
            logger.info("=" * 50)
            logger.info("STEP 4: VISUALIZATION")
            logger.info("=" * 50)
            
            visualization_files = self.visualize_results(detection_results, comments_df)
            
            # Step 6: Generate summary
            summary = self.visualizer.create_summary_report(
                detection_results, comments_df
            )
            
            # Add method info to summary
            summary['detection_method'] = self.method
            
            # Add BotBuster-specific summary info
            if self.method == 'botbuster':
                comment_probs = self.botbuster_detector.get_comment_probabilities()
                if comment_probs:
                    summary['comment_level_stats'] = {
                        'total_comments_analyzed': len(comment_probs),
                        'avg_comment_bot_prob': np.mean(list(comment_probs.values())),
                        'max_comment_bot_prob': max(comment_probs.values()),
                        'high_prob_comments': sum(1 for p in comment_probs.values() if p > 0.7)
                    }
            
            # Step 7: Save results
            logger.info("=" * 50)
            logger.info("STEP 5: SAVING RESULTS")
            logger.info("=" * 50)
            
            results_file = self.save_results(detection_results, summary)
            
            # Save comment-level probabilities for BotBuster
            if self.method == 'botbuster':
                self._save_comment_probabilities(comments_df)
            
            # Print summary
            self.print_summary(summary)
            
            return {
                'summary': summary,
                'results_file': results_file,
                'visualizations': visualization_files,
                'detection_results': detection_results,
                'comments_df': comments_df
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _save_comment_probabilities(self, comments_df: pd.DataFrame):
        """Save per-comment bot probabilities to CSV"""
        comment_probs = self.botbuster_detector.get_comment_probabilities()
        
        if not comment_probs:
            return
        
        # Create DataFrame with comment details and probabilities
        prob_df = comments_df[['comment_id', 'video_id', 'text', 'author_id', 'author', 'published_at']].copy()
        prob_df['bot_probability'] = prob_df['comment_id'].map(comment_probs)
        prob_df = prob_df.sort_values('bot_probability', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(Config.REPORTS_DIR, f"comment_bot_probabilities_{timestamp}.csv")
        prob_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved comment-level probabilities to {csv_path}")
    
    def print_summary(self, summary: Dict):
        """Print detection summary to console"""
        print("\n" + "=" * 60)
        method = summary.get('detection_method', 'unknown').upper()
        print(f"BOT DETECTION SUMMARY ({method} METHOD)")
        print("=" * 60)
        print(f"Total Accounts Analyzed: {summary['total_accounts']}")
        print(f"Total Comments Analyzed: {summary['total_comments']}")
        print(f"Clusters Found: {summary['clusters_found']}")
        print(f"Average Bot Probability: {summary['avg_bot_probability']:.2%}")
        
        print("\nClassification Results:")
        for classification, count in summary['classification_counts'].items():
            percentage = (count / summary['total_accounts']) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")
        
        print(f"\nHigh Confidence Bots (>90% probability): {summary['high_confidence_bots']}")
        
        # BotBuster-specific stats
        if 'comment_level_stats' in summary:
            stats = summary['comment_level_stats']
            print("\nComment-Level Analysis (BotBuster):")
            print(f"  Total Comments Analyzed: {stats['total_comments_analyzed']}")
            print(f"  Average Comment Bot Probability: {stats['avg_comment_bot_prob']:.2%}")
            print(f"  Max Comment Bot Probability: {stats['max_comment_bot_prob']:.2%}")
            print(f"  High-Probability Comments (>70%): {stats['high_prob_comments']}")
        
        if summary.get('top_bot_accounts'):
            print("\nTop 5 Suspected Bot Accounts:")
            for i, account in enumerate(summary['top_bot_accounts'][:5], 1):
                print(f"  {i}. {account['author_id'][:30]}... (Probability: {account['final_bot_probability']:.2%})")
        
        print("=" * 60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YouTube Botnet Detector - Educational Tool for Detecting Coordinated Bot Networks"
    )
    
    parser.add_argument(
        '--mode',
        choices=['urls', 'political', 'search'],
        default='political',
        help='Data collection mode'
    )
    
    parser.add_argument(
        '--urls',
        type=str,
        help='Comma-separated list of YouTube video URLs (for urls mode)'
    )
    
    parser.add_argument(
        '--max-comments',
        type=int,
        default=1000,
        help='Maximum comments to collect per video'
    )
    
    parser.add_argument(
        '--labeled-data',
        type=str,
        help='Path to CSV file with labeled bot data'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable API response caching'
    )
    
    parser.add_argument(
        '--clear-db',
        action='store_true',
        help='Clear database before starting'
    )
    
    parser.add_argument(
        '--method',
        choices=['botbuster', 'original', 'both'],
        default='botbuster',
        help='Detection method: botbuster (semantic+temporal coordination), original (clustering-based), or both for comparison'
    )
    
    args = parser.parse_args()
    
    # Parse URLs if provided
    urls = None
    if args.urls:
        urls = [url.strip() for url in args.urls.split(',')]
    
    # Initialize detector with selected method
    detector = YouTubeBotnetDetector(use_cache=not args.no_cache, method=args.method)
    
    # Clear database if requested
    if args.clear_db:
        logger.warning("Clearing database...")
        detector.db.clear_all_data()
    
    # Run pipeline
    results = detector.run_full_pipeline(
        mode=args.mode,
        urls=urls,
        max_comments=args.max_comments,
        labeled_data_path=args.labeled_data
    )
    
    if results:
        print("\n✅ Bot detection completed successfully!")
        print(f"🔬 Detection Method: {args.method.upper()}")
        print(f"📊 Results saved to: {results['results_file']}")
        if results.get('visualizations'):
            print("📈 Visualizations created:")
            for viz_type, path in results['visualizations'].items():
                print(f"   - {viz_type}: {path}")

if __name__ == "__main__":
    main()
