"""
Text feature extraction for detecting similar/templated comments
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import logging
import re
from collections import Counter
import hashlib

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

from config.config import Config

logger = logging.getLogger(__name__)

class TextFeatures:
    """Extract text-based features for bot detection"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon')
            self.sia = SentimentIntensityAnalyzer()
        
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
    def extract_text_similarity_matrix(self, comments_df: pd.DataFrame) -> np.ndarray:
        """
        Create similarity matrix for all comments
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Similarity matrix
        """
        # Clean and prepare text
        texts = comments_df['text'].fillna('').apply(self._clean_text)
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.tfidf.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            # If TF-IDF fails, use simple Levenshtein distance
            n = len(texts)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    sim = 1 - (Levenshtein.distance(texts.iloc[i], texts.iloc[j]) / 
                              max(len(texts.iloc[i]), len(texts.iloc[j]), 1))
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
            np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def detect_template_comments(self, comments_df: pd.DataFrame, 
                                threshold: float = None) -> Dict[str, float]:
        """
        Detect comments that follow templates or are very similar
        
        Args:
            comments_df: DataFrame with comments
            threshold: Similarity threshold
            
        Returns:
            Dictionary mapping author_id to template score
        """
        if threshold is None:
            threshold = Config.SIMILARITY_THRESHOLD
        
        template_scores = {}
        similarity_matrix = self.extract_text_similarity_matrix(comments_df)
        
        for idx, author_id in enumerate(comments_df['author_id']):
            # Get similarities to other comments
            similarities = similarity_matrix[idx]
            # Exclude self-similarity
            other_similarities = np.concatenate([similarities[:idx], similarities[idx+1:]])
            
            if len(other_similarities) > 0:
                # Calculate template score based on high similarities
                high_similarities = other_similarities[other_similarities > threshold]
                template_score = len(high_similarities) / len(other_similarities)
            else:
                template_score = 0.0
            
            if author_id not in template_scores:
                template_scores[author_id] = []
            template_scores[author_id].append(template_score)
        
        # Average scores per author
        for author_id in template_scores:
            template_scores[author_id] = np.mean(template_scores[author_id])
        
        return template_scores
    
    def extract_linguistic_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract linguistic features from comments
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with linguistic features per author
        """
        linguistic_features = []
        
        for author_id, group in comments_df.groupby('author_id'):
            features = {'author_id': author_id}
            
            texts = group['text'].fillna('')
            all_text = ' '.join(texts)
            
            # Basic text statistics
            features['avg_comment_length'] = texts.str.len().mean()
            features['std_comment_length'] = texts.str.len().std()
            features['total_comments'] = len(texts)
            
            # Vocabulary diversity
            words = word_tokenize(all_text.lower())
            words_no_stop = [w for w in words if w.isalnum() and w not in self.stop_words]
            features['vocabulary_size'] = len(set(words_no_stop))
            features['vocabulary_richness'] = features['vocabulary_size'] / max(len(words_no_stop), 1)
            
            # Readability scores
            if len(all_text) > 10:
                features['flesch_reading_ease'] = textstat.flesch_reading_ease(all_text)
                features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(all_text)
            else:
                features['flesch_reading_ease'] = 0
                features['flesch_kincaid_grade'] = 0
            
            # Sentiment analysis
            sentiments = texts.apply(lambda x: self.sia.polarity_scores(x))
            features['avg_sentiment_compound'] = sentiments.apply(lambda x: x['compound']).mean()
            features['std_sentiment_compound'] = sentiments.apply(lambda x: x['compound']).std()
            features['avg_sentiment_positive'] = sentiments.apply(lambda x: x['pos']).mean()
            features['avg_sentiment_negative'] = sentiments.apply(lambda x: x['neg']).mean()
            features['avg_sentiment_neutral'] = sentiments.apply(lambda x: x['neu']).mean()
            
            # Special character usage
            features['exclamation_ratio'] = texts.str.count('!').sum() / max(len(all_text), 1)
            features['question_ratio'] = texts.str.count(r'\?').sum() / max(len(all_text), 1)
            features['caps_ratio'] = sum(1 for c in all_text if c.isupper()) / max(len(all_text), 1)
            features['emoji_count'] = self._count_emojis(all_text)
            features['url_count'] = texts.str.count(r'http[s]?://').sum()
            
            # Repetition detection
            features['repeated_words_ratio'] = self._calculate_repetition_ratio(words_no_stop)
            features['repeated_phrases_count'] = self._count_repeated_phrases(texts)
            
            linguistic_features.append(features)
        
        return pd.DataFrame(linguistic_features)
    
    def detect_spam_patterns(self, comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect spam-like patterns in comments
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping author_id to spam score
        """
        spam_scores = {}
        spam_indicators = [
            r'\b(?:click|subscribe|follow|check out|visit)\b',
            r'\b(?:free|win|prize|giveaway|discount)\b',
            r'\b(?:bit\.ly|tinyurl|goo\.gl|shorturl)\b',
            r'[A-Z]{5,}',  # Excessive caps
            r'(.)\1{4,}',  # Character repetition
            r'[\$€£¥₹]{1,}[\d,]+',  # Money mentions
            r'\b(?:make money|earn cash|work from home)\b'
        ]
        
        for author_id, group in comments_df.groupby('author_id'):
            spam_count = 0
            total_comments = len(group)
            
            for text in group['text'].fillna(''):
                text_lower = text.lower()
                for pattern in spam_indicators:
                    if re.search(pattern, text_lower):
                        spam_count += 1
                        break
            
            spam_scores[author_id] = spam_count / max(total_comments, 1)
        
        return spam_scores
    
    def find_duplicate_comments(self, comments_df: pd.DataFrame) -> List[Set[str]]:
        """
        Find groups of nearly identical comments
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            List of sets containing comment IDs that are duplicates
        """
        # Hash comments for exact duplicates
        comments_df['text_hash'] = comments_df['text'].apply(
            lambda x: hashlib.md5(self._clean_text(x).encode()).hexdigest()
        )
        
        duplicate_groups = []
        
        # Group by hash
        for hash_val, group in comments_df.groupby('text_hash'):
            if len(group) > 1:
                duplicate_groups.append(set(group['comment_id'].values))
        
        # Also check for near-duplicates using Levenshtein distance
        processed = set()
        for idx1, row1 in comments_df.iterrows():
            if row1['comment_id'] in processed:
                continue
            
            similar_group = {row1['comment_id']}
            
            for idx2, row2 in comments_df.iterrows():
                if idx1 >= idx2 or row2['comment_id'] in processed:
                    continue
                
                similarity = 1 - (Levenshtein.distance(row1['text'], row2['text']) / 
                                max(len(row1['text']), len(row2['text']), 1))
                
                if similarity > 0.9:  # 90% similar
                    similar_group.add(row2['comment_id'])
            
            if len(similar_group) > 1:
                duplicate_groups.append(similar_group)
                processed.update(similar_group)
        
        return duplicate_groups
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ''
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove mentions
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        # Remove hashtags
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower().strip()
    
    def _count_emojis(self, text: str) -> int:
        """Count emoji occurrences in text"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "]+", 
            flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def _calculate_repetition_ratio(self, words: List[str]) -> float:
        """Calculate ratio of repeated words"""
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        repeated = sum(1 for count in word_counts.values() if count > 1)
        return repeated / len(word_counts)
    
    def _count_repeated_phrases(self, texts: pd.Series) -> int:
        """Count repeated phrases across comments"""
        all_phrases = []
        
        for text in texts:
            words = word_tokenize(text.lower())
            # Extract 2-grams and 3-grams
            for n in [2, 3]:
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    all_phrases.append(phrase)
        
        phrase_counts = Counter(all_phrases)
        repeated = sum(1 for count in phrase_counts.values() if count > 2)
        
        return repeated
    
    def calculate_comment_diversity(self, comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate diversity score for each author's comments
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping author_id to diversity score (0-1, higher = more diverse)
        """
        diversity_scores = {}
        
        for author_id, group in comments_df.groupby('author_id'):
            if len(group) < 2:
                diversity_scores[author_id] = 1.0
                continue
            
            texts = group['text'].fillna('').apply(self._clean_text)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    sim = 1 - (Levenshtein.distance(texts.iloc[i], texts.iloc[j]) / 
                              max(len(texts.iloc[i]), len(texts.iloc[j]), 1))
                    similarities.append(sim)
            
            if similarities:
                # Diversity is inverse of average similarity
                avg_similarity = np.mean(similarities)
                diversity_scores[author_id] = 1 - avg_similarity
            else:
                diversity_scores[author_id] = 1.0
        
        return diversity_scores
