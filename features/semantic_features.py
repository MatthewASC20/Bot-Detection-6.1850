"""
Semantic feature extraction for bot detection
Combines OpenAI embeddings (when available) with local fallback methods

Primary approach: OpenAI embeddings (BotBuster paper methodology)
Fallback: Local TF-IDF, LDA, and rule-based analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import logging
import re
import time
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

from config.config import Config

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI not available - will use local semantic methods")


class SemanticFeatures:
    """
    Extract semantic features using OpenAI embeddings (primary) or local methods (fallback).
    
    When OpenAI is available:
    - Uses text-embedding-3-small for high-quality semantic similarity
    - Calculates comment-to-discussion similarity
    - Finds semantically coordinated groups
    
    When OpenAI is NOT available:
    - Falls back to TF-IDF similarity
    - Uses LDA for topic modeling
    - Uses rule-based intent classification
    """
    
    def __init__(self, use_openai: bool = True):
        """
        Initialize SemanticFeatures.
        
        Args:
            use_openai: Whether to use OpenAI (if available and configured)
        """
        self.use_openai = (
            use_openai
            and OPENAI_AVAILABLE
            and hasattr(Config, 'OPENAI_API_KEY')
            and Config.OPENAI_API_KEY
            and Config.OPENAI_API_KEY != "api_key"  # treat placeholder as unset
        )
        
        if self.use_openai:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.model = getattr(Config, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
                logger.info(f"SemanticFeatures initialized with OpenAI ({self.model})")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.use_openai = False
        
        if not self.use_openai:
            logger.info("SemanticFeatures using local methods (no OpenAI)")
        
        # Embedding cache (works for both OpenAI and local)
        self.embedding_cache = {}
        self.batch_size = max(1, getattr(Config, 'EMBEDDING_BATCH_SIZE', 100))
        self.batch_sleep = max(0.0, float(getattr(Config, 'EMBEDDING_BATCH_SLEEP', 0.0)))
        
        # Local method components
        self.tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.lda = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=10
        )
        
        # Intent patterns for rule-based classification
        self.intent_patterns = {
            'promotion': [
                r'\b(check out|visit|subscribe|follow|click|link)\b',
                r'\b(my channel|my profile|my page)\b',
                r'\b(free|giveaway|discount|offer)\b'
            ],
            'engagement_bait': [
                r'\b(like if|comment if|share if)\b',
                r'\b(who else|anyone else|am i the only)\b',
                r'\b(hit like|smash subscribe|ring the bell)\b'
            ],
            'spam': [
                r'\b(earn money|make money|work from home)\b',
                r'\b(click here|tap here|visit now)\b',
                r'(http|www\.)[^\s]+',
                r'\b(winner|congratulations|you won)\b'
            ],
            'generic_praise': [
                r'^(great|awesome|amazing|nice|cool|good|love it)[\s!]*$',
                r'^(first|early|hi|hello)[\s!]*$'
            ],
            'substantive': [
                r'\b(because|therefore|however|although|specifically)\b',
                r'\b(i think|in my opinion|i believe|i disagree)\b'
            ]
        }
    
    def _hash_text(self, text: str) -> str:
        """Create a stable hash for caching/deduping embeddings"""
        return hashlib.sha1(str(text).encode('utf-8')).hexdigest()
    
    # =========================================================================
    # OPENAI EMBEDDING METHODS (Primary - from original)
    # =========================================================================
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for texts using OpenAI API or local TF-IDF fallback.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Clean texts
        cleaned_texts = []
        for text in texts:
            if text is None or pd.isna(text):
                text = ""
            cleaned_texts.append(str(text)[:8000])
        
        if self.use_openai:
            return self._get_openai_embeddings(cleaned_texts)
        else:
            return self._get_tfidf_embeddings(cleaned_texts)
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API with caching and batching"""
        embed_dim = getattr(Config, 'EMBEDDING_DIMENSION', 1536)
        embeddings = []
        batch_size = getattr(self, 'batch_size', getattr(Config, 'EMBEDDING_BATCH_SIZE', 100))
        batch_delay = getattr(self, 'batch_sleep', 0.0)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache
            batch_embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            hashes_to_embed = []
            
            for j, text in enumerate(batch):
                # Skip/zero-fill empty strings to avoid API 400 errors
                if not str(text).strip():
                    zero_embedding = [0.0] * embed_dim
                    batch_embeddings.append((j, zero_embedding))
                    continue

                text_hash = self._hash_text(text)
                if text_hash in self.embedding_cache:
                    batch_embeddings.append((j, self.embedding_cache[text_hash]))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(j)
                    hashes_to_embed.append(text_hash)
            
            # Get embeddings for non-cached texts
            if texts_to_embed:
                try:
                    # Deduplicate within the batch so repeated comments only hit the API once
                    unique_inputs = []
                    unique_hashes = []
                    hash_to_indices = defaultdict(list)
                    for idx, (text, text_hash) in enumerate(zip(texts_to_embed, hashes_to_embed)):
                        if text_hash not in hash_to_indices:
                            unique_inputs.append(text)
                            unique_hashes.append(text_hash)
                        hash_to_indices[text_hash].append(indices_to_embed[idx])

                    response = self.client.embeddings.create(
                        model=self.model,
                        input=unique_inputs
                    )
                    
                    for text_hash, data in zip(unique_hashes, response.data):
                        embedding = data.embedding
                        self.embedding_cache[text_hash] = embedding
                        for j in hash_to_indices[text_hash]:
                            batch_embeddings.append((j, embedding))
                    
                except Exception as e:
                    logger.error(f"Error getting OpenAI embeddings; falling back to zero vectors: {e}")
                    for j in indices_to_embed:
                        batch_embeddings.append((j, [0.0] * embed_dim))
                
                # Optional throttling between batches
                if batch_delay:
                    time.sleep(batch_delay)
            
            # Sort by original index
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        return np.array(embeddings)
    
    def _get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback: Get TF-IDF based embeddings"""
        try:
            # Fit TF-IDF on texts
            tfidf_matrix = self.tfidf.fit_transform(texts)
            return tfidf_matrix.toarray()
        except Exception as e:
            logger.warning(f"TF-IDF embedding failed: {e}")
            return np.zeros((len(texts), 100))
    
    def calculate_pairwise_semantic_similarity(self, comments_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate pairwise semantic similarity matrix for all comments.
        
        Args:
            comments_df: DataFrame with 'text' column
            
        Returns:
            Similarity matrix of shape (n_comments, n_comments)
        """
        logger.info(f"Calculating semantic similarity for {len(comments_df)} comments...")
        
        texts = comments_df['text'].fillna('').tolist()
        embeddings = self.get_embeddings(texts)
        
        if len(embeddings) == 0:
            return np.zeros((len(comments_df), len(comments_df)))
        
        similarity_matrix = cosine_similarity(embeddings)
        
        logger.info(f"Semantic similarity matrix computed: shape {similarity_matrix.shape}")
        return similarity_matrix
    
    def calculate_comment_to_discussion_similarity(self, comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate how similar each comment is to the broader discussion.
        Low similarity = potentially coordinated/off-topic.
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping comment_id to discussion similarity score
        """
        logger.info("Calculating comment-to-discussion similarity...")
        
        discussion_scores = {}
        
        for video_id, group in comments_df.groupby('video_id'):
            if len(group) < 2:
                for comment_id in group['comment_id']:
                    discussion_scores[comment_id] = 0.5
                continue
            
            texts = group['text'].fillna('').tolist()
            embeddings = self.get_embeddings(texts)
            
            if len(embeddings) == 0:
                for comment_id in group['comment_id']:
                    discussion_scores[comment_id] = 0.5
                continue
            
            # Discussion centroid = average of all embeddings
            discussion_centroid = np.mean(embeddings, axis=0, keepdims=True)
            
            # Similarity of each comment to centroid
            for i, (_, row) in enumerate(group.iterrows()):
                comment_embedding = embeddings[i:i+1]
                similarity = cosine_similarity(comment_embedding, discussion_centroid)[0][0]
                discussion_scores[row['comment_id']] = float(similarity)
        
        logger.info(f"Calculated discussion similarity for {len(discussion_scores)} comments")
        return discussion_scores
    
    def calculate_author_semantic_features(self, comments_df: pd.DataFrame,
                                           similarity_matrix: np.ndarray = None) -> pd.DataFrame:
        """
        Calculate semantic features per author based on their comments.
        
        Args:
            comments_df: DataFrame with comments
            similarity_matrix: Pre-computed similarity matrix (optional)
            
        Returns:
            DataFrame with semantic features per author
        """
        logger.info("Calculating author-level semantic features...")
        
        # Compute similarity matrix if not provided
        if similarity_matrix is None:
            similarity_matrix = self.calculate_pairwise_semantic_similarity(comments_df)
        
        comment_ids = comments_df['comment_id'].tolist()
        comment_id_to_idx = {cid: idx for idx, cid in enumerate(comment_ids)}
        
        # Get discussion similarity scores
        discussion_scores = self.calculate_comment_to_discussion_similarity(comments_df)
        
        semantic_features = []
        
        for author_id, group in comments_df.groupby('author_id'):
            features = {'author_id': author_id}
            
            author_indices = [comment_id_to_idx[cid] for cid in group['comment_id'] 
                            if cid in comment_id_to_idx]
            
            if not author_indices:
                features.update({
                    'avg_semantic_similarity_to_others': 0.0,
                    'max_semantic_similarity_to_others': 0.0,
                    'high_similarity_count': 0,
                    'avg_discussion_similarity': 0.5,
                    'semantic_diversity': 1.0
                })
                semantic_features.append(features)
                continue
            
            # Similarities to OTHER authors' comments
            other_similarities = []
            for idx in author_indices:
                for j in range(similarity_matrix.shape[1]):
                    if j not in author_indices:
                        other_similarities.append(similarity_matrix[idx, j])
            
            if other_similarities:
                features['avg_semantic_similarity_to_others'] = float(np.mean(other_similarities))
                features['max_semantic_similarity_to_others'] = float(np.max(other_similarities))
                threshold = getattr(Config, 'SEMANTIC_SIMILARITY_THRESHOLD', 0.85)
                features['high_similarity_count'] = sum(1 for s in other_similarities if s > threshold)
            else:
                features['avg_semantic_similarity_to_others'] = 0.0
                features['max_semantic_similarity_to_others'] = 0.0
                features['high_similarity_count'] = 0
            
            # Discussion similarity
            author_discussion_scores = [discussion_scores.get(cid, 0.5) for cid in group['comment_id']]
            features['avg_discussion_similarity'] = float(np.mean(author_discussion_scores))
            
            # Semantic diversity (how varied are this author's own comments)
            if len(author_indices) > 1:
                self_similarities = []
                for i, idx1 in enumerate(author_indices):
                    for idx2 in author_indices[i+1:]:
                        self_similarities.append(similarity_matrix[idx1, idx2])
                
                if self_similarities:
                    features['semantic_diversity'] = 1.0 - float(np.mean(self_similarities))
                else:
                    features['semantic_diversity'] = 1.0
            else:
                features['semantic_diversity'] = 1.0
            
            semantic_features.append(features)
        
        result_df = pd.DataFrame(semantic_features)
        logger.info(f"Calculated semantic features for {len(result_df)} authors")
        return result_df
    
    def find_semantically_similar_groups(self, comments_df: pd.DataFrame,
                                         similarity_matrix: np.ndarray = None,
                                         threshold: float = None) -> List[Set[str]]:
        """
        Find groups of comments that are semantically very similar
        (potential coordinated messages from different authors).
        
        Args:
            comments_df: DataFrame with comments
            similarity_matrix: Pre-computed similarity matrix (optional)
            threshold: Similarity threshold
            
        Returns:
            List of sets, each containing author_ids in a similar group
        """
        if threshold is None:
            threshold = getattr(Config, 'SEMANTIC_SIMILARITY_THRESHOLD', 0.85)
        
        if similarity_matrix is None:
            similarity_matrix = self.calculate_pairwise_semantic_similarity(comments_df)
        
        n = len(comments_df)
        author_ids = comments_df['author_id'].tolist()
        
        # Find pairs above threshold (different authors only)
        similar_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] > threshold and author_ids[i] != author_ids[j]:
                    similar_pairs.append((i, j))
        
        # Union-find to build groups
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i, j in similar_pairs:
            union(i, j)
        
        # Collect groups
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = set()
            groups[root].add(author_ids[i])
        
        # Filter to groups with multiple authors
        min_size = getattr(Config, 'MIN_CLUSTER_SIZE', 3)
        similar_groups = [g for g in groups.values() if len(g) >= min_size]
        
        logger.info(f"Found {len(similar_groups)} semantically similar groups")
        return similar_groups
    
    def calculate_comment_bot_probability(self, comments_df: pd.DataFrame,
                                          similarity_matrix: np.ndarray = None,
                                          discussion_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate bot probability for each comment based on semantic features.
        
        Args:
            comments_df: DataFrame with comments
            similarity_matrix: Pre-computed similarity matrix (optional)
            discussion_scores: Pre-computed comment-to-discussion similarity (optional)
            
        Returns:
            Dictionary mapping comment_id to bot probability
        """
        logger.info("Calculating per-comment bot probabilities...")
        
        if similarity_matrix is None:
            similarity_matrix = self.calculate_pairwise_semantic_similarity(comments_df)
        
        if discussion_scores is None:
            discussion_scores = self.calculate_comment_to_discussion_similarity(comments_df)
        
        comment_ids = comments_df['comment_id'].tolist()
        author_ids = comments_df['author_id'].tolist()
        
        comment_bot_probs = {}
        
        for i, comment_id in enumerate(comment_ids):
            # Similarities to OTHER authors
            other_author_sims = []
            for j in range(len(comment_ids)):
                if i != j and author_ids[i] != author_ids[j]:
                    other_author_sims.append(similarity_matrix[i, j])
            
            if not other_author_sims:
                semantic_score = 0.0
            else:
                avg_sim = np.mean(other_author_sims)
                max_sim = np.max(other_author_sims)
                semantic_score = 0.4 * avg_sim + 0.6 * max_sim
            
            # Discussion divergence
            discussion_sim = discussion_scores.get(comment_id, 0.5)
            discussion_divergence = abs(discussion_sim - 0.5) * 2
            
            # Combined score
            bot_probability = 0.7 * semantic_score + 0.3 * discussion_divergence
            comment_bot_probs[comment_id] = min(max(bot_probability, 0.0), 1.0)
        
        logger.info(f"Calculated bot probabilities for {len(comment_bot_probs)} comments")
        return comment_bot_probs
    
    # =========================================================================
    # LOCAL FALLBACK METHODS (Additional features when OpenAI unavailable)
    # =========================================================================
    
    def _extract_topics(self, comments_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract topic distributions using LDA (local method)"""
        topic_distributions = {}
        
        try:
            author_texts = comments_df.groupby('author_id')['text'].apply(
                lambda x: ' '.join(x.fillna(''))
            ).to_dict()
            
            if len(author_texts) < 2:
                return topic_distributions
            
            texts = list(author_texts.values())
            author_ids = list(author_texts.keys())
            
            count_matrix = self.count_vectorizer.fit_transform(texts)
            
            if count_matrix.shape[1] < 10:
                return topic_distributions
            
            topic_matrix = self.lda.fit_transform(count_matrix)
            
            for i, author_id in enumerate(author_ids):
                topic_distributions[author_id] = topic_matrix[i]
                
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
        
        return topic_distributions
    
    def _classify_intents(self, comments_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Classify comment intents using rule-based patterns (local method)"""
        intent_scores = {}
        
        for author_id, group in comments_df.groupby('author_id'):
            texts = group['text'].fillna('').tolist()
            author_intents = defaultdict(float)
            
            for text in texts:
                text_lower = text.lower()
                
                for intent_type, patterns in self.intent_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text_lower, re.IGNORECASE):
                            author_intents[intent_type] += 1
                            break
            
            n_comments = max(len(texts), 1)
            for intent_type in author_intents:
                author_intents[intent_type] /= n_comments
            
            intent_scores[author_id] = dict(author_intents)
        
        return intent_scores
    
    def _detect_talking_points(self, comments_df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
        """Detect coordinated talking points (phrases used by 3+ authors)"""
        phrase_authors = defaultdict(set)
        
        for _, row in comments_df.iterrows():
            text = row.get('text', '')
            if pd.isna(text):
                continue
            
            words = str(text).lower().split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if len(phrase) > 10:
                    phrase_authors[phrase].add(row['author_id'])
        
        coordinated_phrases = [
            phrase for phrase, authors in phrase_authors.items()
            if len(authors) >= 3
        ]
        
        talking_point_matches = defaultdict(int)
        for _, row in comments_df.iterrows():
            text = str(row.get('text', '')).lower()
            author_id = row['author_id']
            
            for phrase in coordinated_phrases:
                if phrase in text:
                    talking_point_matches[author_id] += 1
        
        return dict(talking_point_matches), coordinated_phrases
    
    # =========================================================================
    # COMBINED FEATURE EXTRACTION
    # =========================================================================
    
    def extract_semantic_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all semantic features for each author.
        Uses OpenAI embeddings when available, adds local features as supplement.
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with semantic features per author
        """
        logger.info("Extracting semantic features...")
        
        # Get embedding-based features (primary)
        similarity_matrix = self.calculate_pairwise_semantic_similarity(comments_df)
        author_features_df = self.calculate_author_semantic_features(comments_df, similarity_matrix)
        
        # Add local method features (supplement)
        topic_distributions = self._extract_topics(comments_df)
        intent_scores = self._classify_intents(comments_df)
        talking_point_matches, _ = self._detect_talking_points(comments_df)
        
        # Merge additional features
        additional_features = []
        for author_id in author_features_df['author_id']:
            features = {'author_id': author_id}
            
            # Topic features
            if author_id in topic_distributions:
                dist = topic_distributions[author_id]
                features['dominant_topic'] = int(np.argmax(dist))
                features['topic_concentration'] = float(np.max(dist))
            else:
                features['dominant_topic'] = -1
                features['topic_concentration'] = 0.0
            
            # Intent features
            author_intents = intent_scores.get(author_id, {})
            bot_intents = ['promotion', 'engagement_bait', 'spam', 'generic_praise']
            human_intents = ['substantive']
            
            bot_score = sum(author_intents.get(i, 0) for i in bot_intents)
            human_score = sum(author_intents.get(i, 0) for i in human_intents)
            total = max(bot_score + human_score, 0.01)
            
            features['intent_bot_score'] = bot_score / total
            features['intent_human_score'] = human_score / total
            
            # Talking points
            features['talking_point_matches'] = talking_point_matches.get(author_id, 0)
            features['uses_coordinated_language'] = 1 if features['talking_point_matches'] > 0 else 0
            
            additional_features.append(features)
        
        additional_df = pd.DataFrame(additional_features)
        
        # Merge all features
        result_df = author_features_df.merge(additional_df, on='author_id', how='left')
        
        logger.info(f"Extracted {len(result_df.columns) - 1} semantic features for {len(result_df)} authors")
        return result_df
    
    def compile_semantic_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Alias for extract_semantic_features (matches other modules' pattern)"""
        return self.extract_semantic_features(comments_df)
