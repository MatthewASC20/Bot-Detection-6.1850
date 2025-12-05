"""
Semantic feature extraction using OpenAI embeddings for BotBuster
Implements semantic similarity analysis as described in the BotBuster paper.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import time

from config.config import Config

logger = logging.getLogger(__name__)


class SemanticFeatures:
    """Extract semantic features using OpenAI embeddings for bot detection"""
    
    def __init__(self):
        """Initialize OpenAI client with API key from config"""
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_EMBEDDING_MODEL
        self.embedding_cache = {}
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI API
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Clean and truncate texts (OpenAI has token limits)
        cleaned_texts = []
        for text in texts:
            if text is None or pd.isna(text):
                text = ""
            # Truncate to ~8000 characters to stay within token limits
            cleaned_texts.append(str(text)[:8000])
        
        embeddings = []
        batch_size = Config.EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]
            
            # Check cache first
            batch_embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            
            for j, text in enumerate(batch):
                text_hash = hash(text)
                if text_hash in self.embedding_cache:
                    batch_embeddings.append((j, self.embedding_cache[text_hash]))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(j)
            
            # Get embeddings for non-cached texts
            if texts_to_embed:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=texts_to_embed
                    )
                    
                    for idx, (j, text) in enumerate(zip(indices_to_embed, texts_to_embed)):
                        embedding = response.data[idx].embedding
                        text_hash = hash(text)
                        self.embedding_cache[text_hash] = embedding
                        batch_embeddings.append((j, embedding))
                    
                except Exception as e:
                    logger.error(f"Error getting embeddings: {e}")
                    # Return zero embeddings on error
                    for j, text in zip(indices_to_embed, texts_to_embed):
                        batch_embeddings.append((j, [0.0] * Config.EMBEDDING_DIMENSION))
                
                # Rate limiting
                time.sleep(0.1)
            
            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        return np.array(embeddings)
    
    def calculate_pairwise_semantic_similarity(self, comments_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate pairwise semantic similarity matrix for all comments
        
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
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        logger.info(f"Semantic similarity matrix computed: shape {similarity_matrix.shape}")
        return similarity_matrix
    
    def calculate_comment_to_discussion_similarity(self, comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate how similar each comment is to the broader discussion.
        Low similarity to discussion = potentially more coordinated/off-topic
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping comment_id to discussion similarity score
        """
        logger.info("Calculating comment-to-discussion similarity...")
        
        # Group by video to calculate per-video discussion similarity
        discussion_scores = {}
        
        for video_id, group in comments_df.groupby('video_id'):
            if len(group) < 2:
                # Single comment video - neutral score
                for comment_id in group['comment_id']:
                    discussion_scores[comment_id] = 0.5
                continue
            
            texts = group['text'].fillna('').tolist()
            embeddings = self.get_embeddings(texts)
            
            if len(embeddings) == 0:
                for comment_id in group['comment_id']:
                    discussion_scores[comment_id] = 0.5
                continue
            
            # Calculate centroid of all embeddings (represents "average discussion")
            discussion_centroid = np.mean(embeddings, axis=0, keepdims=True)
            
            # Calculate similarity of each comment to the discussion centroid
            for i, (_, row) in enumerate(group.iterrows()):
                comment_embedding = embeddings[i:i+1]
                similarity = cosine_similarity(comment_embedding, discussion_centroid)[0][0]
                discussion_scores[row['comment_id']] = similarity
        
        logger.info(f"Calculated discussion similarity for {len(discussion_scores)} comments")
        return discussion_scores
    
    def calculate_author_semantic_features(self, comments_df: pd.DataFrame, 
                                           similarity_matrix: np.ndarray) -> pd.DataFrame:
        """
        Calculate semantic features per author based on their comments
        
        Args:
            comments_df: DataFrame with comments
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            DataFrame with semantic features per author
        """
        logger.info("Calculating author-level semantic features...")
        
        semantic_features = []
        comment_ids = comments_df['comment_id'].tolist()
        comment_id_to_idx = {cid: idx for idx, cid in enumerate(comment_ids)}
        
        # Get discussion similarity scores
        discussion_scores = self.calculate_comment_to_discussion_similarity(comments_df)
        
        for author_id, group in comments_df.groupby('author_id'):
            features = {'author_id': author_id}
            
            author_indices = [comment_id_to_idx[cid] for cid in group['comment_id'] if cid in comment_id_to_idx]
            
            if not author_indices:
                features['avg_semantic_similarity_to_others'] = 0.0
                features['max_semantic_similarity_to_others'] = 0.0
                features['high_similarity_count'] = 0
                features['avg_discussion_similarity'] = 0.5
                features['semantic_diversity'] = 1.0
                semantic_features.append(features)
                continue
            
            # Get all similarities from this author's comments to other comments
            other_similarities = []
            for idx in author_indices:
                for j in range(similarity_matrix.shape[1]):
                    if j not in author_indices:  # Exclude self-comparisons
                        other_similarities.append(similarity_matrix[idx, j])
            
            if other_similarities:
                features['avg_semantic_similarity_to_others'] = np.mean(other_similarities)
                features['max_semantic_similarity_to_others'] = np.max(other_similarities)
                # Count of high-similarity comments (potential coordination)
                features['high_similarity_count'] = sum(
                    1 for s in other_similarities if s > Config.SEMANTIC_SIMILARITY_THRESHOLD
                )
            else:
                features['avg_semantic_similarity_to_others'] = 0.0
                features['max_semantic_similarity_to_others'] = 0.0
                features['high_similarity_count'] = 0
            
            # Average discussion similarity for this author's comments
            author_discussion_scores = [
                discussion_scores.get(cid, 0.5) for cid in group['comment_id']
            ]
            features['avg_discussion_similarity'] = np.mean(author_discussion_scores)
            
            # Semantic diversity: how diverse are this author's own comments?
            if len(author_indices) > 1:
                self_similarities = []
                for i, idx1 in enumerate(author_indices):
                    for idx2 in author_indices[i+1:]:
                        self_similarities.append(similarity_matrix[idx1, idx2])
                
                if self_similarities:
                    # Low self-similarity = high diversity
                    features['semantic_diversity'] = 1.0 - np.mean(self_similarities)
                else:
                    features['semantic_diversity'] = 1.0
            else:
                features['semantic_diversity'] = 1.0
            
            semantic_features.append(features)
        
        result_df = pd.DataFrame(semantic_features)
        logger.info(f"Calculated semantic features for {len(result_df)} authors")
        return result_df
    
    def find_semantically_similar_groups(self, comments_df: pd.DataFrame,
                                          similarity_matrix: np.ndarray,
                                          threshold: float = None) -> List[set]:
        """
        Find groups of comments that are semantically very similar
        (potential coordinated messages)
        
        Args:
            comments_df: DataFrame with comments
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold (default from config)
            
        Returns:
            List of sets, each containing author_ids in a similar group
        """
        if threshold is None:
            threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD
        
        n = len(comments_df)
        comment_ids = comments_df['comment_id'].tolist()
        author_ids = comments_df['author_id'].tolist()
        
        # Find pairs above threshold
        similar_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] > threshold:
                    # Only count if different authors
                    if author_ids[i] != author_ids[j]:
                        similar_pairs.append((i, j))
        
        # Build groups using union-find
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
        similar_groups = [g for g in groups.values() if len(g) >= Config.MIN_CLUSTER_SIZE]
        
        logger.info(f"Found {len(similar_groups)} semantically similar groups")
        return similar_groups
    
    def calculate_comment_bot_probability(self, comments_df: pd.DataFrame,
                                          similarity_matrix: np.ndarray,
                                          discussion_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate bot probability for each comment based on:
        1. Semantic similarity to other comments (high = suspicious)
        2. Similarity to broader discussion (low diversity = suspicious)
        
        Args:
            comments_df: DataFrame with comments
            similarity_matrix: Pairwise similarity matrix
            discussion_scores: Comment-to-discussion similarity scores
            
        Returns:
            Dictionary mapping comment_id to bot probability
        """
        logger.info("Calculating per-comment bot probabilities...")
        
        comment_ids = comments_df['comment_id'].tolist()
        author_ids = comments_df['author_id'].tolist()
        
        comment_bot_probs = {}
        
        for i, comment_id in enumerate(comment_ids):
            # Get similarities to comments by other authors
            other_author_sims = []
            for j in range(len(comment_ids)):
                if i != j and author_ids[i] != author_ids[j]:
                    other_author_sims.append(similarity_matrix[i, j])
            
            if not other_author_sims:
                semantic_score = 0.0
            else:
                # High average similarity to other authors' comments = suspicious
                avg_sim = np.mean(other_author_sims)
                max_sim = np.max(other_author_sims)
                # Weight max higher to catch exact copies
                semantic_score = 0.4 * avg_sim + 0.6 * max_sim
            
            # Discussion divergence score
            # Low similarity to discussion centroid = potentially off-topic/coordinated
            discussion_sim = discussion_scores.get(comment_id, 0.5)
            # Invert: low discussion similarity = higher bot score
            # But we also want to flag comments that are TOO generic/similar to discussion
            # Use distance from 0.5 (moderate similarity is most human-like)
            discussion_divergence = abs(discussion_sim - 0.5) * 2  # Scale to 0-1
            
            # Combine scores
            # High semantic similarity to others AND divergent from discussion = bot-like
            bot_probability = 0.7 * semantic_score + 0.3 * discussion_divergence
            
            comment_bot_probs[comment_id] = min(max(bot_probability, 0.0), 1.0)
        
        logger.info(f"Calculated bot probabilities for {len(comment_bot_probs)} comments")
        return comment_bot_probs

