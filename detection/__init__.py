"""
Detection modules for bot identification.
"""
from detection.clustering import ClusteringDetector
from detection.botbuster_detector import BotBusterDetector

__all__ = [
    'ClusteringDetector',
    'BotBusterDetector'
]

