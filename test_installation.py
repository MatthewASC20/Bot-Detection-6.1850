#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config.config import Config
        print("✓ Config module imported")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from data_collection.youtube_api import YouTubeAPI
        from data_collection.data_collector import DataCollector
        print("✓ Data collection modules imported")
    except Exception as e:
        print(f"✗ Data collection import failed: {e}")
        return False
    
    try:
        from features.temporal_features import TemporalFeatures
        from features.text_features import TextFeatures
        from features.network_features import NetworkFeatures
        from features.behavioral_features import BehavioralFeatures
        print("✓ Feature extraction modules imported")
    except Exception as e:
        print(f"✗ Feature extraction import failed: {e}")
        return False
    
    try:
        from detection.clustering import ClusteringDetector
        print("✓ Detection modules imported")
    except Exception as e:
        print(f"✗ Detection import failed: {e}")
        return False
    
    try:
        from visualization.network_viz import NetworkVisualizer
        print("✓ Visualization modules imported")
    except Exception as e:
        print(f"✗ Visualization import failed: {e}")
        return False
    
    try:
        from storage.database import DatabaseHandler
        print("✓ Storage modules imported")
    except Exception as e:
        print(f"✗ Storage import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        from config.config import Config
        
        # Check if API keys are configured
        if not Config.YOUTUBE_API_KEYS:
            print("⚠ No API keys configured. Please add your API key to .env file")
            return False
        
        print(f"✓ Found {len(Config.YOUTUBE_API_KEYS)} API key(s)")
        
        # Check directories
        for dir_path in [Config.OUTPUT_DIR, Config.GRAPHS_DIR, Config.REPORTS_DIR]:
            if os.path.exists(dir_path):
                print(f"✓ Directory exists: {dir_path}")
            else:
                print(f"⚠ Directory missing: {dir_path} (will be created on first run)")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_dependencies():
    """Test that required packages are installed"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'networkx', 
        'plotly', 'nltk', 'hdbscan', 'Levenshtein'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("YouTube Botnet Detector - Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check your installation.")
        sys.exit(1)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Dependency test failed. Install missing packages.")
        sys.exit(1)
    
    # Test configuration
    if not test_config():
        print("\n⚠ Configuration incomplete. Add your API key to .env file.")
    
    print("\n" + "=" * 60)
    print("✅ Installation test completed!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and add your YouTube API key")
    print("2. Run: python main.py --mode political --max-comments 100")
    print("   (for a small test with political content)")
    print("3. Or: python main.py --urls 'VIDEO_URL' --max-comments 100")
    print("   (to analyze specific videos)")
    print("=" * 60)

if __name__ == "__main__":
    main()
