// Content script that runs on YouTube pages
class YouTubeBotDetector {
  constructor() {
    this.apiEndpoint = null; // Will be loaded from storage
    this.processedComments = new Set();
    this.votingData = {};
    this.init();
  }

  async init() {
    console.log('🤖 YouTube Bot Detector initializing...');
    
    // Load settings from storage
    const stored = await chrome.storage.local.get(['votingData', 'apiEndpoint']);
    this.votingData = stored.votingData || {};
    this.apiEndpoint = stored.apiEndpoint || 'http://localhost:5001/api';
    
    console.log('🤖 API Endpoint:', this.apiEndpoint);
    
    // Start observing for comments
    this.observeComments();
    
    // Process existing comments
    this.processExistingComments();
  }

  observeComments() {
    // Watch for dynamically loaded comments
    const observer = new MutationObserver((mutations) => {
      this.processExistingComments();
    });

    // Observe the main content area
    const targetNode = document.querySelector('ytd-app');
    if (targetNode) {
      observer.observe(targetNode, {
        childList: true,
        subtree: true
      });
    }
  }

  processExistingComments() {
    // Find all comment elements
    const comments = document.querySelectorAll('ytd-comment-thread-renderer');
    
    console.log(`🤖 Found ${comments.length} comments to process`);
    
    comments.forEach(comment => {
      const commentId = this.getCommentId(comment);
      if (commentId && !this.processedComments.has(commentId)) {
        this.processedComments.add(commentId);
        console.log(`🤖 Processing comment: ${commentId}`);
        this.analyzeComment(comment, commentId);
      }
    });
  }

  getCommentId(commentElement) {
    // Try to get the actual comment ID from the element
    if (commentElement.id) {
      return commentElement.id;
    }
    
    // Fallback: create a unique ID based on comment content
    const contentElement = commentElement.querySelector('#content-text');
    const authorElement = commentElement.querySelector('#author-text');
    
    if (contentElement && authorElement) {
      const content = contentElement.textContent?.trim() || '';
      const author = authorElement.textContent?.trim() || '';
      // Create a simple hash
      const hash = `${author}-${content}`.substring(0, 50).replace(/\s/g, '-');
      return `comment-${hash}-${Date.now()}`;
    }
    
    // Last resort: use timestamp and random
    return `comment-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  }

  async analyzeComment(commentElement, commentId) {
    try {
      // Extract comment data
      const commentData = this.extractCommentData(commentElement);
      
      console.log(`🤖 Analyzing comment from: ${commentData.author}`);
      
      // Get bot probability from backend or calculate locally
      const botProbability = await this.getBotProbability(commentData);
      
      console.log(`🤖 Bot probability: ${(botProbability * 100).toFixed(1)}%`);
      
      // Add UI elements to show probability and voting
      this.addBotIndicator(commentElement, commentId, botProbability, commentData);
    } catch (error) {
      console.error('🤖 Error analyzing comment:', error);
    }
  }

  extractCommentData(commentElement) {
    const authorElement = commentElement.querySelector('#author-text');
    const contentElement = commentElement.querySelector('#content-text');
    const timeElement = commentElement.querySelector('.published-time-text a');
    const likesElement = commentElement.querySelector('#vote-count-middle');
    
    return {
      author: authorElement?.textContent?.trim() || 'Unknown',
      authorLink: authorElement?.querySelector('a')?.href || '',
      content: contentElement?.textContent?.trim() || '',
      timestamp: timeElement?.textContent?.trim() || '',
      likes: likesElement?.textContent?.trim() || '0',
      isPinned: !!commentElement.querySelector('ytd-pinned-comment-badge-renderer'),
      isHeartedByCreator: !!commentElement.querySelector('yt-icon.ytd-creator-heart-renderer')
    };
  }

  async getBotProbability(commentData) {
    // Try to get from backend API first
    try {
      const response = await fetch(`${this.apiEndpoint}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(commentData),
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      
      if (response.ok) {
        const result = await response.json();
        return result.bot_probability || 0;
      }
    } catch (error) {
      // Backend not available, fall back to local heuristics
      console.log('Using local detection:', error.message);
    }
    
    // Local heuristic-based detection
    return this.calculateLocalBotScore(commentData);
  }

  calculateLocalBotScore(commentData) {
    let score = 0;
    const { author, content, likes } = commentData;
    
    // Check for suspicious username patterns
    if (/\d{4,}/.test(author)) score += 0.2; // Many numbers
    if (author.length < 5) score += 0.1; // Very short
    if (/^[A-Z]{4,}$/.test(author)) score += 0.15; // All caps
    
    // Check for spam patterns in content
    const lowerContent = content.toLowerCase();
    const spamKeywords = ['click here', 'check out', 'visit', 'subscribe', 'http', 'www.', 
                          'earn money', 'free', 'winner', 'congratulations'];
    const spamCount = spamKeywords.filter(kw => lowerContent.includes(kw)).length;
    score += Math.min(spamCount * 0.1, 0.3);
    
    // Check for excessive emojis
    const emojiCount = (content.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
    if (emojiCount > 5) score += 0.15;
    
    // Check for repetitive characters
    if (/(.)\1{3,}/.test(content)) score += 0.1;
    
    // Very short or very long comments
    if (content.length < 10 || content.length > 1000) score += 0.1;
    
    // Check for generic phrases
    const genericPhrases = ['nice video', 'great content', 'awesome', 'cool', 'first', 'early'];
    if (genericPhrases.some(phrase => lowerContent === phrase)) score += 0.15;
    
    return Math.min(score, 1.0); // Cap at 1.0
  }

  addBotIndicator(commentElement, commentId, probability, commentData) {
    // Don't add if already exists
    if (commentElement.querySelector('.bot-detector-widget')) {
      console.log(`🤖 Widget already exists for comment ${commentId}`);
      return;
    }

    console.log(`🤖 Adding bot indicator for comment ${commentId}`);

    // Create the bot indicator widget
    const widget = document.createElement('div');
    widget.className = 'bot-detector-widget';
    
    // Get current user vote if exists
    const userVote = this.votingData[commentId]?.vote || 0;
    
    // Determine risk level
    let riskLevel = 'low';
    let riskText = 'Low';
    let riskColor = '#00a152';
    
    if (probability > 0.7) {
      riskLevel = 'high';
      riskText = 'High';
      riskColor = '#d32f2f';
    } else if (probability > 0.4) {
      riskLevel = 'medium';
      riskText = 'Medium';
      riskColor = '#ff9800';
    }
    
    widget.innerHTML = `
      <div class="bot-indicator ${riskLevel}">
        <span class="bot-icon">🤖</span>
        <span class="bot-label">Bot Risk: ${riskText}</span>
        <span class="bot-probability">${(probability * 100).toFixed(0)}%</span>
        <div class="bot-voting">
          <button class="vote-btn vote-bot ${userVote === 1 ? 'active' : ''}" 
                  data-comment-id="${commentId}" 
                  data-vote="1"
                  title="This is a bot">
            👍 Bot
          </button>
          <button class="vote-btn vote-human ${userVote === -1 ? 'active' : ''}" 
                  data-comment-id="${commentId}" 
                  data-vote="-1"
                  title="This is human">
            👤 Human
          </button>
        </div>
      </div>
    `;
    
    // Add click handlers for voting
    const voteButtons = widget.querySelectorAll('.vote-btn');
    voteButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.handleVote(commentId, parseInt(btn.dataset.vote), commentData, widget);
      });
    });
    
    // Insert the widget into the comment
    const toolbar = commentElement.querySelector('#toolbar');
    if (toolbar) {
      toolbar.appendChild(widget);
      console.log(`🤖 Widget successfully added to DOM for comment ${commentId}`);
    } else {
      console.error(`🤖 Could not find toolbar for comment ${commentId}`);
    }
  }

  async handleVote(commentId, voteValue, commentData, widgetElement) {
    // Toggle vote if clicking the same button
    const currentVote = this.votingData[commentId]?.vote || 0;
    const newVote = currentVote === voteValue ? 0 : voteValue;
    
    // Update local storage
    this.votingData[commentId] = {
      vote: newVote,
      timestamp: Date.now(),
      commentData: commentData
    };
    
    await chrome.storage.local.set({ votingData: this.votingData });
    
    // Update UI
    const buttons = widgetElement.querySelectorAll('.vote-btn');
    buttons.forEach(btn => {
      btn.classList.remove('active');
      if (parseInt(btn.dataset.vote) === newVote) {
        btn.classList.add('active');
      }
    });
    
    // Send vote to backend
    try {
      await fetch(`${this.apiEndpoint}/vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          commentId,
          vote: newVote,
          commentData,
          timestamp: Date.now()
        })
      });
    } catch (error) {
      console.log('Could not sync vote to backend:', error.message);
    }
    
    // Show feedback
    this.showVoteFeedback(widgetElement, newVote);
  }

  showVoteFeedback(widgetElement, vote) {
    const feedback = document.createElement('span');
    feedback.className = 'vote-feedback';
    feedback.textContent = vote === 1 ? 'Marked as bot' : vote === -1 ? 'Marked as human' : 'Vote removed';
    
    widgetElement.appendChild(feedback);
    
    setTimeout(() => {
      feedback.remove();
    }, 2000);
  }
}

// Initialize the detector when the page loads
console.log('🤖 YouTube Bot Detector script loaded');

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('🤖 DOM Content Loaded - initializing detector');
    new YouTubeBotDetector();
  });
} else {
  console.log('🤖 DOM already loaded - initializing detector immediately');
  new YouTubeBotDetector();
}
