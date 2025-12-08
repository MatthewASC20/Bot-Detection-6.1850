// content.js - Batch-first YouTube Bot Detector with Full Pipeline
class YouTubeBotDetector {
  constructor() {
    this.apiEndpoint = null;
    this.processedComments = new Map(); // comment_id -> result
    this.votingData = {};
    this.currentVideoId = null;
    this.commentQueue = [];
    this.queuedCommentIds = new Set();
    this.detectionEnabled = true;
    this.votingEnabled = true;
    this.collectTimer = null;
    this.analysisInProgress = false;
    this.BATCH_SIZE = 50;
    this.BATCH_DELAY = 2000;
    this.handleNavigation = this.handleNavigation.bind(this);
    this.init();
  }

  async init() {
    console.log('🤖 YouTube Bot Detector v2.0 initializing...');
    
    const stored = await chrome.storage.local.get(['votingData', 'apiEndpoint', 'detectionEnabled', 'votingEnabled']);
    this.votingData = stored.votingData || {};
    this.apiEndpoint = stored.apiEndpoint || 'http://localhost:5001/api';
    this.detectionEnabled = stored.detectionEnabled !== false;
    this.votingEnabled = stored.votingEnabled !== false;
    
    this.currentVideoId = this.extractVideoId();
    console.log('🤖 Video ID:', this.currentVideoId);
    
    // Listen for SPA navigation and settings changes
    window.addEventListener('yt-navigate-finish', this.handleNavigation);
    window.addEventListener('popstate', this.handleNavigation);
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area !== 'local') return;
      if (changes.detectionEnabled) {
        this.detectionEnabled = changes.detectionEnabled.newValue !== false;
        if (this.detectionEnabled) {
          this.collectAllVisibleComments();
        } else {
          document.querySelectorAll('.bot-detector-widget').forEach(el => el.remove());
          this.commentQueue = [];
          this.queuedCommentIds.clear();
        }
      }
      if (changes.votingEnabled) {
        this.votingEnabled = changes.votingEnabled.newValue !== false;
      }
      if (changes.apiEndpoint && changes.apiEndpoint.newValue) {
        this.apiEndpoint = changes.apiEndpoint.newValue;
      }
    });

    this.observeComments();
    this.startBatchProcessor();
    
    // Initial collection after page load
    setTimeout(() => this.collectAllVisibleComments(), 2000);
  }

  extractVideoId() {
    const url = window.location.href;
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : 'unknown';
  }

  observeComments() {
    const observer = new MutationObserver(() => {
      this.scheduleCollect();
    });

    const targetNode = document.querySelector('ytd-app');
    if (targetNode) {
      observer.observe(targetNode, { childList: true, subtree: true });
    }
  }

  scheduleCollect() {
    if (!this.detectionEnabled) return;
    clearTimeout(this.collectTimer);
    this.collectTimer = setTimeout(() => this.collectAllVisibleComments(), 300);
  }

  collectAllVisibleComments() {
    if (!this.detectionEnabled) return;
    const comments = document.querySelectorAll('ytd-comment-thread-renderer');
    
    comments.forEach(comment => {
      const commentData = this.extractFullCommentData(comment);
      if (commentData && 
          !this.processedComments.has(commentData.comment_id) && 
          !this.queuedCommentIds.has(commentData.comment_id)) {
        this.commentQueue.push({ element: comment, data: commentData });
        this.queuedCommentIds.add(commentData.comment_id);
      }
    });
    
    console.log(`🤖 Queue: ${this.commentQueue.length} new comments`);
  }

  extractFullCommentData(commentElement) {
    const authorElement = commentElement.querySelector('#author-text');
    const contentElement = commentElement.querySelector('#content-text');
    const timeElement = commentElement.querySelector('.published-time-text a');
    const likesElement = commentElement.querySelector('#vote-count-middle');
    const authorLinkElement = commentElement.querySelector('#author-text');
    
    if (!contentElement || !authorElement) return null;
    
    const content = contentElement.textContent?.trim() || '';
    const author = authorElement.textContent?.trim() || '';
    
    // Generate unique comment ID
    const comment_id = this.hashString(`${author}|${content}|${this.currentVideoId}`);
    
    // Extract author channel ID from link
    let author_id = '';
    const authorLink = authorLinkElement?.href || '';
    const channelMatch = authorLink.match(/\/channel\/(UC[^\/\?]+)/);
    if (channelMatch) {
      author_id = channelMatch[1];
    } else {
      author_id = this.hashString(author);
    }
    
    // Parse timestamp
    const timeText = timeElement?.textContent?.trim() || '';
    const published_at = this.parseRelativeTime(timeText);
    
    return {
      comment_id,
      author,
      author_id,
      content,
      published_at,
      like_count: this.parseLikeCount(likesElement?.textContent?.trim()),
      is_reply: !!commentElement.closest('ytd-comment-replies-renderer'),
      is_pinned: !!commentElement.querySelector('ytd-pinned-comment-badge-renderer'),
      is_hearted: !!commentElement.querySelector('yt-icon.ytd-creator-heart-renderer')
    };
  }

  parseLikeCount(text) {
    if (!text) return 0;
    const normalized = text.replace(/[, ]/g, '').toUpperCase();
    const match = normalized.match(/^([\d.]+)([KM]?)$/);
    if (!match) return parseInt(normalized, 10) || 0;
    const value = parseFloat(match[1]);
    const suffix = match[2];
    const multipliers = { '': 1, K: 1_000, M: 1_000_000 };
    return Math.round(value * (multipliers[suffix] || 1));
  }

  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(36);
  }

  parseRelativeTime(timeText) {
    const now = new Date();
    const match = timeText.match(/(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago/i);
    
    if (!match) return now.toISOString();
    
    const value = parseInt(match[1]);
    const unit = match[2].toLowerCase();
    
    const offsets = {
      second: 1000, minute: 60000, hour: 3600000,
      day: 86400000, week: 604800000, month: 2592000000, year: 31536000000
    };
    
    return new Date(now.getTime() - value * (offsets[unit] || 0)).toISOString();
  }

  handleNavigation() {
    const newVideoId = this.extractVideoId();
    if (!newVideoId || newVideoId === this.currentVideoId) return;
    
    console.log('🤖 Navigation detected, resetting state for video', newVideoId);
    this.currentVideoId = newVideoId;
    this.processedComments.clear();
    this.commentQueue = [];
    this.queuedCommentIds.clear();
    document.querySelectorAll('.bot-detector-widget').forEach(el => el.remove());
    this.collectAllVisibleComments();
  }

  async startBatchProcessor() {
    setInterval(async () => {
      if (this.commentQueue.length >= this.BATCH_SIZE || 
          (this.commentQueue.length > 0 && !this.analysisInProgress)) {
        await this.processBatch();
      }
    }, this.BATCH_DELAY);
  }

  async processBatch() {
    if (this.analysisInProgress || this.commentQueue.length === 0) return;
    
    this.analysisInProgress = true;
    const batch = this.commentQueue.splice(0, this.BATCH_SIZE);
    batch.forEach(item => this.queuedCommentIds.delete(item.data.comment_id));
    if (!this.detectionEnabled) {
      this.analysisInProgress = false;
      return;
    }
    
    console.log(`🤖 Processing batch of ${batch.length} comments`);
    
    try {
      const results = await this.sendBatchForAnalysis(batch.map(b => b.data));
      
      // Map results to elements and render
      const resultsMap = new Map();
      for (const comment of results.comments || []) {
        resultsMap.set(comment.comment_id, comment);
      }
      
      for (const item of batch) {
        const result = resultsMap.get(item.data.comment_id);
        const finalResult = result || this.calculateLocalScore(item.data);
        this.processedComments.set(item.data.comment_id, finalResult);
        this.renderIndicator(item.element, item.data.comment_id, finalResult);
      }
      
      // Log summary
      const summary = results.analysis_summary;
      if (summary) {
        console.log(`🤖 Analysis: ${summary.likely_bots} bots, ${summary.suspicious} suspicious, ${summary.clusters_found} clusters`);
      }
      
    } catch (error) {
      console.error('🤖 Batch analysis failed:', error);
      // Fall back to local analysis
      for (const item of batch) {
        const localResult = this.calculateLocalScore(item.data);
        this.processedComments.set(item.data.comment_id, localResult);
        this.renderIndicator(item.element, item.data.comment_id, localResult);
      }
    }
    
    this.analysisInProgress = false;
  }

  async sendBatchForAnalysis(comments) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    try {
      // Call single-comment analyze endpoint for compatibility
      const analysis = await Promise.allSettled(comments.map(async (comment) => {
        const response = await fetch(`${this.apiEndpoint}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(comment),
          signal: controller.signal
        });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        const data = await response.json();
        return {
          ...data,
          comment_id: comment.comment_id,
          author_id: comment.author_id,
          cluster_id: data.cluster_id ?? -1,
          flags: data.flags || [],
          classification: data.classification || (data.bot_probability > 0.7 ? 'likely_bot' : data.bot_probability > 0.4 ? 'suspicious' : 'likely_human')
        };
      }));

      const commentsWithIds = analysis
        .filter(r => r.status === 'fulfilled')
        .map(r => r.value);

      if (commentsWithIds.length === 0) {
        throw new Error('No analysis results received');
      }

      return { comments: commentsWithIds, analysis_summary: null };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  calculateLocalScore(commentData) {
    let score = 0;
    const { author, content } = commentData;
    
    if (/\d{4,}/.test(author)) score += 0.2;
    if (author.length < 5) score += 0.1;
    if (/^[A-Z]{4,}$/.test(author)) score += 0.15;
    
    const lower = content.toLowerCase();
    const spamWords = ['click here', 'check out', 'subscribe', 'http', 'www.', 'earn money', 'free', 'winner'];
    score += Math.min(spamWords.filter(w => lower.includes(w)).length * 0.1, 0.3);
    
    const emojis = (content.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
    if (emojis > 5) score += 0.15;
    if (/(.)\1{3,}/.test(content)) score += 0.1;
    
    return {
      comment_id: commentData.comment_id,
      bot_probability: Math.min(score, 1.0),
      confidence: 0.5,
      classification: score > 0.7 ? 'likely_bot' : score > 0.4 ? 'suspicious' : 'likely_human',
      cluster_id: -1,
      flags: [],
      similar_to: []
    };
  }

  renderIndicator(commentElement, commentId, result) {
    if (commentElement.querySelector('.bot-detector-widget')) return;
    
    const { bot_probability, confidence, classification, cluster_id, flags } = result;
    const userVote = this.votingData[commentId]?.vote || 0;
    
    // Determine risk level
    let riskLevel = 'low', riskText = 'Low';
    if (bot_probability > 0.7) { riskLevel = 'high'; riskText = 'High'; }
    else if (bot_probability > 0.4) { riskLevel = 'medium'; riskText = 'Medium'; }
    
    const widget = document.createElement('div');
    widget.className = 'bot-detector-widget';
    
    // Build flags badges HTML
    const flagsHtml = flags.length > 0 ? `
      <div class="bot-flags">
        ${flags.slice(0, 3).map(f => `<span class="flag-badge">${this.formatFlag(f)}</span>`).join('')}
      </div>
    ` : '';
    
    // Build cluster badge HTML
    const clusterHtml = cluster_id !== -1 ? `
      <span class="cluster-badge" title="Part of coordinated group">🔗 Cluster ${cluster_id}</span>
    ` : '';
    
    const votingHtml = this.votingEnabled ? `
      <div class="bot-voting">
        <button class="vote-btn vote-bot ${userVote === 1 ? 'active' : ''}" 
                data-comment-id="${commentId}" data-vote="1" title="Mark as bot">
          👍 Bot
        </button>
        <button class="vote-btn vote-human ${userVote === -1 ? 'active' : ''}" 
                data-comment-id="${commentId}" data-vote="-1" title="Mark as human">
          👤 Human
        </button>
      </div>
    ` : '';

    widget.innerHTML = `
      <div class="bot-indicator ${riskLevel}">
        <span class="bot-icon">🤖</span>
        <span class="bot-label">Bot Risk: ${riskText}</span>
        <span class="bot-probability">${(bot_probability * 100).toFixed(0)}%</span>
        <span class="confidence-range" title="Confidence: ${(confidence * 100).toFixed(0)}%">±${((1 - confidence) * 100 / 2).toFixed(0)}%</span>
        ${clusterHtml}
        ${flagsHtml}
        ${votingHtml}
      </div>
    `;
    
    // Add vote handlers
    if (this.votingEnabled) {
      widget.querySelectorAll('.vote-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.handleVote(commentId, parseInt(btn.dataset.vote), result, widget);
        });
      });
    }
    
    const toolbar = commentElement.querySelector('#toolbar');
    if (toolbar) toolbar.appendChild(widget);
  }

  formatFlag(flag) {
    const flagLabels = {
      'template_match': '📋 Template',
      'burst_posting': '⚡ Burst',
      'spam_keywords': '🚫 Spam',
      'synchronized': '🔄 Sync',
      'clique_member': '👥 Clique',
      'duplicate_content': '📄 Duplicate',
      'automated_behavior': '🤖 Auto',
      'suspicious_username': '👤 Name',
      'new_account': '🆕 New',
      'highly_connected': '🌐 Network'
    };
    return flagLabels[flag] || flag;
  }

  async handleVote(commentId, voteValue, result, widgetElement) {
    if (!this.votingEnabled) return;
    const currentVote = this.votingData[commentId]?.vote || 0;
    const newVote = currentVote === voteValue ? 0 : voteValue;
    
    this.votingData[commentId] = { vote: newVote, timestamp: Date.now() };
    await chrome.storage.local.set({ votingData: this.votingData });
    
    // Update UI
    widgetElement.querySelectorAll('.vote-btn').forEach(btn => {
      btn.classList.remove('active');
      if (parseInt(btn.dataset.vote) === newVote) btn.classList.add('active');
    });
    
    // Sync to backend
    try {
      await fetch(`${this.apiEndpoint}/vote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          comment_id: commentId,
          author_id: result.author_id,
          vote: newVote,
          video_id: this.currentVideoId,
          ml_prediction: result.bot_probability
        })
      });
    } catch (e) {
      console.log('Vote sync failed:', e.message);
    }
    
    this.showFeedback(widgetElement, newVote);
  }

  showFeedback(widget, vote) {
    const feedback = document.createElement('span');
    feedback.className = 'vote-feedback';
    feedback.textContent = vote === 1 ? 'Marked as bot' : vote === -1 ? 'Marked as human' : 'Vote removed';
    widget.appendChild(feedback);
    setTimeout(() => feedback.remove(), 2000);
  }
}

// Initialize
console.log('🤖 YouTube Bot Detector v2.0 loaded');
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new YouTubeBotDetector());
} else {
  new YouTubeBotDetector();
}
