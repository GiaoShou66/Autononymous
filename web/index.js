// State Management
const state = {
    conversations: [],
    currentConversation: null,
    messages: [],
    isStreaming: false,
    currentAbortController: null,
    settings: {
        model: 'claude-3-5-sonnet',
        autonomous: true,
    },
    mode: 'claude-code',
    pendingAttachments: [],
    activeNav: 'home',
    searchQuery: '',
    manageMode: false,
    selectedIds: new Set(),
    isRecording: false,
};

let voiceRecognition = null;

// DOM Elements
const elements = {
    sidebar: document.getElementById('sidebar'),
    toggleSidebar: document.getElementById('toggleSidebar'),
    newChat: document.getElementById('newChat'),
    mobileMenu: document.getElementById('mobileMenu'),
    conversationList: document.getElementById('conversationList'),
    emptyState: document.getElementById('emptyState'),
    chatContainer: document.getElementById('chatContainer'),
    messages: document.getElementById('messages'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    stopButton: document.getElementById('stopButton'),
    addButton: document.getElementById('addButton'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    themeToggle: document.getElementById('themeToggle'),
    attachmentBar: document.getElementById('attachmentBar'),
    voiceInput: document.getElementById('voiceInput'),
    imageInput: document.getElementById('imageInput'),
    searchInput: document.getElementById('searchInput'),
    navItems: document.querySelectorAll('.nav-item'),
    manageBar: document.getElementById('manageBar'),
    selectAllBtn: document.getElementById('selectAllBtn'),
    deleteSelectedBtn: document.getElementById('deleteSelectedBtn'),
    selectedCount: document.getElementById('selectedCount'),
    modeSelect: document.getElementById('modeSelect'),
    settingsButton: document.getElementById('settingsBtn'),
    settingsModal: document.getElementById('settingsModal'),
    closeModal: document.getElementById('closeModal'),
    saveSettings: document.getElementById('saveSettings'),
    cancelSettings: document.getElementById('cancelSettings'),
    chatTitle: document.getElementById('chatTitle'),
};

// Initialize
function init() {
    applySavedTheme();
    loadSettings();
    loadConversations();
    // Create an initial conversation only if none were loaded from storage.
    // This must run after loadConversations(), not as an unguarded check at
    // script scope — document.readyState can still be "loading" when this
    // script executes, which defers init() to DOMContentLoaded; a check
    // against state.conversations before that fires would always see the
    // initial empty array and overwrite real saved history in localStorage.
    if (state.conversations.length === 0) {
        createNewChat();
    }
    attachEventListeners();
    autoResizeTextarea();
}

// Event Listeners
function attachEventListeners() {
    // Sidebar toggle
    elements.toggleSidebar.addEventListener('click', toggleSidebar);
    elements.mobileMenu.addEventListener('click', toggleSidebar);

    // New chat
    elements.newChat.addEventListener('click', createNewChat);

    // Sidebar nav (首页/管理/收藏/近期)
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => setActiveNav(item.dataset.nav));
    });

    // Conversation search
    elements.searchInput.addEventListener('input', (e) => {
        state.searchQuery = e.target.value.trim().toLowerCase();
        renderConversations();
    });

    // Manage mode bulk actions
    elements.selectAllBtn.addEventListener('click', toggleSelectAllConversations);
    elements.deleteSelectedBtn.addEventListener('click', deleteSelectedConversations);

    // Send message
    elements.sendButton.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keydown', handleInputKeydown);

    // Suggested prompts
    document.querySelectorAll('.prompt-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            elements.messageInput.value = prompt;
            sendMessage();
        });
    });

    // Settings
    elements.settingsButton.addEventListener('click', openSettings);
    elements.closeModal.addEventListener('click', closeSettings);
    elements.cancelSettings.addEventListener('click', closeSettings);
    elements.saveSettings.addEventListener('click', saveSettings);
    document.getElementById('routingModeSelect').addEventListener('change', updateRoutingGroupVisibility);

    // Mode selection
    elements.modeSelect.addEventListener('change', (e) => {
        state.mode = e.target.value;
    });

    // Input buttons
    elements.addButton.addEventListener('click', showAddMenu);
    elements.voiceInput.addEventListener('click', startVoiceInput);
    elements.imageInput.addEventListener('click', uploadImage);
    elements.stopButton.addEventListener('click', stopGeneration);

    // Header actions
    elements.clearChatBtn.addEventListener('click', clearCurrentChat);
    elements.themeToggle.addEventListener('click', toggleTheme);
}

function clearCurrentChat() {
    if (state.messages.length === 0) return;
    if (!confirm('清空当前对话？')) return;

    state.messages = [];
    saveCurrentConversation();
    showEmptyState();
    clearServerHistory();
}

function toggleTheme() {
    const current = document.body.dataset.theme === 'light' ? 'light' : 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.body.dataset.theme = next;
    localStorage.setItem('theme', next);

    const icon = elements.themeToggle.querySelector('i');
    if (icon) icon.className = next === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
}

function applySavedTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') {
        document.body.dataset.theme = saved;
    }
    const icon = elements.themeToggle && elements.themeToggle.querySelector('i');
    if (icon) {
        const current = document.body.dataset.theme === 'light' ? 'light' : 'dark';
        icon.className = current === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

function toggleSidebar() {
    elements.sidebar.classList.toggle('active');
}

function createNewChat() {
    const conversation = {
        id: Date.now().toString(),
        title: '新对话',
        messages: [],
        createdAt: new Date().toISOString(),
    };

    state.conversations.unshift(conversation);
    state.currentConversation = conversation.id;
    state.messages = [];

    saveConversations();
    renderConversations();
    showEmptyState();
    clearServerHistory();
}

// There's only one agent instance on the server, shared by every
// "conversation" the sidebar shows — switching or starting a new one
// must reset its history, or old context (and anything the model said
// about itself) bleeds into the next conversation.
function clearServerHistory() {
    fetch('/api/clear', { method: 'POST' }).catch(err => {
        console.error('Failed to clear server-side conversation history:', err);
    });
}

function sendMessage() {
    const message = elements.messageInput.value.trim();
    const attachments = state.pendingAttachments;
    if ((!message && attachments.length === 0) || state.isStreaming) return;
    if (attachments.some(att => att.uploading)) {
        alert('请等待附件上传完成后再发送。');
        return;
    }

    // Hide empty state, show chat
    elements.emptyState.style.display = 'none';
    elements.chatContainer.style.display = 'flex';

    // Add user message (shown with attachment chips, sent with their content inlined)
    addMessage('user', message, attachments);
    const fullMessage = buildMessageWithAttachments(message, attachments);

    // Clear input
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    clearAttachments();

    // Send to API
    sendToAPI(fullMessage);
}

function buildMessageWithAttachments(message, attachments) {
    if (!attachments || attachments.length === 0) return message;
    const blocks = attachments.map(a => a.text).join('\n\n');
    return message ? `${blocks}\n\n${message}` : blocks;
}

function addMessage(role, content, attachments = [], extra = {}) {
    const message = {
        role,
        content,
        attachments: attachments.map(a => a.filename),
        progress: extra.progress && extra.progress.length ? extra.progress : null,
        timestamp: new Date().toISOString(),
    };

    state.messages.push(message);

    // Update conversation title if first message
    if (state.messages.length === 1 && state.currentConversation) {
        const conv = state.conversations.find(c => c.id === state.currentConversation);
        if (conv) {
            conv.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
            renderConversations();
        }
    }

    renderMessage(message);
    saveCurrentConversation();
}

function renderMessage(message) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.role}`;

    const isAssistant = message.role === 'assistant';
    const avatar = isAssistant ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
    const sender = isAssistant ? 'AI 助手' : '你';

    messageEl.innerHTML = `
        <div class="message-header">
            <div class="message-avatar">${avatar}</div>
            <span class="message-sender">${sender}</span>
            <span class="message-time">${formatTime(message.timestamp)}</span>
        </div>
        ${message.attachments && message.attachments.length ? `
            <div class="message-attachments">
                ${message.attachments.map(name => `
                    <span class="attachment-chip"><i class="fas fa-paperclip"></i> ${escapeHtml(name)}</span>
                `).join('')}
            </div>
        ` : ''}
        <div class="message-content">${formatContent(message.content)}</div>
        ${message.progress && message.progress.length ? `
            <details class="autonomous-progress">
                <summary><i class="fas fa-robot"></i> 自主执行记录 (${message.progress.length})</summary>
                <div class="autonomous-progress-log">
                    ${message.progress.map(step => `<div>${escapeHtml(step)}</div>`).join('')}
                </div>
            </details>
        ` : ''}
        ${isAssistant ? `
            <div class="message-actions">
                <button class="btn-icon" onclick="copyMessage(this)">
                    <i class="fas fa-copy"></i>
                </button>
                <button class="btn-icon" onclick="regenerateMessage(this)">
                    <i class="fas fa-sync"></i>
                </button>
            </div>
        ` : ''}
    `;

    elements.messages.appendChild(messageEl);
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Parses a ```ask / ```ask-multi block body: first non-blank line is the
// question, each "- " line after it is one option.
function parseQuestionBlock(body) {
    const lines = body.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length === 0) return null;
    const [question, ...rest] = lines;
    const options = rest
        .filter(l => l.startsWith('- '))
        .map(l => l.slice(2).trim())
        .filter(Boolean);
    return { question, options };
}

function formatContent(content) {
    // ```ask blocks (see agent.py's SYSTEM_PROMPT) render as clickable
    // option buttons instead of a code block — single choice, sent
    // immediately on click.
    content = content.replace(/```ask\n([\s\S]+?)```/g, (_match, body) => {
        const parsed = parseQuestionBlock(body);
        if (!parsed) return '';
        const { question, options } = parsed;
        if (options.length === 0) return escapeHtml(question);

        return `
            <div class="question-options">
                <div class="question-options-prompt">${escapeHtml(question)}</div>
                <div class="question-options-list">
                    ${options.map(opt => `
                        <button type="button" class="question-option-btn" onclick="selectQuestionOption(this)">${escapeHtml(opt)}</button>
                    `).join('')}
                </div>
            </div>
        `;
    });

    // ```ask-multi blocks render as checkboxes plus a Submit button —
    // multiple choice, sent as the checked options joined with ", ".
    content = content.replace(/```ask-multi\n([\s\S]+?)```/g, (_match, body) => {
        const parsed = parseQuestionBlock(body);
        if (!parsed) return '';
        const { question, options } = parsed;
        if (options.length === 0) return escapeHtml(question);

        return `
            <div class="question-options">
                <div class="question-options-prompt">${escapeHtml(question)}</div>
                <div class="question-options-list question-options-list-multi">
                    ${options.map((opt, i) => `
                        <label class="question-option-checkbox">
                            <input type="checkbox" value="${escapeHtml(opt)}" id="qopt-${i}-${Date.now()}">
                            <span>${escapeHtml(opt)}</span>
                        </label>
                    `).join('')}
                </div>
                <button type="button" class="question-options-submit" onclick="submitQuestionMulti(this)">Submit</button>
            </div>
        `;
    });

    // Basic markdown-like formatting
    content = content
        .replace(/```(\w+)?\n([\s\S]+?)```/g, '<pre><code class="language-$1">$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');

    return content;
}

function selectQuestionOption(button) {
    const list = button.closest('.question-options-list');
    if (list) {
        list.querySelectorAll('.question-option-btn').forEach(btn => { btn.disabled = true; });
        button.classList.add('selected');
    }
    elements.messageInput.value = button.textContent;
    sendMessage();
}

function submitQuestionMulti(button) {
    const container = button.closest('.question-options');
    if (!container) return;
    const checked = Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    if (checked.length === 0) {
        alert('选择至少一项。');
        return;
    }
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = true; });
    button.disabled = true;
    elements.messageInput.value = checked.join(', ');
    sendMessage();
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
    
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function stopGeneration() {
    if (state.currentAbortController) {
        state.currentAbortController.abort();
    }
}

async function sendToAPI(message) {
    state.isStreaming = true;
    state.currentAbortController = new AbortController();
    elements.sendButton.disabled = true;
    elements.sendButton.style.display = 'none';
    elements.stopButton.style.display = 'flex';

    // Show typing indicator
    const typingEl = document.createElement('div');
    typingEl.className = 'message assistant';
    typingEl.innerHTML = `
        <div class="message-header">
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <span class="message-sender">AI 助手</span>
        </div>
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    elements.messages.appendChild(typingEl);

    try {
        if (state.settings.autonomous) {
            // Autonomous mode runs through its own synchronous endpoint
            // (auto-continuation until the task looks complete) — it has no
            // streaming variant.
            await sendToAPIAutonomous(message, typingEl);
        } else {
            // Streaming is always on — there is no non-streaming option.
            await sendToAPIStreaming(message, typingEl);
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            // User-initiated stop, already handled where the abort happened.
        } else {
            console.error('Error calling API:', error);
            typingEl.remove();
            addMessage('assistant', `抱歉，发生了错误: ${error.message}\n\n请检查你的设置或稍后重试。`);
        }
    } finally {
        state.isStreaming = false;
        state.currentAbortController = null;
        elements.sendButton.disabled = false;
        elements.sendButton.style.display = '';
        elements.stopButton.style.display = 'none';
    }
}

async function sendToAPIAutonomous(message, typingEl) {
    let response;
    try {
        response = await fetch('/api/autonomous/continuous', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                task: message,
                mode: state.mode,
            }),
            signal: state.currentAbortController.signal,
        });
    } catch (err) {
        if (err.name === 'AbortError') {
            // Autonomous mode has no streaming/partial output to keep —
            // the whole request is one synchronous call, so stopping it
            // just cancels it outright.
            typingEl.remove();
            addMessage('assistant', '(已停止)');
            return;
        }
        throw err;
    }

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `自主模式请求失败 (status ${response.status})`);
    }

    const data = await response.json();

    // Remove typing indicator
    typingEl.remove();

    // Add assistant response, with the auto-continuation log attached
    addMessage('assistant', data.result || '(自主模式没有返回内容)', [], { progress: data.progress || [] });
}

async function sendToAPIStreaming(message, typingEl) {
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message,
            mode: state.mode,
        }),
        signal: state.currentAbortController.signal,
    });

    if (!response.ok) {
        throw new Error(`Streaming request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let messageStarted = false;

    // Remove typing indicator and create message element
    typingEl.remove();

    const messageEl = document.createElement('div');
    messageEl.className = 'message assistant';
    messageEl.innerHTML = `
        <div class="message-header">
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <span class="message-sender">AI 助手</span>
            <span class="message-time">${formatTime(new Date().toISOString())}</span>
        </div>
        <div class="message-content"></div>
        <div class="message-actions">
            <button class="btn-icon" onclick="copyMessage(this)">
                <i class="fas fa-copy"></i>
            </button>
            <button class="btn-icon" onclick="regenerateMessage(this)">
                <i class="fas fa-sync"></i>
            </button>
        </div>
    `;
    elements.messages.appendChild(messageEl);
    const contentEl = messageEl.querySelector('.message-content');

    try {
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonStr = line.slice(6);
                        const data = JSON.parse(jsonStr);

                        if (data.type === 'start') {
                            messageStarted = true;
                        } else if (data.type === 'chunk') {
                            fullResponse += data.content;
                            contentEl.innerHTML = formatContent(fullResponse);
                            elements.messages.scrollTop = elements.messages.scrollHeight;
                        } else if (data.type === 'end') {
                            // Streaming complete
                            break;
                        } else if (data.type === 'error') {
                            throw new Error(data.message || 'Streaming error');
                        }
                    } catch (parseError) {
                        console.error('Error parsing SSE data:', parseError, line);
                    }
                }
            }
        }
    } catch (streamError) {
        if (streamError.name === 'AbortError') {
            // User stopped generation mid-stream — keep whatever text
            // already arrived instead of discarding it as an error.
        } else {
            console.error('Streaming error:', streamError);
            throw streamError;
        }
    }

    // Save the message
    const assistantMessage = {
        role: 'assistant',
        content: fullResponse,
        timestamp: new Date().toISOString(),
    };
    state.messages.push(assistantMessage);
    saveCurrentConversation();

    // Update conversation title if first message
    if (state.messages.length === 2 && state.currentConversation) {
        const conv = state.conversations.find(c => c.id === state.currentConversation);
        if (conv) {
            const userMsg = state.messages[0].content;
            conv.title = userMsg.substring(0, 30) + (userMsg.length > 30 ? '...' : '');
            renderConversations();
        }
    }
}

function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    elements.messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    });
}

function showEmptyState() {
    elements.emptyState.style.display = 'flex';
    elements.chatContainer.style.display = 'none';
    elements.messages.innerHTML = '';
}

function loadConversations() {
    const saved = localStorage.getItem('conversations');
    if (saved) {
        state.conversations = JSON.parse(saved);
        renderConversations();
    }
}

function saveConversations() {
    localStorage.setItem('conversations', JSON.stringify(state.conversations));
}

function saveCurrentConversation() {
    if (!state.currentConversation) return;

    const conv = state.conversations.find(c => c.id === state.currentConversation);
    if (conv) {
        conv.messages = state.messages;
        conv.updatedAt = new Date().toISOString();
        saveConversations();
    }
}

function setActiveNav(nav) {
    state.activeNav = nav;
    state.manageMode = nav === 'manage';
    if (!state.manageMode) state.selectedIds.clear();

    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.nav === nav);
    });
    elements.manageBar.style.display = state.manageMode ? 'flex' : 'none';

    renderConversations();
}

function getFilteredConversations() {
    let list = state.conversations.slice();

    if (state.activeNav === 'favorites') {
        list = list.filter(c => c.favorite);
    } else if (state.activeNav === 'recent') {
        const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
        list = list
            .filter(c => new Date(c.updatedAt || c.createdAt).getTime() >= sevenDaysAgo)
            .sort((a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt));
    }

    if (state.searchQuery) {
        list = list.filter(c => (c.title || '').toLowerCase().includes(state.searchQuery));
    }

    return list;
}

function renderConversations() {
    const list = getFilteredConversations();

    if (list.length === 0) {
        elements.conversationList.innerHTML = `<div class="conversation-empty">没有对话</div>`;
        updateSelectedCount();
        return;
    }

    elements.conversationList.innerHTML = list.map(conv => `
        <div class="conversation-item ${conv.id === state.currentConversation ? 'active' : ''}"
             data-id="${conv.id}"
             onclick="${state.manageMode ? `toggleSelectConversation('${conv.id}')` : `loadConversation('${conv.id}')`}">
            ${state.manageMode ? `
                <input type="checkbox" class="conversation-checkbox"
                       ${state.selectedIds.has(conv.id) ? 'checked' : ''}
                       onclick="event.stopPropagation(); toggleSelectConversation('${conv.id}')">
            ` : ''}
            <span class="conversation-title">${escapeHtml(conv.title)}</span>
            <div class="conversation-actions">
                <button class="btn-icon favorite-btn ${conv.favorite ? 'favorited' : ''}"
                        onclick="event.stopPropagation(); toggleFavorite('${conv.id}')"
                        title="${conv.favorite ? '取消收藏' : '收藏'}">
                    <i class="fas fa-star"></i>
                </button>
                <button class="btn-icon" onclick="event.stopPropagation(); deleteConversation('${conv.id}')" title="删除">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');

    updateSelectedCount();
}

function toggleFavorite(id) {
    const conv = state.conversations.find(c => c.id === id);
    if (!conv) return;
    conv.favorite = !conv.favorite;
    saveConversations();
    renderConversations();
}

function toggleSelectConversation(id) {
    if (state.selectedIds.has(id)) {
        state.selectedIds.delete(id);
    } else {
        state.selectedIds.add(id);
    }
    renderConversations();
}

function toggleSelectAllConversations() {
    const list = getFilteredConversations();
    const allSelected = list.length > 0 && list.every(c => state.selectedIds.has(c.id));
    if (allSelected) {
        list.forEach(c => state.selectedIds.delete(c.id));
    } else {
        list.forEach(c => state.selectedIds.add(c.id));
    }
    renderConversations();
}

function deleteSelectedConversations() {
    if (state.selectedIds.size === 0) return;
    if (!confirm(`确定要删除选中的 ${state.selectedIds.size} 个对话吗？`)) return;

    const idsToDelete = new Set(state.selectedIds);
    state.conversations = state.conversations.filter(c => !idsToDelete.has(c.id));
    state.selectedIds.clear();
    saveConversations();

    if (idsToDelete.has(state.currentConversation)) {
        if (state.conversations.length > 0) {
            loadConversation(state.conversations[0].id);
        } else {
            createNewChat();
        }
    } else {
        renderConversations();
    }
}

function updateSelectedCount() {
    if (elements.selectedCount) {
        elements.selectedCount.textContent = state.selectedIds.size;
    }
}

function loadConversation(id) {
    const conv = state.conversations.find(c => c.id === id);
    if (!conv) return;

    state.currentConversation = id;
    state.messages = conv.messages || [];
    clearServerHistory();

    elements.chatTitle.textContent = conv.title;
    elements.messages.innerHTML = '';

    if (state.messages.length === 0) {
        showEmptyState();
    } else {
        elements.emptyState.style.display = 'none';
        elements.chatContainer.style.display = 'flex';
        state.messages.forEach(renderMessage);
    }

    renderConversations();
}

function deleteConversation(id) {
    if (!confirm('确定要删除这个对话吗？')) return;

    state.conversations = state.conversations.filter(c => c.id !== id);
    
    if (state.currentConversation === id) {
        if (state.conversations.length > 0) {
            loadConversation(state.conversations[0].id);
        } else {
            createNewChat();
        }
    }

    saveConversations();
    renderConversations();
}

// Mirrors models.py's backend-detection rules (_is_commandcode_model /
// _is_openrouter_model / _is_deepseek_model / _is_anthropic_model):
// anything else is routed to the local Ollama backend.
function isCloudModel(m) {
    const lower = m.toLowerCase();
    return lower.startsWith('commandcode:')
        || lower.startsWith('openrouter:')
        || lower.startsWith('deepseek:')
        || lower.startsWith('claude');
}

function modelOptionsHtml(models) {
    const localModels = models.filter(m => !isCloudModel(m));
    const cloudModels = models.filter(isCloudModel);
    let html = '';
    if (localModels.length > 0) {
        html += `<optgroup label="本地模型 (Ollama)">${
            localModels.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join('')
        }</optgroup>`;
    }
    if (cloudModels.length > 0) {
        html += `<optgroup label="云端模型">${
            cloudModels.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join('')
        }</optgroup>`;
    }
    return html;
}

function updateRoutingGroupVisibility() {
    const mode = document.getElementById('routingModeSelect').value;
    document.getElementById('singleModelGroup').style.display = mode === 'hybrid' ? 'none' : '';
    document.getElementById('hybridModelGroup').style.display = mode === 'hybrid' ? '' : 'none';
}

async function openSettings() {
    elements.settingsModal.classList.add('active');

    document.getElementById('autonomousCheckbox').checked = state.settings.autonomous !== false;

    const modelSelect = document.getElementById('modelSelect');
    const fastSelect = document.getElementById('fastModelSelect');
    const reasoningSelect = document.getElementById('reasoningModelSelect');
    const routingSelect = document.getElementById('routingModeSelect');

    [modelSelect, fastSelect, reasoningSelect].forEach(s => {
        s.innerHTML = '<option value="">正在加载模型列表...</option>';
        s.disabled = true;
    });

    try {
        const [modelsRes, configRes] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/config'),
        ]);
        if (!modelsRes.ok) throw new Error(`status ${modelsRes.status}`);
        const modelsData = await modelsRes.json();
        const config = configRes.ok ? await configRes.json() : {};
        const models = modelsData.models || [];

        if (models.length === 0) {
            [modelSelect, fastSelect, reasoningSelect].forEach(s => {
                s.innerHTML = '<option value="">没有可用模型</option>';
            });
            return;
        }

        const html = modelOptionsHtml(models);
        modelSelect.innerHTML = html;
        fastSelect.innerHTML = html;
        reasoningSelect.innerHTML = html;

        const currentModel = (state.settings.model && models.includes(state.settings.model))
            ? state.settings.model
            : (config.model || modelsData.current || '');
        if (currentModel) modelSelect.value = currentModel;
        if (config.fast_model) fastSelect.value = config.fast_model;
        if (config.reasoning_model) reasoningSelect.value = config.reasoning_model;

        routingSelect.value = config.routing_mode === 'hybrid' ? 'hybrid' : 'single';
        updateRoutingGroupVisibility();
    } catch (error) {
        console.error('Failed to load models:', error);
        [modelSelect, fastSelect, reasoningSelect].forEach(s => {
            s.innerHTML = '<option value="">模型列表加载失败</option>';
        });
    } finally {
        [modelSelect, fastSelect, reasoningSelect].forEach(s => { s.disabled = false; });
    }
}

function closeSettings() {
    elements.settingsModal.classList.remove('active');
}

async function saveSettings() {
    const routingMode = document.getElementById('routingModeSelect').value;
    const newModel = document.getElementById('modelSelect').value;
    const fastModel = document.getElementById('fastModelSelect').value;
    const reasoningModel = document.getElementById('reasoningModelSelect').value;

    state.settings = {
        model: newModel,
        autonomous: document.getElementById('autonomousCheckbox').checked,
    };
    localStorage.setItem('settings', JSON.stringify(state.settings));

    try {
        const body = routingMode === 'hybrid'
            ? { routing_mode: 'hybrid', fast_model: fastModel, reasoning_model: reasoningModel }
            : { routing_mode: 'single', model: newModel };

        const response = await fetch('/api/routing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `status ${response.status}`);
        }
    } catch (error) {
        console.error('Failed to update routing:', error);
        alert(`更新路由设置失败: ${error.message}`);
    }

    closeSettings();
}

function loadSettings() {
    const saved = localStorage.getItem('settings');
    if (saved) {
        state.settings = { ...state.settings, ...JSON.parse(saved) };
    }
}

function copyMessage(button) {
    const content = button.closest('.message').querySelector('.message-content').innerText;
    navigator.clipboard.writeText(content);
    
    const icon = button.querySelector('i');
    icon.className = 'fas fa-check';
    setTimeout(() => {
        icon.className = 'fas fa-copy';
    }, 2000);
}

function regenerateMessage(button) {
    // Get the last user message and resend
    const userMessages = state.messages.filter(m => m.role === 'user');
    if (userMessages.length > 0) {
        const lastMessage = userMessages[userMessages.length - 1];
        sendToAPI(lastMessage.content);
    }
}

function showAddMenu() {
    const existing = document.getElementById('addMenu');
    if (existing) {
        existing.remove();
        document.removeEventListener('click', handleAddMenuOutsideClick, true);
        return;
    }

    const menu = document.createElement('div');
    menu.className = 'add-menu';
    menu.id = 'addMenu';
    menu.innerHTML = `
        <button type="button" data-action="file"><i class="fas fa-file"></i> 上传文件</button>
        <button type="button" data-action="image"><i class="fas fa-image"></i> 上传图片</button>
        <button type="button" data-action="voice"><i class="fas fa-microphone"></i> 语音输入</button>
    `;
    menu.querySelector('[data-action="file"]').addEventListener('click', () => {
        closeAddMenu();
        uploadFile();
    });
    menu.querySelector('[data-action="image"]').addEventListener('click', () => {
        closeAddMenu();
        uploadImage();
    });
    menu.querySelector('[data-action="voice"]').addEventListener('click', () => {
        closeAddMenu();
        startVoiceInput();
    });

    elements.addButton.insertAdjacentElement('afterend', menu);
    setTimeout(() => document.addEventListener('click', handleAddMenuOutsideClick, true), 0);
}

function closeAddMenu() {
    const menu = document.getElementById('addMenu');
    if (menu) menu.remove();
    document.removeEventListener('click', handleAddMenuOutsideClick, true);
}

function handleAddMenuOutsideClick(e) {
    const menu = document.getElementById('addMenu');
    if (menu && !menu.contains(e.target) && e.target !== elements.addButton) {
        closeAddMenu();
    }
}

function startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('当前浏览器不支持语音输入，请使用最新版 Chrome。');
        return;
    }

    if (state.isRecording) {
        stopVoiceInput();
        return;
    }

    const baseText = elements.messageInput.value;
    const baseTextWithSpace = baseText && !/[\s\n]$/.test(baseText) ? baseText + ' ' : baseText;

    voiceRecognition = new SpeechRecognition();
    voiceRecognition.lang = 'zh-CN';
    voiceRecognition.continuous = false;
    voiceRecognition.interimResults = true;

    voiceRecognition.onstart = () => {
        state.isRecording = true;
        elements.voiceInput.classList.add('recording');
    };

    voiceRecognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        elements.messageInput.value = baseTextWithSpace + finalTranscript + interimTranscript;
        elements.messageInput.dispatchEvent(new Event('input'));
    };

    voiceRecognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopVoiceInput();
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
            alert('无法使用麦克风，请检查浏览器权限设置。');
        } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
            alert(`语音输入出错：${event.error}`);
        }
    };

    voiceRecognition.onend = () => {
        stopVoiceInput();
    };

    try {
        voiceRecognition.start();
    } catch (e) {
        console.error('Failed to start speech recognition:', e);
        stopVoiceInput();
    }
}

function stopVoiceInput() {
    state.isRecording = false;
    elements.voiceInput.classList.remove('recording');
    if (voiceRecognition) {
        try {
            voiceRecognition.stop();
        } catch (e) {
            // already stopped
        }
    }
}

function uploadImage() {
    pickAndUploadFile('image/*');
}

function uploadFile() {
    pickAndUploadFile('');
}

function pickAndUploadFile(accept) {
    const input = document.createElement('input');
    input.type = 'file';
    if (accept) input.accept = accept;
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            attachFile(file);
        }
    };
    input.click();
}

async function attachFile(file) {
    const placeholder = { filename: file.name, text: '', uploading: true };
    state.pendingAttachments.push(placeholder);
    renderAttachments();

    try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `上传失败 (status ${response.status})`);
        }

        const data = await response.json();
        placeholder.filename = data.filename || file.name;
        placeholder.text = data.block || `[Uploaded file: ${placeholder.filename}]`;
        placeholder.uploading = false;
        renderAttachments();
    } catch (error) {
        console.error('Upload error:', error);
        const index = state.pendingAttachments.indexOf(placeholder);
        if (index !== -1) state.pendingAttachments.splice(index, 1);
        renderAttachments();
        alert(`文件上传失败: ${error.message}`);
    }
}

function renderAttachments() {
    const bar = elements.attachmentBar;
    if (!bar) return;

    if (state.pendingAttachments.length === 0) {
        bar.style.display = 'none';
        bar.innerHTML = '';
        return;
    }

    bar.style.display = 'flex';
    bar.innerHTML = state.pendingAttachments.map((att, index) => `
        <span class="attachment-chip ${att.uploading ? 'uploading' : ''}">
            <i class="fas ${att.uploading ? 'fa-spinner fa-spin' : 'fa-paperclip'}"></i>
            <span class="attachment-chip-name">${escapeHtml(att.filename)}</span>
            <button type="button" class="attachment-chip-remove" data-index="${index}" title="移除附件">
                <i class="fas fa-times"></i>
            </button>
        </span>
    `).join('');

    bar.querySelectorAll('.attachment-chip-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            const index = parseInt(btn.getAttribute('data-index'), 10);
            state.pendingAttachments.splice(index, 1);
            renderAttachments();
        });
    });
}

function clearAttachments() {
    state.pendingAttachments = [];
    renderAttachments();
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
