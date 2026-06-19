/**
 * Legal RAG Chatbot — Frontend Logic
 * Interface minimaliste avec streaming SSE
 */

// === State ===
const state = {
    messages: [],
    isLoading: false,
    conversationHistory: [],
};

// === DOM Elements ===
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const docCount = document.getElementById('doc-count');

// === Initialization ===
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    autoResizeTextarea();
});

// === Event Listeners ===
function setupEventListeners() {
    userInput.addEventListener('input', () => {
        autoResizeTextarea();
        toggleSendButton();
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!state.isLoading && userInput.value.trim()) {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener('click', sendMessage);
}

function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
}

function toggleSendButton() {
    sendBtn.disabled = !userInput.value.trim() || state.isLoading;
}

function setExample(btn) {
    userInput.value = btn.textContent;
    autoResizeTextarea();
    toggleSendButton();
    userInput.focus();
}

// === Health Check ===
async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();

        if (data.status === 'healthy') {
            statusDot.classList.add('online');
            docCount.textContent = `${data.documents_indexed} doc${data.documents_indexed !== 1 ? 's' : ''}`;
        } else {
            statusDot.classList.add('offline');
        }
    } catch (err) {
        statusDot.classList.add('offline');
        console.error('Health check failed:', err);
    }
}

// === Message Rendering ===
function createMessageElement(role, content = '', isStreaming = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.dataset.role = role;

    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${role}`;
    avatar.textContent = role === 'user' ? 'V' : '⚖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const header = document.createElement('div');
    header.className = 'message-header';

    const author = document.createElement('span');
    author.className = 'message-author';
    author.textContent = role === 'user' ? 'Vous' : 'Assistant Juridique';

    const time = document.createElement('span');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

    header.appendChild(author);
    header.appendChild(time);

    const body = document.createElement('div');
    body.className = 'message-body';
    body.id = `msg-body-${Date.now()}`;

    if (isStreaming && role === 'assistant') {
        const typing = document.createElement('div');
        typing.className = 'typing-indicator';
        typing.innerHTML = '<span></span><span></span><span></span>';
        body.appendChild(typing);
    } else {
        body.innerHTML = formatMessage(content);
    }

    contentDiv.appendChild(header);
    contentDiv.appendChild(body);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);

    return { element: msgDiv, bodyElement: body };
}

function formatMessage(text) {
    if (!text) return '';

    // Échapper le HTML
    let safe = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Formatage markdown léger
    // Gras
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italique
    safe = safe.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Titres
    safe = safe.replace(/^###\s+(.+)$/gm, '<h3 style="color:var(--accent);font-size:1rem;margin:12px 0 6px;">$1</h3>');
    safe = safe.replace(/^##\s+(.+)$/gm, '<h2 style="color:var(--accent);font-size:1.1rem;margin:14px 0 8px;">$1</h2>');
    safe = safe.replace(/^#\s+(.+)$/gm, '<h1 style="color:var(--accent);font-size:1.2rem;margin:16px 0 10px;">$1</h1>');
    // Listes
    safe = safe.replace(/^-\s+(.+)$/gm, '• $1');
    safe = safe.replace(/^\d+\.\s+(.+)$/gm, '$1');
    // Lignes
    safe = safe.replace(/\n/g, '<br>');

    return safe;
}

function appendSources(bodyElement, sources) {
    if (!sources || sources.length === 0) return;

    const sourcesSection = document.createElement('div');
    sourcesSection.className = 'sources-section';

    const title = document.createElement('div');
    title.className = 'sources-title';
    title.textContent = 'Sources';
    sourcesSection.appendChild(title);

    sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';

        const doc = document.createElement('span');
        doc.className = 'source-doc';
        doc.textContent = src.document || 'Inconnu';

        const ref = document.createElement('span');
        ref.className = 'source-ref';
        const parts = [];
        if (src.section) parts.push(src.section);
        if (src.article) parts.push(`Art. ${src.article}`);
        if (src.alinea) parts.push(src.alinea);
        ref.textContent = parts.join(' | ') || 'Référence générale';

        const sim = document.createElement('span');
        sim.className = 'source-similarity';
        sim.textContent = src.similarity !== undefined ? `${(src.similarity * 100).toFixed(0)}%` : '';

        item.appendChild(doc);
        item.appendChild(ref);
        if (src.similarity !== undefined) item.appendChild(sim);
        sourcesSection.appendChild(item);
    });

    bodyElement.appendChild(sourcesSection);
}

// === Send Message ===
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || state.isLoading) return;

    // Hide welcome
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.style.display = 'none';

    // Add user message
    const userMsg = createMessageElement('user', text);
    chatContainer.appendChild(userMsg.element);
    scrollToBottom();

    // Update state
    state.messages.push({ role: 'user', content: text });
    state.isLoading = true;
    toggleSendButton();

    // Clear input
    userInput.value = '';
    autoResizeTextarea();

    // Add assistant placeholder
    const assistantMsg = createMessageElement('assistant', '', true);
    chatContainer.appendChild(assistantMsg.element);
    scrollToBottom();

    // Stream response
    let fullResponse = '';
    let sources = [];

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: state.conversationHistory,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // Remove typing indicator
        assistantMsg.bodyElement.innerHTML = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);

                    if (data === '[DONE]') {
                        break;
                    }

                    if (data.startsWith('[ERROR]')) {
                        throw new Error(data.slice(8));
                    }

                    // Unescape newlines
                    const token = data.replace(/\\n/g, '\n');
                    fullResponse += token;
                    assistantMsg.bodyElement.innerHTML = formatMessage(fullResponse);
                    scrollToBottom();
                }
            }
        }

        // Fetch sources separately
        try {
            const sourcesRes = await fetch(`/api/sources?query=${encodeURIComponent(text)}`);
            if (sourcesRes.ok) {
                const sourcesData = await sourcesRes.json();
                sources = sourcesData.sources || [];
                appendSources(assistantMsg.bodyElement, sources);
                scrollToBottom();
            }
        } catch (e) {
            console.error('Failed to fetch sources:', e);
        }

        // Update conversation history
        state.conversationHistory.push({ role: 'user', content: text });
        state.conversationHistory.push({ role: 'assistant', content: fullResponse });

        // Keep only last 6 messages
        if (state.conversationHistory.length > 6) {
            state.conversationHistory = state.conversationHistory.slice(-6);
        }

    } catch (err) {
        console.error('Chat error:', err);
        assistantMsg.bodyElement.innerHTML = `
            <div class="error-message">
                ⚠️ Une erreur est survenue lors de la génération de la réponse. 
                Veuillez réessayer.<br><small>${err.message}</small>
            </div>
        `;
    } finally {
        state.isLoading = false;
        toggleSendButton();
        userInput.focus();
    }
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
