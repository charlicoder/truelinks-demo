/**
 * QCS 2024 Submittal Review Agent
 * Frontend JavaScript
 */

const API_BASE_URL = '';

// DOM Elements
const form = document.getElementById('submittal-form');
const submitBtn = document.getElementById('submit-btn');
const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const loadingStep = document.getElementById('loading-step');
const resultsContent = document.getElementById('results-content');

// Decision elements
const decisionBadge = document.getElementById('decision-badge');
const decisionIcon = document.getElementById('decision-icon');
const decisionText = document.getElementById('decision-text');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceValue = document.getElementById('confidence-value');

// Content elements
const explanationText = document.getElementById('explanation-text');
const recommendationsList = document.getElementById('recommendations-list');
const recommendationsSection = document.getElementById('recommendations-section');
const citationsList = document.getElementById('citations-list');
const analysisContent = document.getElementById('analysis-content');

// Loading step messages
const loadingSteps = [
    'Retrieving relevant QCS 2024 standards',
    'Analyzing submittal compliance',
    'Evaluating against requirements',
    'Generating decision and citations'
];

let loadingInterval = null;

// Show loading state
function showLoading() {
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
    submitBtn.disabled = true;

    let stepIndex = 0;
    loadingStep.textContent = loadingSteps[0];

    loadingInterval = setInterval(() => {
        stepIndex = (stepIndex + 1) % loadingSteps.length;
        loadingStep.textContent = loadingSteps[stepIndex];
    }, 2000);
}

// Hide loading state
function hideLoading() {
    loadingState.classList.add('hidden');
    submitBtn.disabled = false;

    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

// Show results
function showResults(data) {
    hideLoading();
    resultsContent.classList.remove('hidden');

    // Set decision badge
    const decision = data.decision.toUpperCase();
    decisionText.textContent = decision;
    decisionBadge.className = 'decision-badge';

    if (decision === 'APPROVED') {
        decisionBadge.classList.add('approved');
        decisionIcon.innerHTML = '✓';
    } else if (decision === 'REJECTED') {
        decisionBadge.classList.add('rejected');
        decisionIcon.innerHTML = '✗';
    } else {
        decisionBadge.classList.add('needs-review');
        decisionIcon.innerHTML = '?';
    }

    // Set confidence
    const confidence = Math.round(data.confidence * 100);
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;

    // Set explanation
    explanationText.textContent = data.explanation;

    // Set recommendations
    if (data.recommendations && data.recommendations.length > 0) {
        recommendationsSection.classList.remove('hidden');
        recommendationsList.innerHTML = data.recommendations
            .map(rec => `<li>${escapeHtml(rec)}</li>`)
            .join('');
    } else {
        recommendationsSection.classList.add('hidden');
    }

    // Set citations
    if (data.citations && data.citations.length > 0) {
        citationsList.innerHTML = data.citations.map(citation => `
            <div class="citation-item">
                <div class="citation-source">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14,2 14,8 20,8"/>
                    </svg>
                    ${escapeHtml(citation.source)}
                </div>
                <div class="citation-text">"${escapeHtml(citation.text)}"</div>
            </div>
        `).join('');
    } else {
        citationsList.innerHTML = '<p style="color: var(--text-muted);">No citations available</p>';
    }

    // Set full analysis
    analysisContent.textContent = data.analysis || 'No detailed analysis available.';
}

// Show error
function showError(message) {
    hideLoading();
    resultsContent.classList.remove('hidden');

    decisionBadge.className = 'decision-badge rejected';
    decisionIcon.innerHTML = '!';
    decisionText.textContent = 'ERROR';

    confidenceFill.style.width = '0%';
    confidenceValue.textContent = '0%';

    explanationText.textContent = message;
    recommendationsSection.classList.add('hidden');
    citationsList.innerHTML = '';
    analysisContent.textContent = '';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Submit form
async function handleSubmit(event) {
    event.preventDefault();

    const submittalType = document.getElementById('submittal-type').value;
    const description = document.getElementById('description').value;
    const specifications = document.getElementById('specifications').value;

    if (!submittalType || !description || !specifications) {
        alert('Please fill in all fields');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/api/review`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: submittalType,
                description: description,
                specifications: specifications
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        showResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to review submittal: ${error.message}`);
    }
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();

        if (data.knowledge_base_ready) {
            console.log(`API ready! Knowledge base has ${data.chunks_count} chunks.`);
        } else {
            console.warn('Knowledge base is not ready yet.');
        }
    } catch (error) {
        console.warn('API not available:', error.message);
    }
}

// Initialize
form.addEventListener('submit', handleSubmit);
checkHealth();
