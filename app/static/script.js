/**
 * script.js
 * Frontend logic for SlopRadar.
 * Handles UI interactions, micro-animations, and communicates with the 
 * FastAPI /predict endpoint to display real-time classification results.
 */
const textArea = document.getElementById('text-input');
const charCount = document.getElementById('char-count');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingState = document.getElementById('loading-state');
const resultsPanel = document.getElementById('results-panel');

// UI Elements for results
const predictionTitle = document.getElementById('prediction-title');
const confidenceBadge = document.getElementById('confidence-badge');
const progressFill = document.getElementById('progress-fill');

textArea.addEventListener('input', () => {
    const len = textArea.value.length;
    charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
    analyzeBtn.disabled = len < 10;
});

analyzeBtn.addEventListener('click', async () => {
    const text = textArea.value.trim();
    if (!text) return;

    // UI Updates
    analyzeBtn.disabled = true;
    resultsPanel.classList.add('hidden');
    loadingState.classList.remove('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        
        // Hide loading
        loadingState.classList.add('hidden');
        
        // Update results
        const isAI = data.is_ai;
        const confidence = isAI ? data.ai_probability : data.human_probability;
        
        resultsPanel.classList.remove('hidden');
        resultsPanel.classList.remove('is-ai');
        if (isAI) {
            resultsPanel.classList.add('is-ai');
        }

        predictionTitle.textContent = isAI ? 'AI Generated Slop' : 'Human Written';
        confidenceBadge.textContent = `${confidence.toFixed(1)}% Confidence`;
        
        // The bar goes from Human (0%) to AI (100%)
        // We set width based on AI probability
        setTimeout(() => {
            progressFill.style.width = `${data.ai_probability}%`;
        }, 50);

    } catch (error) {
        alert("Error connecting to the detector model. Please try again.");
        loadingState.classList.add('hidden');
    } finally {
        analyzeBtn.disabled = false;
    }
});
