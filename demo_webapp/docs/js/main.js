// GTMØ Constitutional Metrics Analyzer - Frontend JS

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000'; // Use 127.0.0.1 instead of localhost to avoid browser issues

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const textEditor = document.getElementById('textEditor');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressBar = document.getElementById('progressBar');
const useLLMCheckbox = document.getElementById('useLLM');

const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

const documentStats = document.getElementById('documentStats');
const metricsTableBody = document.getElementById('metricsTableBody');
const recommendationsContainer = document.getElementById('recommendationsContainer');

const searchInput = document.getElementById('searchInput');
const filterSelect = document.getElementById('filterSelect');

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const fileTab = document.getElementById('fileTab');
const textTab = document.getElementById('textTab');

// State
let currentData = null;
let allRows = [];

// Event Listeners
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = file.name;
    } else {
        fileName.textContent = 'Wybierz plik...';
    }
});

// Text editor character counter
textEditor.addEventListener('input', (e) => {
    const count = e.target.value.length;
    charCount.textContent = count;

    // Visual feedback when approaching limit
    if (count > 1100) {
        charCount.style.color = 'var(--danger-color)';
    } else if (count > 900) {
        charCount.style.color = 'var(--warning-color)';
    } else {
        charCount.style.color = 'var(--primary-color)';
    }
});

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        // Update active tab button
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Show/hide tab content
        if (tabName === 'file') {
            fileTab.classList.add('active');
            textTab.classList.remove('active');
        } else {
            fileTab.classList.remove('active');
            textTab.classList.add('active');
        }
    });
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await analyzeDocument();
});

searchInput.addEventListener('input', filterTable);
filterSelect.addEventListener('change', filterTable);

// Main Functions
async function analyzeDocument() {
    // Check which tab is active
    const isTextMode = textTab.classList.contains('active');
    let file;

    if (isTextMode) {
        // Text editor mode - create file from text
        const text = textEditor.value.trim();

        if (!text) {
            showError('Proszę wpisać tekst do analizy');
            return;
        }

        if (text.length < 10) {
            showError('Tekst jest za krótki (minimum 10 znaków)');
            return;
        }

        // Create a Blob from text and convert to File
        const blob = new Blob([text], { type: 'text/plain' });
        file = new File([blob], 'user_input.txt', { type: 'text/plain' });

    } else {
        // File upload mode
        file = fileInput.files[0];

        if (!file) {
            showError('Proszę wybrać plik');
            return;
        }

        // Validate file type
        const validExtensions = ['.txt', '.md'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(fileExt)) {
            showError('Nieprawidłowy typ pliku. Dozwolone są tylko pliki .txt i .md');
            return;
        }
    }

    // Show progress, hide sections
    analyzeBtn.disabled = true;
    progressBar.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);

        const useLLM = useLLMCheckbox.checked;

        console.log('Sending request to:', `${API_BASE_URL}/analyze?use_llm=${useLLM}`);
        console.log('File:', file.name, 'Size:', file.size);

        // Call API
        const response = await fetch(`${API_BASE_URL}/analyze?use_llm=${useLLM}`, {
            method: 'POST',
            body: formData,
            mode: 'cors',
            headers: {
                // Don't set Content-Type - browser will set it automatically with boundary
            }
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('Response data:', data);

        if (!data.success) {
            throw new Error(data.error || data.message || 'Analiza nie powiodła się');
        }

        // Store data and display results
        currentData = data;
        displayResults(data);

    } catch (error) {
        console.error('Full error:', error);

        // Detailed error message
        let errorMsg = error.message;
        if (error.message === 'Failed to fetch') {
            errorMsg = `Nie można połączyć z backendem (${API_BASE_URL}).

Sprawdź czy:
1. Backend działa: python demo_webapp/api/main.py
2. Port 8000 jest wolny
3. Otwórz konsolę przeglądarki (F12) aby zobaczyć szczegóły błędu`;
        }

        showError(errorMsg);
    } finally {
        analyzeBtn.disabled = false;
        progressBar.style.display = 'none';
    }
}

function displayResults(data) {
    // Hide error, show results
    errorSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Display document stats
    displayDocumentStats(data.aggregate_stats, data.document_metadata);

    // Display metrics table
    displayMetricsTable(data.metrics_table);

    // Display recommendations
    displayRecommendations(data.recommendations);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayDocumentStats(stats, metadata) {
    if (!stats) return;

    const statsHTML = `
        <div class="stat-item">
            <span class="stat-value">${stats.total_articles || 0}</span>
            <span class="stat-label">Artykuły</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${stats.total_sentences || 0}</span>
            <span class="stat-label">Zdania</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${stats.total_words || 0}</span>
            <span class="stat-label">Słowa</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${stats.average_SA || 0}%</span>
            <span class="stat-label">Średnia SA</span>
        </div>
        <div class="stat-item">
            <span class="stat-value" style="color: var(--danger-color);">${stats.critical_sentences || 0}</span>
            <span class="stat-label">Krytyczne (SA < 10%)</span>
        </div>
        <div class="stat-item">
            <span class="stat-value" style="color: var(--warning-color);">${stats.warning_sentences || 0}</span>
            <span class="stat-label">Ostrzeżenie (SA < 30%)</span>
        </div>
    `;

    documentStats.innerHTML = statsHTML;
}

function displayMetricsTable(metricsTable) {
    if (!metricsTable || metricsTable.length === 0) {
        metricsTableBody.innerHTML = '<tr><td colspan="12">Brak danych</td></tr>';
        return;
    }

    allRows = metricsTable;
    renderTableRows(metricsTable);
}

function renderTableRows(rows) {
    const html = rows.map(row => {
        // Determine SA class
        let saClass = 'sa-good';
        if (row.SA < 10) saClass = 'sa-critical';
        else if (row.SA < 30) saClass = 'sa-warning';

        return `
            <tr data-sa="${row.SA}" data-text="${row.full_text.toLowerCase()}">
                <td>${row.article || 'N/A'}</td>
                <td>${row.sentence_id || 'N/A'}</td>
                <td class="text-preview" title="${escapeHtml(row.full_text)}">${escapeHtml(row.text_preview)}</td>
                <td class="${saClass}">${row.SA}%</td>
                <td>${row.D}</td>
                <td>${row.S}</td>
                <td>${row.E}</td>
                <td>${row.CD}</td>
                <td>${row.CI}</td>
                <td>${row.depth}</td>
                <td>${row.ambiguity}</td>
                <td>${row.classification}</td>
            </tr>
        `;
    }).join('');

    metricsTableBody.innerHTML = html;
}

function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        recommendationsContainer.innerHTML = '<p>Brak rekomendacji - wszystkie przepisy są czytelne!</p>';
        return;
    }

    // Check if there's a success message
    if (recommendations[0].success && recommendations[0].problematic_count === 0) {
        recommendationsContainer.innerHTML = `
            <div class="recommendation-card" style="border-left-color: var(--success-color);">
                <h3 style="color: var(--success-color);">✅ ${recommendations[0].message}</h3>
            </div>
        `;
        return;
    }

    // Check for errors
    if (recommendations[0].error) {
        recommendationsContainer.innerHTML = `
            <div class="recommendation-card" style="border-left-color: var(--danger-color);">
                <h3 style="color: var(--danger-color);">❌ ${recommendations[0].message || 'Błąd generowania rekomendacji'}</h3>
                <p>${recommendations[0].error}</p>
            </div>
        `;
        return;
    }

    const html = recommendations.map((rec, index) => {
        const severityClass = rec.SA_percent < 10 ? 'severity-critical' : 'severity-warning';
        const severityLabel = rec.SA_percent < 10 ? 'KRYTYCZNE' : 'WYMAGA POPRAWY';

        return `
            <div class="recommendation-card">
                <div class="recommendation-header">
                    <div class="recommendation-id">Przepis #${index + 1} (ID: ${rec.sentence_id})</div>
                    <div class="recommendation-severity ${severityClass}">
                        ${severityLabel} - SA: ${rec.SA_percent}%
                    </div>
                </div>

                <div class="recommendation-text">
                    "${escapeHtml(rec.full_text)}"
                </div>

                <div class="recommendation-section">
                    <h4>🔍 Problem:</h4>
                    <p>Ten przepis jest <strong>${rec.severity}</strong>.</p>
                    <p>${rec.main_problem_detailed}</p>
                </div>

                <div class="recommendation-section">
                    <h4>⚡ Szybkie poprawki:</h4>
                    <ul>
                        ${rec.quick_fixes.map(fix => `<li>${escapeHtml(fix)}</li>`).join('')}
                    </ul>
                </div>

                <div class="recommendation-section">
                    <h4>📅 Zmiany długoterminowe:</h4>
                    <ul>
                        ${rec.long_term_fixes.map(fix => `<li>${escapeHtml(fix)}</li>`).join('')}
                    </ul>
                </div>

                ${rec.example_better_version ? `
                    <div class="example-better">
                        <h4>💡 Przykład lepszej wersji:</h4>
                        ${escapeHtml(rec.example_better_version)}
                    </div>
                ` : ''}

                <div class="legal-risks">
                    <h4>⚖️ Ryzyko prawne:</h4>
                    <p>${escapeHtml(rec.legal_risks)}</p>
                </div>
            </div>
        `;
    }).join('');

    recommendationsContainer.innerHTML = html;
}

function filterTable() {
    const searchTerm = searchInput.value.toLowerCase();
    const filterValue = filterSelect.value;

    const filteredRows = allRows.filter(row => {
        // Text search
        const matchesSearch = searchTerm === '' ||
            row.full_text.toLowerCase().includes(searchTerm) ||
            row.classification.toLowerCase().includes(searchTerm);

        // SA filter
        let matchesFilter = true;
        if (filterValue === 'critical') {
            matchesFilter = row.SA < 10;
        } else if (filterValue === 'warning') {
            matchesFilter = row.SA < 30;
        } else if (filterValue === 'good') {
            matchesFilter = row.SA >= 30;
        }

        return matchesSearch && matchesFilter;
    });

    renderTableRows(filteredRows);
}

function showError(message) {
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorMessage.textContent = message;

    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Initialize
console.log('GTMØ Analyzer initialized');
console.log('API URL:', API_BASE_URL);
