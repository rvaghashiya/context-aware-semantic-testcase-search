// Application Data
const testCases = [
    {"id": 1, "name": "test_user_login_success", "description": "Test successful user authentication with valid credentials", "category": "Authentication"},
    {"id": 2, "name": "verify_authentication_valid_creds", "description": "Verify user authentication using valid login credentials", "category": "Authentication"},
    {"id": 3, "name": "check_user_signin_functionality", "description": "Check that user can sign in with correct username and password", "category": "Authentication"},
    {"id": 4, "name": "test_login_invalid_password", "description": "Test login fails with invalid password", "category": "Authentication"},
    {"id": 5, "name": "validate_payment_processing", "description": "Validate payment processing with credit card information", "category": "Payment"},
    {"id": 6, "name": "test_payment_gateway_integration", "description": "Test integration with external payment gateway service", "category": "Payment"},
    {"id": 7, "name": "verify_payment_validation", "description": "Verify payment amount and currency validation", "category": "Payment"},
    {"id": 8, "name": "test_order_creation_workflow", "description": "Test complete order creation from cart to confirmation", "category": "Orders"},
    {"id": 9, "name": "verify_order_status_updates", "description": "Verify order status changes correctly through workflow", "category": "Orders"},
    {"id": 10, "name": "test_order_cancellation", "description": "Test user can cancel order before processing", "category": "Orders"},
    {"id": 11, "name": "validate_data_input_rules", "description": "Validate data input follows business rules and constraints", "category": "Validation"},
    {"id": 12, "name": "test_form_validation_errors", "description": "Test form shows appropriate validation error messages", "category": "Validation"},
    {"id": 13, "name": "verify_data_sanitization", "description": "Verify user input is properly sanitized against XSS", "category": "Validation"},
    {"id": 14, "name": "test_api_response_format", "description": "Test API returns data in expected JSON format", "category": "API"},
    {"id": 15, "name": "verify_api_error_handling", "description": "Verify API handles errors and returns proper status codes", "category": "API"},
    {"id": 16, "name": "test_api_authentication", "description": "Test API requires proper authentication tokens", "category": "API"},
    {"id": 17, "name": "validate_database_queries", "description": "Validate database queries return correct data", "category": "Database"},
    {"id": 18, "name": "test_data_persistence", "description": "Test data is properly saved and persisted", "category": "Database"},
    {"id": 19, "name": "verify_transaction_rollback", "description": "Verify database transactions rollback on failure", "category": "Database"},
    {"id": 20, "name": "test_ui_element_visibility", "description": "Test UI elements are visible and properly rendered", "category": "UI"}
];

const embeddings2D = [
    [-2.1, 1.5], [-1.8, 1.3], [-2.0, 1.7], [-1.5, 1.2],
    [2.2, -1.1], [2.0, -0.8], [1.9, -1.3],
    [0.5, 2.1], [0.3, 1.9], [0.7, 2.3],
    [-0.8, -2.0], [-0.5, -1.8], [-0.9, -2.2],
    [1.8, 0.2], [2.1, 0.5], [1.6, 0.0],
    [-1.2, 0.8], [-0.9, 1.1], [-1.4, 0.5],
    [0.2, -0.5]
];

const clusterLabels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6];

const models = {
    "elmo": {"embedding_dim": 1024, "accuracy": 0.847, "silhouette": 0.485, "davies_bouldin": 0.742},
    "bert": {"embedding_dim": 384, "accuracy": 0.892, "silhouette": 0.521, "davies_bouldin": 0.698},
    "t5": {"embedding_dim": 512, "accuracy": 0.824, "silhouette": 0.467, "davies_bouldin": 0.789}
};

const architectureComponents = [
    {"layer": "Data Layer", "components": ["Methods2Test Dataset", "Text Preprocessor", "Synthetic Generator"], "description": "Data ingestion and preprocessing pipeline"},
    {"layer": "Model Layer", "components": ["ELMo Embedder", "BERT Embedder", "T5 Embedder"], "description": "State-of-the-art NLP models for semantic embedding generation"},
    {"layer": "Analysis Layer", "components": ["Similarity Engine", "Clustering Engine", "Dimensionality Reduction"], "description": "Core clustering and similarity analysis algorithms"},
    {"layer": "Visualization Layer", "components": ["Interactive Plots", "3D Visualizations", "Static Charts"], "description": "Rich visualization and exploration interfaces"},
    {"layer": "Interface Layer", "components": ["FastAPI Backend", "Streamlit Frontend", "REST APIs"], "description": "User interfaces and API endpoints"}
];

const clusterColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C'];

// Global variables
let clusterChart = null;
let metricsChart = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    initializeTabs();
    initializeMetrics();
    initializeArchitecture();
    setupEventListeners();
});

function initializeTabs() {
    console.log('Initializing tabs...');
    const tabButtons = document.querySelectorAll('.nav__tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    console.log('Found tab buttons:', tabButtons.length);
    console.log('Found tab contents:', tabContents.length);
    
    tabButtons.forEach((button, index) => {
        console.log(`Setting up tab button ${index}:`, button.getAttribute('data-tab'));
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTab = button.getAttribute('data-tab');
            console.log('Tab clicked:', targetTab);
            switchTab(targetTab);
        });
    });
}

function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    const tabButtons = document.querySelectorAll('.nav__tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Update active tab button
    tabButtons.forEach(btn => {
        btn.classList.remove('nav__tab--active');
        if (btn.getAttribute('data-tab') === tabName) {
            btn.classList.add('nav__tab--active');
        }
    });
    
    // Update active content
    tabContents.forEach(content => {
        content.classList.remove('tab-content--active');
        if (content.id === tabName) {
            content.classList.add('tab-content--active');
        }
    });

    // Initialize tab-specific content
    if (tabName === 'demo') {
        setTimeout(() => {
            if (!clusterChart) {
                generateClusters();
            }
        }, 100);
    } else if (tabName === 'metrics') {
        setTimeout(() => {
            if (!metricsChart) {
                initializeMetricsChart();
            }
        }, 100);
    }
}

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    const generateBtn = document.getElementById('generateBtn');
    const searchBtn = document.getElementById('searchBtn');
    const searchInput = document.getElementById('searchInput');
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateClusters);
        console.log('Generate button listener added');
    }
    
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
        console.log('Search button listener added');
    }
    
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });
        console.log('Search input listener added');
    }
}

function generateClusters() {
    console.log('Generating clusters...');
    const modelSelect = document.getElementById('modelSelect');
    const loadingState = document.getElementById('loadingState');
    const generateBtn = document.getElementById('generateBtn');
    
    if (!modelSelect) {
        console.error('Model select not found');
        return;
    }
    
    const selectedModel = modelSelect.value;
    console.log('Selected model:', selectedModel);
    
    // Show loading state
    if (loadingState) {
        loadingState.classList.remove('hidden');
    }
    if (generateBtn) {
        generateBtn.disabled = true;
    }
    
    // Simulate processing time
    setTimeout(() => {
        createClusterVisualization(selectedModel);
        if (loadingState) {
            loadingState.classList.add('hidden');
        }
        if (generateBtn) {
            generateBtn.disabled = false;
        }
    }, 1500);
}

function createClusterVisualization(selectedModel) {
    console.log('Creating cluster visualization for:', selectedModel);
    const canvas = document.getElementById('clusterChart');
    if (!canvas) {
        console.error('Cluster chart canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    if (clusterChart) {
        clusterChart.destroy();
    }

    // Prepare data for Chart.js
    const datasets = [];
    const uniqueClusters = [...new Set(clusterLabels)];
    
    uniqueClusters.forEach(cluster => {
        const clusterData = testCases
            .map((testCase, index) => ({
                x: embeddings2D[index][0],
                y: embeddings2D[index][1],
                testCase: testCase,
                cluster: clusterLabels[index]
            }))
            .filter(item => item.cluster === cluster);

        datasets.push({
            label: `Cluster ${cluster} (${testCases.filter((_, idx) => clusterLabels[idx] === cluster)[0]?.category || 'Mixed'})`,
            data: clusterData,
            backgroundColor: clusterColors[cluster % clusterColors.length],
            borderColor: clusterColors[cluster % clusterColors.length],
            pointRadius: 8,
            pointHoverRadius: 12
        });
    });

    clusterChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Test Case Clustering - ${selectedModel.toUpperCase()} Model (${models[selectedModel].embedding_dim}D → 2D via t-SNE)`,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return context[0].raw.testCase.name;
                        },
                        label: function(context) {
                            const testCase = context.raw.testCase;
                            return [
                                `Category: ${testCase.category}`,
                                `Description: ${testCase.description}`,
                                `Position: (${context.raw.x.toFixed(2)}, ${context.raw.y.toFixed(2)})`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 't-SNE Dimension 1'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 't-SNE Dimension 2'
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const dataIndex = elements[0].index;
                    const datasetIndex = elements[0].datasetIndex;
                    const testCase = clusterChart.data.datasets[datasetIndex].data[dataIndex].testCase;
                    showTestCaseDetails(testCase);
                }
            }
        }
    });
    
    console.log('Cluster chart created successfully');
}

function showTestCaseDetails(testCase) {
    const detailsContainer = document.getElementById('testCaseDetails');
    if (!detailsContainer) return;
    
    detailsContainer.innerHTML = `
        <h3>Test Case Details</h3>
        <div class="test-case-info">
            <p><strong>Name:</strong> ${testCase.name}</p>
            <p><strong>Category:</strong> <span class="search-result__category">${testCase.category}</span></p>
            <p><strong>Description:</strong> ${testCase.description}</p>
            <p><strong>ID:</strong> ${testCase.id}</p>
        </div>
    `;
}

function performSearch() {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) return;
    
    const query = searchInput.value.trim();
    if (!query) return;

    console.log('Performing search for:', query);
    const resultsContainer = document.getElementById('searchResults');
    if (!resultsContainer) return;
    
    // Simulate search processing
    resultsContainer.innerHTML = '<h3>Search Results</h3><p>Searching...</p>';
    
    setTimeout(() => {
        const results = findSimilarTestCases(query);
        displaySearchResults(results, query);
    }, 800);
}

function findSimilarTestCases(query) {
    // Simple similarity calculation based on keywords
    const queryWords = query.toLowerCase().split(' ');
    
    const scoredResults = testCases.map(testCase => {
        const text = `${testCase.name} ${testCase.description} ${testCase.category}`.toLowerCase();
        
        let score = 0;
        queryWords.forEach(word => {
            if (text.includes(word)) {
                score += 0.3;
            }
            // Bonus for exact matches
            if (testCase.name.toLowerCase().includes(word)) {
                score += 0.2;
            }
            if (testCase.category.toLowerCase().includes(word)) {
                score += 0.25;
            }
        });
        
        // Add some randomness to simulate semantic similarity
        score += Math.random() * 0.1;
        
        return { ...testCase, similarity: Math.min(score, 0.95) };
    });

    return scoredResults
        .filter(result => result.similarity > 0.1)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 8);
}

function displaySearchResults(results, query) {
    const resultsContainer = document.getElementById('searchResults');
    if (!resultsContainer) return;
    
    if (results.length === 0) {
        resultsContainer.innerHTML = `
            <h3>Search Results</h3>
            <p>No similar test cases found for "${query}". Try different keywords.</p>
        `;
        return;
    }

    const resultsHTML = results.map(result => `
        <div class="search-result">
            <div class="search-result__header">
                <h4 class="search-result__name">${result.name}</h4>
                <span class="similarity-score">${(result.similarity * 100).toFixed(1)}%</span>
            </div>
            <p class="search-result__description">${result.description}</p>
            <span class="search-result__category">${result.category}</span>
        </div>
    `).join('');

    resultsContainer.innerHTML = `
        <h3>Search Results for "${query}"</h3>
        ${resultsHTML}
    `;
}

function initializeMetrics() {
    console.log('Initializing metrics...');
    const tableBody = document.getElementById('metricsTable');
    if (!tableBody) {
        console.error('Metrics table body not found');
        return;
    }
    
    const tableHTML = Object.entries(models).map(([modelName, metrics]) => `
        <tr>
            <td><strong>${modelName.toUpperCase()}</strong></td>
            <td>${metrics.embedding_dim}</td>
            <td>${(metrics.accuracy * 100).toFixed(1)}%</td>
            <td>${metrics.silhouette.toFixed(3)}</td>
            <td>${metrics.davies_bouldin.toFixed(3)}</td>
        </tr>
    `).join('');
    
    tableBody.innerHTML = tableHTML;
    console.log('Metrics table populated');
}

function initializeMetricsChart() {
    console.log('Initializing metrics chart...');
    const canvas = document.getElementById('metricsChart');
    if (!canvas) {
        console.error('Metrics chart canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    if (metricsChart) {
        metricsChart.destroy();
    }

    const modelNames = Object.keys(models).map(name => name.toUpperCase());
    const accuracyData = Object.values(models).map(m => m.accuracy * 100);
    const silhouetteData = Object.values(models).map(m => m.silhouette * 100);

    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracyData,
                    backgroundColor: '#1FB8CD',
                    borderColor: '#1FB8CD',
                    borderWidth: 1
                },
                {
                    label: 'Silhouette Score (×100)',
                    data: silhouetteData,
                    backgroundColor: '#FFC185',
                    borderColor: '#FFC185',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    font: { size: 14, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Model'
                    }
                }
            }
        }
    });
    
    console.log('Metrics chart created successfully');
}

function initializeArchitecture() {
    console.log('Initializing architecture...');
    const componentsContainer = document.getElementById('architectureComponents');
    if (!componentsContainer) {
        console.error('Architecture components container not found');
        return;
    }
    
    const componentsHTML = architectureComponents.map(layer => `
        <div class="component-layer">
            <h3>${layer.layer}</h3>
            <p>${layer.description}</p>
            <div class="component-items">
                ${layer.components.map(component => 
                    `<span class="component-item">${component}</span>`
                ).join('')}
            </div>
        </div>
    `).join('');
    
    componentsContainer.innerHTML = componentsHTML;
    console.log('Architecture components populated');
}

// Add some interactive enhancements
document.addEventListener('DOMContentLoaded', function() {
    console.log('Adding interactive enhancements...');
    
    // Add hover effects to feature cards
    setTimeout(() => {
        const featureCards = document.querySelectorAll('.feature-card');
        featureCards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-4px) scale(1.02)';
            });
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
        });
        console.log('Feature card hover effects added');
    }, 100);

    // Add search suggestions
    setTimeout(() => {
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            const suggestions = ['user authentication', 'payment processing', 'data validation', 'API testing', 'database queries', 'order workflow'];
            let currentSuggestion = 0;
            
            searchInput.addEventListener('focus', function() {
                if (!this.value) {
                    this.placeholder = `Try: "${suggestions[currentSuggestion]}"`;
                    currentSuggestion = (currentSuggestion + 1) % suggestions.length;
                }
            });
            console.log('Search suggestions added');
        }
    }, 100);
});