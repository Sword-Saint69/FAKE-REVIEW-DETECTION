document.addEventListener('DOMContentLoaded', function() {
    const reviewText = document.getElementById('reviewText');
    const checkButton = document.getElementById('checkButton');
    const clearButton = document.getElementById('clearButton');
    const resultContainer = document.getElementById('resultContainer');
    const resultDisplay = document.getElementById('resultDisplay');
    const confidenceScore = document.getElementById('confidenceScore');
    const analysisDetails = document.getElementById('analysisDetails');
    const loader = document.getElementById('loader');
    
    // Backend API URL
    // API_URL correct aayi set cheyyuka
    const API_URL = 'http://localhost:5001';
    
    // Page load cheyyumbol models trained aayittundo enn check cheyyuka
    checkModelsStatus();
    
    // Backend connection test cheyyuka
    testBackendConnection();
    
    checkButton.addEventListener('click', function() {
        if (reviewText.value.trim() === '') {
            alert('Please enter a review to analyze');
            return;
        }
        
        // Loading state kanikkuka
        resultDisplay.style.display = 'none';
        confidenceScore.textContent = '';
        analysisDetails.textContent = '';
        resultContainer.style.display = 'block';
        loader.style.display = 'flex';
        
        // Backend API call cheyyuka
        fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                review_text: reviewText.value
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            loader.style.display = 'none';
            resultDisplay.style.display = 'flex';
            displayResults(data);
        })
        .catch(error => {
            loader.style.display = 'none';
            resultDisplay.style.display = 'flex';
            
            console.error('Error:', error);
            
            // API fail cheyyumbol client-side analysis fallback cheyyuka
            alert('Could not connect to the backend server. Falling back to basic analysis.');
            analyzeReviewClientSide(reviewText.value);
        });
    });
    
    clearButton.addEventListener('click', function() {
        reviewText.value = '';
        resultContainer.style.display = 'none';
    });
    
    function checkModelsStatus() {
        // Models directory access cheyyan try cheyyuka
        fetch(`${API_URL}/dataset_stats`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Models not found');
                }
                return response.json();
            })
            .then(data => {
                console.log('Dataset statistics:', data);
                // Models ready aayittundenn UI update cheyyuka
                updateModelStatus(true, data);
            })
            .catch(error => {
                console.error('Error checking models:', error);
                // Error state kanikkunnathinu vendi UI update cheyyuka
                updateModelStatus(false);
            });
    }
    
    function updateModelStatus(modelsReady, data = null) {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (modelsReady) {
            // Models trained aayi
            statusIndicator.className = 'status-indicator trained';
            statusText.textContent = 'Models are trained and ready';
            
            // Stats sectionil accuracy update cheyyuka, available aayittundenkil
            if (data && data.average_accuracy) {
                const accuracyElement = document.getElementById('modelAccuracy');
                if (accuracyElement) {
                    accuracyElement.textContent = `${(data.average_accuracy * 100).toFixed(1)}%`;
                }
            }
        } else {
            // Models kittiyilla athava error
            statusIndicator.className = 'status-indicator untrained';
            statusText.textContent = 'Error: Models not found';
        }
    }
    
    function testBackendConnection() {
        fetch(`${API_URL}/ping`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Server responded with an error');
                }
                return response.json();
            })
            .then(data => {
                console.log('Backend connection successful:', data);
                document.getElementById('connectionStatus').textContent = 'Connected to backend server';
                document.getElementById('connectionStatus').className = 'connected';
            })
            .catch(error => {
                console.error('Backend connection failed:', error);
                document.getElementById('connectionStatus').textContent = 'Could not connect to the backend server. Falling back to basic analysis.';
                document.getElementById('connectionStatus').className = 'disconnected';
            });
    }
    
    function displayResults(data) {
        const prediction = data.ensemble_prediction;
        const confidence = Math.round(prediction.confidence * 100);
        
        if (prediction.label === 'fake') {
            resultDisplay.innerHTML = '<i class="fas fa-times-circle"></i> Likely Computer-Generated (Fake) Review';
            resultDisplay.className = 'fake';
        } else {
            resultDisplay.innerHTML = '<i class="fas fa-check-circle"></i> Likely Human-Written (Genuine) Review';
            resultDisplay.className = 'genuine';
        }
        
        confidenceScore.textContent = `Confidence: ${confidence}%`;
        
        // Detailed analysis kanikkuka
        let details = '<strong>Analysis Details:</strong><br><br>';
        details += '<div class="algorithm-results">';
        
        // Individual algorithm results add cheyyuka
        for (const [algorithm, result] of Object.entries(data.individual_predictions)) {
            const algoConfidence = Math.round(result.confidence * 100);
            const algoName = algorithm.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            
            details += `<div class="algorithm">`;
            details += `<strong>${algoName}:</strong> `;
            details += `<span class="${result.label}">${result.label === 'fake' ? 'Computer-Generated' : 'Human-Written'}</span> `;
            details += `(${algoConfidence}% confidence)`;
            details += `</div>`;
        }
        
        details += '</div>';
        
        // Key features section add cheyyuka
        if (data.key_features) {
            details += '<br><strong>Key Features Detected:</strong><br>';
            
            // Top TF-IDF features
            if (data.key_features.top_features && data.key_features.top_features.length > 0) {
                details += '<div class="feature-section">';
                details += '<strong>Most Significant Terms:</strong>';
                details += '<ul>';
                data.key_features.top_features.forEach(feature => {
                    details += `<li>${feature[0]}</li>`;
                });
                details += '</ul>';
                details += '</div>';
            }
            
            // Fake indicators
            if (data.key_features.fake_indicators && data.key_features.fake_indicators.length > 0) {
                details += '<div class="feature-section fake-indicators">';
                details += '<strong>Potential Computer-Generated Indicators:</strong>';
                details += '<ul>';
                data.key_features.fake_indicators.forEach(indicator => {
                    details += `<li>${indicator}</li>`;
                });
                details += '</ul>';
                details += '</div>';
            }
            
            // Genuine indicators
            if (data.key_features.genuine_indicators && data.key_features.genuine_indicators.length > 0) {
                details += '<div class="feature-section genuine-indicators">';
                details += '<strong>Potential Human-Written Indicators:</strong>';
                details += '<ul>';
                data.key_features.genuine_indicators.forEach(indicator => {
                    details += `<li>${indicator}</li>`;
                });
                details += '</ul>';
                details += '</div>';
            }
        }
        
        // Prediction based explanation add cheyyuka
        details += '<br><strong>Explanation:</strong><br>';
        if (prediction.label === 'fake') {
            details += 'This review shows characteristics commonly found in computer-generated reviews, such as:';
            details += '<ul>';
            details += '<li>Excessive use of superlatives or extreme language</li>';
            details += '<li>Lack of specific details about the product/service</li>';
            details += '<li>Generic descriptions that could apply to many products</li>';
            details += '<li>Repetitive patterns or formulaic language</li>';
            details += '<li>Inconsistencies in the review content</li>';
            details += '</ul>';
        } else {
            details += 'This review shows characteristics commonly found in human-written reviews, such as:';
            details += '<ul>';
            details += '<li>Balanced opinions (mentioning both pros and cons)</li>';
            details += '<li>Specific details about the product/service</li>';
            details += '<li>Personal experiences and context</li>';
            details += '<li>Nuanced language and realistic expectations</li>';
            details += '<li>Coherent narrative structure</li>';
            details += '</ul>';
        }
        
        analysisDetails.innerHTML = details;
        
        // Performance visualization update cheyyuka
        if (data.individual_predictions) {
            updatePerformanceBars(data);
            document.getElementById('modelPerformance').style.display = 'block';
        }
    }
    
    // Fallback client-side analysis function (originalinte simplified version)
    function analyzeReviewClientSide(text) {
        // Demonstrationinu vendi simplified fake detection algorithm
        // Real applicationil, machine learning athava API upayogikkuka
        
        const review = text.toLowerCase();
        
        // Fake review indicators (demo vendi simplified)
        const fakeIndicators = [
            { pattern: /too good to be true/i, weight: 0.7 },
            { pattern: /amazing amazing/i, weight: 0.6 },
            { pattern: /best (ever|thing)/i, weight: 0.3 },
            { pattern: /(never|will never) (use|buy|shop).+again/i, weight: 0.5 },
            { pattern: /worst (ever|experience)/i, weight: 0.4 },
            { pattern: /!{3,}/i, weight: 0.4 }, // Multiple exclamation marks
            { pattern: /\b(perfect|perfectly|amazing|awesome|excellent|outstanding)\b/gi, weight: 0.2 },
            { pattern: /\b(terrible|horrible|awful|worst)\b/gi, weight: 0.2 },
            { pattern: /\b(free|discount|offer|deal|promotion)\b/gi, weight: 0.3 },
            { pattern: /\b(buy|purchase|order|recommend)\b/gi, weight: 0.1 }
        ];
        
        // Genuine review indicators
        const genuineIndicators = [
            { pattern: /\b(however|though|although|but)\b/i, weight: 0.4 }, // Balanced opinion
            { pattern: /\b(pros|cons)\b/i, weight: 0.5 }, // Structured review
            { pattern: /\b(specifically|particular|detail)\b/i, weight: 0.3 }, // Specific details
            { pattern: /\d+ (days|weeks|months|years)/i, weight: 0.4 }, // Time references
            { pattern: /\b(slightly|somewhat|fairly|quite|rather)\b/i, weight: 0.3 } // Nuanced opinion
        ];
        
        // Extreme review length check cheyyuka
        const wordCount = review.split(/\s+/).length;
        let lengthScore = 0;
        if (wordCount < 10) lengthScore = 0.3; // Too short
        if (wordCount > 500) lengthScore = 0.3; // Too long
        
        // Fake score calculate cheyyuka
        let fakeScore = 0;
        let fakeMatches = [];
        
        fakeIndicators.forEach(indicator => {
            const matches = review.match(indicator.pattern) || [];
            if (matches.length > 0) {
                fakeScore += indicator.weight * matches.length;
                fakeMatches.push(`Found "${matches[0]}" (${(indicator.weight * matches.length).toFixed(2)} points)`);
            }
        });
        
        // Genuine score calculate cheyyuka
        let genuineScore = 0;
        let genuineMatches = [];
        
        genuineIndicators.forEach(indicator => {
            const matches = review.match(indicator.pattern) || [];
            if (matches.length > 0) {
                genuineScore += indicator.weight * matches.length;
                genuineMatches.push(`Found "${matches[0]}" (${(indicator.weight * matches.length).toFixed(2)} points)`);
            }
        });
        
        // Length penalty add cheyyuka, applicable aayittundenkil
        if (lengthScore > 0) {
            fakeScore += lengthScore;
            fakeMatches.push(`Review length: ${wordCount} words (${lengthScore.toFixed(2)} points)`);
        }
        
        // Final score calculate cheyyuka (0-1 scale, higher means more likely fake)
        const totalScore = fakeScore + genuineScore;
        let fakeProbability = 0.5; // Default neutral
        
        if (totalScore > 0) {
            fakeProbability = fakeScore / totalScore;
        }
        
        // Very short reviews with no signals adjust cheyyuka
        if (totalScore < 0.3 && wordCount < 20) {
            fakeProbability = 0.6; // Very short reviews with no signals vendi slightly lean toward fake
        }
        
        // Results display cheyyuka
        const confidencePercentage = Math.abs((fakeProbability - 0.5) * 2 * 100).toFixed(0);
        
        if (fakeProbability > 0.55) {
            resultDisplay.innerHTML = '<i class="fas fa-times-circle"></i> Likely Computer-Generated (Fake) Review';
            resultDisplay.className = 'fake';
            confidenceScore.textContent = `Confidence: ${confidencePercentage}%`;
        } else if (fakeProbability < 0.45) {
            resultDisplay.innerHTML = '<i class="fas fa-check-circle"></i> Likely Human-Written (Genuine) Review';
            resultDisplay.className = 'genuine';
            confidenceScore.textContent = `Confidence: ${confidencePercentage}%`;
        } else {
            resultDisplay.innerHTML = '<i class="fas fa-question-circle"></i> Uncertain';
            resultDisplay.className = 'uncertain';
            confidenceScore.textContent = 'Not enough information to make a determination';
        }
        
        // Analysis details kanikkuka
        let details = '<strong>Analysis Details:</strong><br>';
        
        if (fakeMatches.length > 0) {
            details += '<br><strong><i class="fas fa-exclamation-triangle"></i> Potential computer-generated indicators:</strong><br>';
            details += fakeMatches.join('<br>');
        }
        
        if (genuineMatches.length > 0) {
            details += '<br><br><strong><i class="fas fa-check"></i> Potential human-written indicators:</strong><br>';
            details += genuineMatches.join('<br>');
        }
        
        if (fakeMatches.length === 0 && genuineMatches.length === 0) {
            details += '<br>No strong indicators found in this review.';
        }
        
        analysisDetails.innerHTML = details;
    }
    
    function updatePerformanceBars(data) {
        const performanceBars = document.getElementById('performanceBars');
        if (!performanceBars) return;
        
        performanceBars.innerHTML = '';
        
        for (const [algorithm, result] of Object.entries(data.individual_predictions)) {
            const algoName = algorithm.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            const confidence = Math.round(result.confidence * 100);
            
            const barHTML = `
                <div class="performance-bar">
                    <div class="performance-bar-label">
                        <span>${algoName}</span>
                        <span>${confidence}%</span>
                    </div>
                    <div class="performance-bar-track">
                        <div class="performance-bar-fill" style="width: ${confidence}%"></div>
                    </div>
                </div>
            `;
            
            performanceBars.innerHTML += barHTML;
        }
    }
});

// Theme toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            
            // Theme based icon change cheyyuka
            const icon = themeToggle.querySelector('i');
            if (document.body.classList.contains('dark-mode')) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
            
            // Preference localStoragil save cheyyuka
            const isDarkMode = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDarkMode);
        });
        
        // Saved theme preference check cheyyuka
        const savedDarkMode = localStorage.getItem('darkMode') === 'true';
        if (savedDarkMode) {
            document.body.classList.add('dark-mode');
            const icon = themeToggle.querySelector('i');
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }
    }
});