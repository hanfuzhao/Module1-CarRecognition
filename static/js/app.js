document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const changeBtn = document.getElementById('changeBtn');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const errorMsg = document.getElementById('errorMsg');

    // Upload box click
    uploadBox.addEventListener('click', () => imageInput.click());

    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageSelect(files[0]);
        }
    });

    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageSelect(e.target.files[0]);
        }
    });

    // Change image button
    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.value = '';
        imagePreview.style.display = 'none';
        uploadBox.style.display = 'block';
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
    });

    function handleImageSelect(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please select a valid image file');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadBox.style.display = 'none';
            imagePreview.style.display = 'block';
            resultsDiv.style.display = 'none';
            errorDiv.style.display = 'none';

            // Auto-submit for prediction
            setTimeout(() => predictImage(file), 300);
        };
        reader.readAsDataURL(file);
    }

    function predictImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                loadingDiv.style.display = 'none';

                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                loadingDiv.style.display = 'none';
                showError('Network error: ' + error.message);
            });
    }

    function displayResults(data) {
        const mainPrediction = document.getElementById('mainPrediction');
        const topCandidates = document.getElementById('topCandidates');
        const feedbackBox = document.getElementById('feedback');

        // Main prediction
        if (data.prediction) {
            const confidence = (data.confidence * 100).toFixed(1);

            mainPrediction.innerHTML = `
                <div class="prediction-label">Model Prediction</div>
                <div class="prediction-item">${data.prediction}</div>
                <div class="prediction-label">Confidence Score</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%">
                        ${confidence}%
                    </div>
                </div>
                <div class="prediction-label">Model: ResNet50 features + MLP head</div>
            `;

            // Top 5 candidates
            topCandidates.innerHTML = '';
            (data.top_k || []).forEach((item, index) => {
                const pct = (item.confidence * 100).toFixed(1);
                topCandidates.innerHTML += `
                    <div class="candidate-item">
                        <div class="candidate-name">#${index + 1}. ${item.label}</div>
                        <div class="candidate-score">${pct}%</div>
                    </div>
                `;
            });
        }

        // Feedback
        if (data.feedback) {
            const fb = data.feedback;
            const feedbackClass = fb.level === 'low_confidence' ? 'low_confidence' : 'confident';

            feedbackBox.className = `feedback-box ${feedbackClass}`;
            feedbackBox.innerHTML = `
                <p>${fb.message}</p>
                ${fb.suggestion ? `<div class="feedback-suggestion">${fb.suggestion}</div>` : ''}
            `;
        }

        resultsDiv.style.display = 'block';
        errorDiv.style.display = 'none';
    }

    function showError(message) {
        errorMsg.textContent = message;
        errorDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
    }
});
