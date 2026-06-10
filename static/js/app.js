document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const changeBtn = document.getElementById('changeBtn');
    const modelSelect = document.getElementById('modelSelect');
    const emptyState = document.getElementById('empty');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const errorMsg = document.getElementById('errorMsg');
    const top5Label = document.getElementById('top5Label');
    const topCandidates = document.getElementById('topCandidates');

    let lastFile = null;

    const show = (el) => { el.hidden = false; };
    const hide = (el) => { el.hidden = true; };

    uploadBox.addEventListener('click', () => imageInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });
    uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('drag-over'));
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleImageSelect(e.dataTransfer.files[0]);
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleImageSelect(e.target.files[0]);
    });

    document.querySelectorAll('.sample').forEach((btn) => {
        btn.addEventListener('click', () => {
            fetch(btn.dataset.src)
                .then((r) => r.blob())
                .then((blob) => handleImageSelect(new File([blob], 'sample.jpg', { type: blob.type || 'image/jpeg' })))
                .catch(() => showError('Could not load the sample image.'));
        });
    });

    // Re-run prediction with the same photo when the model changes.
    modelSelect.addEventListener('change', () => {
        if (lastFile) predictImage(lastFile);
    });

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.value = '';
        lastFile = null;
        hide(imagePreview);
        show(uploadBox);
        hide(resultsDiv);
        hide(errorDiv);
        show(emptyState);
    });

    function handleImageSelect(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please choose an image file.');
            return;
        }
        lastFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            hide(uploadBox);
            show(imagePreview);
            hide(resultsDiv);
            hide(errorDiv);
            setTimeout(() => predictImage(file), 250);
        };
        reader.readAsDataURL(file);
    }

    function predictImage(file) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('model', modelSelect.value);

        hide(emptyState);
        hide(resultsDiv);
        hide(errorDiv);
        show(loadingDiv);

        fetch('/predict', { method: 'POST', body: formData })
            .then((r) => r.json())
            .then((data) => {
                hide(loadingDiv);
                if (data.error) showError(data.error);
                else displayResults(data);
            })
            .catch((err) => {
                hide(loadingDiv);
                showError('Network error: ' + err.message);
            });
    }

    function displayResults(data) {
        const mainPrediction = document.getElementById('mainPrediction');
        const feedbackBox = document.getElementById('feedback');

        const acc = (data.model_accuracy * 100).toFixed(1);
        let html =
            '<div class="model-meta">' +
                '<span class="model-name">' + data.model_label + '</span>' +
                '<span class="model-acc">test accuracy ' + acc + '%</span>' +
            '</div>' +
            '<div class="pred-label">' + (data.confidence === null ? 'Prediction' : 'Most likely') + '</div>' +
            '<div class="pred-name">' + data.prediction + '</div>';

        if (data.confidence !== null && data.confidence !== undefined) {
            const conf = (data.confidence * 100).toFixed(1);
            html +=
                '<div class="pred-label">Confidence</div>' +
                '<div class="pred-conf-row">' +
                    '<div class="conf-track"><div class="conf-fill" style="width:' + conf + '%"></div></div>' +
                    '<div class="conf-num">' + conf + '%</div>' +
                '</div>';
        }
        if (data.note) html += '<div class="pred-note">' + data.note + '</div>';
        mainPrediction.innerHTML = html;

        const top = data.top_k || [];
        if (top.length) {
            show(top5Label);
            show(topCandidates);
            topCandidates.innerHTML = top.map((item, i) => {
                return '<div class="candidate">' +
                    '<div class="cand-rank">' + (i + 1) + '</div>' +
                    '<div class="cand-name">' + item.label + '</div>' +
                    '<div class="cand-bar"><span style="width:' + item.bar + '%"></span></div>' +
                    '<div class="cand-pct">' + (item.pct ? item.pct : '') + '</div>' +
                '</div>';
            }).join('');
        } else {
            hide(top5Label);
            hide(topCandidates);
            topCandidates.innerHTML = '';
        }

        if (data.feedback) {
            const fb = data.feedback;
            const cls = fb.level === 'low_confidence' ? 'low_confidence' : 'confident';
            feedbackBox.className = 'feedback-box ' + cls;
            feedbackBox.innerHTML = '<div>' + fb.message + '</div>' +
                (fb.suggestion ? '<div class="feedback-suggestion">' + fb.suggestion + '</div>' : '');
        }

        show(resultsDiv);
        hide(errorDiv);
    }

    function showError(message) {
        errorMsg.textContent = message;
        hide(emptyState);
        show(errorDiv);
        hide(resultsDiv);
    }
});
