document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const changeBtn = document.getElementById('changeBtn');
    const emptyState = document.getElementById('empty');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const errorMsg = document.getElementById('errorMsg');

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

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.value = '';
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
        const topCandidates = document.getElementById('topCandidates');
        const feedbackBox = document.getElementById('feedback');

        if (data.prediction) {
            const conf = (data.confidence * 100).toFixed(1);
            mainPrediction.innerHTML =
                '<div class="pred-label">Most likely</div>' +
                '<div class="pred-name">' + data.prediction + '</div>' +
                '<div class="pred-label">Confidence</div>' +
                '<div class="pred-conf-row">' +
                    '<div class="conf-track"><div class="conf-fill" style="width:' + conf + '%"></div></div>' +
                    '<div class="conf-num">' + conf + '%</div>' +
                '</div>';

            const top = data.top_k || [];
            const peak = top.length ? top[0].confidence : 1;
            topCandidates.innerHTML = top.map((item, i) => {
                const pct = (item.confidence * 100).toFixed(1);
                const rel = peak > 0 ? Math.max(3, (item.confidence / peak) * 100) : 0;
                return '<div class="candidate">' +
                    '<div class="cand-rank">' + (i + 1) + '</div>' +
                    '<div class="cand-name">' + item.label + '</div>' +
                    '<div class="cand-bar"><span style="width:' + rel + '%"></span></div>' +
                    '<div class="cand-pct">' + pct + '%</div>' +
                '</div>';
            }).join('');
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
