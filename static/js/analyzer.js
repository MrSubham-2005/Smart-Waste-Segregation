/**
 * EcoSort AI - Waste Analyzer Client Module (Phase 4)
 * Handles image selection, drag-and-drop, preview, client validation,
 * API submission to /api/predict, and honest result rendering.
 */

document.addEventListener('DOMContentLoaded', function () {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('waste-image-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const previewName = document.getElementById('preview-name');
    const previewSize = document.getElementById('preview-size');
    const previewDimensions = document.getElementById('preview-dimensions');
    const resetBtn = document.getElementById('reset-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingBox = document.getElementById('loading-box');
    const resultContainer = document.getElementById('result-container');

    let currentFile = null;
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB
    const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'];

    if (!dropzone || !fileInput) return;

    // -------------------------------------------------------------
    // Helper: Format bytes to human readable format
    // -------------------------------------------------------------
    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // -------------------------------------------------------------
    // Helper: Client-side validation
    // -------------------------------------------------------------
    function validateFile(file) {
        if (!file) return { valid: false, error: 'No file selected.' };

        const ext = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            return {
                valid: false,
                error: `Unsupported file format (.${ext}). Please upload JPG, PNG, or WEBP.`
            };
        }

        if (file.size > MAX_FILE_SIZE) {
            return {
                valid: false,
                error: `File is too large (${formatBytes(file.size)}). Maximum upload size is 5 MB.`
            };
        }

        return { valid: true };
    }

    // -------------------------------------------------------------
    // Image Handling & Preview Generation
    // -------------------------------------------------------------
    function handleSelectedFile(file) {
        const validation = validateFile(file);
        if (!validation.valid) {
            alert(validation.error);
            fileInput.value = '';
            return;
        }

        currentFile = file;
        previewName.textContent = file.name;
        previewSize.textContent = formatBytes(file.size);
        previewDimensions.textContent = 'Calculating dimensions...';

        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;

            // Extract image natural dimensions
            const tempImg = new Image();
            tempImg.onload = function () {
                previewDimensions.textContent = `${this.naturalWidth} × ${this.naturalHeight} px`;
            };
            tempImg.src = e.target.result;

            // Toggle visibility
            dropzone.style.display = 'none';
            previewContainer.style.display = 'block';
            resultContainer.style.display = 'none';
            resultContainer.innerHTML = '';
        };
        reader.readAsDataURL(file);
    }

    // -------------------------------------------------------------
    // Event Listeners: Drag & Drop
    // -------------------------------------------------------------
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('drag-active'));
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('drag-active'));
    });

    dropzone.addEventListener('drop', function (e) {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleSelectedFile(dt.files[0]);
        }
    });

    // -------------------------------------------------------------
    // Event Listeners: Browse & Input
    // -------------------------------------------------------------
    browseBtn.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        if (this.files && this.files.length > 0) {
            handleSelectedFile(this.files[0]);
        }
    });

    // -------------------------------------------------------------
    // Reset / Remove Image
    // -------------------------------------------------------------
    function resetUploader() {
        currentFile = null;
        fileInput.value = '';
        previewImg.src = '';
        previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
        resultContainer.style.display = 'none';
        resultContainer.innerHTML = '';
        loadingBox.style.display = 'none';
        analyzeBtn.disabled = false;
    }

    resetBtn.addEventListener('click', resetUploader);

    // -------------------------------------------------------------
    // API Submission: POST /api/predict
    // -------------------------------------------------------------
    analyzeBtn.addEventListener('click', function () {
        if (!currentFile) {
            alert('Please select an image before submitting.');
            return;
        }

        // Check if user is authenticated before triggering AI analysis
        const isLoggedIn = window.ECOSORT_CONFIG && window.ECOSORT_CONFIG.isLoggedIn;
        if (!isLoggedIn) {
            const loginUrl = (window.ECOSORT_CONFIG && window.ECOSORT_CONFIG.loginUrl) || ('/login?next=' + encodeURIComponent('/analyze'));
            window.location.href = loginUrl;
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        resetBtn.disabled = true;
        loadingBox.style.display = 'block';
        resultContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('image', currentFile);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
            .then(async response => {
                const data = await response.json();
                loadingBox.style.display = 'none';
                analyzeBtn.disabled = false;
                resetBtn.disabled = false;

                if (response.status === 401 || data.status === 'unauthorized') {
                    const loginUrl = (window.ECOSORT_CONFIG && window.ECOSORT_CONFIG.loginUrl) || ('/login?next=' + encodeURIComponent('/analyze'));
                    window.location.href = loginUrl;
                    return;
                }

                renderAnalysisResult(data);
            })
            .catch(error => {
                loadingBox.style.display = 'none';
                analyzeBtn.disabled = false;
                resetBtn.disabled = false;
                renderError('An unexpected network error occurred while communicating with the analysis server.');
                console.error('Prediction API Error:', error);
            });
    });

    // -------------------------------------------------------------
    // Result Rendering Handlers
    // -------------------------------------------------------------
    function renderAnalysisResult(data) {
        resultContainer.style.display = 'block';

        // 1. Untrained Model State
        if (data.status === 'model_not_trained' || !data.is_trained) {
            resultContainer.innerHTML = `
                <div class="untrained-card">
                    <div class="untrained-icon">⚙️</div>
                    <h2 style="color: #b45309; margin-bottom: 8px;">AI Model Not Trained Yet</h2>
                    <p style="font-size: 1.05rem; color: #78350f; max-width: 600px; margin: 0 auto 16px auto;">
                        The waste classification model is currently being prepared. Please train the model before using AI classification.
                    </p>
                    <div style="background: #ffffff; border: 1px solid #fde68a; border-radius: 8px; padding: 14px; max-width: 580px; margin: 0 auto 20px auto; font-size: 0.9rem; color: #92400e; text-align: left;">
                        <strong>Academic Transparency Notice:</strong><br>
                        EcoSort AI does not generate fabricated predictions or artificial confidence scores. Real classification requires trained MobileNetV2 model weights.
                    </div>
                    <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                        <button type="button" class="btn btn-primary" id="btn-analyze-another">Analyze Another Image</button>
                        <a href="/dashboard" class="btn btn-secondary">Back to Dashboard</a>
                    </div>
                </div>
            `;
            document.getElementById('btn-analyze-another').addEventListener('click', resetUploader);
            resultContainer.scrollIntoView({ behavior: 'smooth' });
            return;
        }

        // 2. Low Confidence State
        if (data.status === 'low_confidence') {
            resultContainer.innerHTML = `
                <div class="low-conf-card">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                        <span style="font-size: 2rem;">⚠️</span>
                        <div>
                            <h2 style="color: #c2410c; margin: 0;">Unable to Confidently Identify Waste</h2>
                            <p style="color: #9a3412; margin: 0; font-size: 0.95rem;">Confidence score (${data.confidence_percentage}%) is below the required 50% threshold.</p>
                        </div>
                    </div>
                    <div style="background: #ffffff; border-radius: 8px; padding: 16px; border: 1px solid #fed7aa; margin-bottom: 18px;">
                        <p style="font-weight: 600; margin-bottom: 6px; color: #1f2937;">To help the AI classify accurately, please:</p>
                        <ul style="margin-left: 20px; color: #4b5563; font-size: 0.9rem; line-height: 1.6;">
                            <li>Upload a clearer, well-lit photo.</li>
                            <li>Ensure the waste item is centered and in focus.</li>
                            <li>Avoid cluttered backgrounds with multiple mixed objects.</li>
                            <li>Capture the item against a clean, plain background.</li>
                        </ul>
                    </div>
                    <div style="display: flex; gap: 12px;">
                        <button type="button" class="btn btn-primary" id="btn-retry-low">Try Another Image</button>
                        <a href="/awareness" class="btn btn-secondary">Review Segregation Rules</a>
                    </div>
                </div>
            `;
            document.getElementById('btn-retry-low').addEventListener('click', resetUploader);
            resultContainer.scrollIntoView({ behavior: 'smooth' });
            return;
        }

        // 3. Genuine Successful Prediction (Only displayed from real backend output)
        if (data.success) {
            resultContainer.innerHTML = `
                <div class="result-card-inner">
                    <div class="result-grid">
                        <div class="result-img-box">
                            <img src="${data.image_url || previewImg.src}" alt="Analyzed Waste">
                            <div style="padding: 10px; font-size: 0.8rem; color: #6b7280;">Input Waste Item</div>
                        </div>

                        <div>
                            <span class="category-tag ${data.badge_class}">
                                ${data.category_name}
                            </span>
                            <h2 style="margin: 0 0 6px 0; font-size: 1.6rem;">${data.category_name}</h2>
                            
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; margin-top: 10px;">
                                <span>Classification Confidence</span>
                                <span style="color: var(--primary);">${data.confidence_percentage}%</span>
                            </div>
                            <div class="confidence-meter">
                                <div class="confidence-fill" style="width: ${data.confidence_percentage}%;"></div>
                            </div>

                            <div class="guidance-box">
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                                    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 50%; background: ${data.bin_color_code};"></span>
                                    <strong>Recommended Disposal Bin:</strong> ${data.bin_name}
                                </div>
                                <p style="margin: 0; font-size: 0.92rem; color: #374151;">
                                    ${data.disposal_instructions}
                                </p>
                            </div>

                            <div class="tip-box">
                                <strong>💡 Environmental Impact:</strong>
                                <p style="margin: 4px 0 0 0;">${data.environmental_impact || data.sdg_impact}</p>
                            </div>

                            <div style="display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap;">
                                <button type="button" class="btn btn-primary" id="btn-analyze-again">Analyze Another Image</button>
                                <a href="/history" class="btn btn-secondary">View History</a>
                                <a href="/dashboard" class="btn btn-secondary">Dashboard</a>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('btn-analyze-again').addEventListener('click', resetUploader);
            resultContainer.scrollIntoView({ behavior: 'smooth' });
            return;
        }

        // 4. Fallback for validation or server error
        renderError(data.message || 'Unable to complete classification request.');
    }

    function renderError(message) {
        resultContainer.style.display = 'block';
        resultContainer.innerHTML = `
            <div class="alert alert-danger" style="display: block;">
                <strong>Error:</strong> ${message}
                <div style="margin-top: 12px;">
                    <button type="button" class="btn btn-secondary" id="btn-error-reset">Try Again</button>
                </div>
            </div>
        `;
        document.getElementById('btn-error-reset').addEventListener('click', resetUploader);
    }
});
