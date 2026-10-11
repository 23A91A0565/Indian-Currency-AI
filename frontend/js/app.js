/**
 * app.js — Client-Side Logic for Indian Currency Recognition
 * ==========================================================
 * Manages mobile camera capture, image uploads, client-side
 * optimization, API requests, and dynamic result rendering.
 */

// DOM Elements
const viewport = document.getElementById('viewport');
const previewImg = document.getElementById('previewImg');
const placeholder = document.getElementById('placeholder');
const reticle = document.getElementById('reticle');
const scanline = document.getElementById('scanline');
const cameraInput = document.getElementById('cameraInput');
const fileInput = document.getElementById('fileInput');
const loadingCard = document.getElementById('loadingCard');
const resultCard = document.getElementById('resultCard');
const errorToast = document.getElementById('errorToast');

// Track active preview object URL for memory cleanup
let activePreviewUrl = null;

// Event Listeners for file selection
cameraInput.addEventListener('change', handleFileSelected);
fileInput.addEventListener('change', handleFileSelected);

// Viewport click triggers file chooser if no image yet
viewport.addEventListener('click', (e) => {
  if (e.target !== previewImg) {
    fileInput.click();
  }
});

// Drag and Drop Support
viewport.addEventListener('dragover', (e) => {
  e.preventDefault();
  viewport.classList.add('dragover');
});

viewport.addEventListener('dragleave', () => {
  viewport.classList.remove('dragover');
});

viewport.addEventListener('drop', (e) => {
  e.preventDefault();
  viewport.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    processImageFile(e.dataTransfer.files[0]);
  }
});

/**
 * Triggered when a user selects or snaps a photo from the camera or file picker.
 */
function handleFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  // CRITICAL FOR MOBILE: Reset the input value so selecting the same file
  // or capturing multiple camera shots consecutively triggers 'change' every time.
  event.target.value = '';
  hideError();

  if (file) {
    processImageFile(file);
  }
}

/**
 * Optimizes large smartphone camera photos before uploading.
 * High-resolution phone cameras produce 12MP to 48MP photos (5MB - 20MB).
 * Uploading uncompressed files over local Wi-Fi causes timeouts and mobile crashes.
 * Downscales to max 1280px and converts to JPEG (~150KB), uploading in < 100ms.
 */
async function optimizeImageForUpload(file) {
  // If already small (< 400KB) JPEG/PNG, upload directly
  if (file.size < 400 * 1024 && (file.type === 'image/jpeg' || file.type === 'image/png')) {
    return file;
  }

  return new Promise((resolve) => {
    const img = new Image();
    const tempUrl = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(tempUrl);
      let width = img.naturalWidth || img.width;
      let height = img.naturalHeight || img.height;
      const maxDim = 1280;

      if (width > maxDim || height > maxDim) {
        if (width > height) {
          height = Math.round((height * maxDim) / width);
          width = maxDim;
        } else {
          width = Math.round((width * maxDim) / height);
          height = maxDim;
        }
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob((blob) => {
        if (blob) {
          resolve(new File([blob], 'banknote.jpg', { type: 'image/jpeg' }));
        } else {
          resolve(file);
        }
      }, 'image/jpeg', 0.88);
    };

    img.onerror = () => {
      URL.revokeObjectURL(tempUrl);
      resolve(file); // Fallback to raw file if canvas fails
    };

    img.src = tempUrl;
  });
}

/**
 * Displays image preview and submits it to the /api/predict endpoint.
 */
async function processImageFile(file) {
  hideError();

  // Instant zero-memory preview using Object URL (prevents mobile Base64 OOM crashes)
  if (activePreviewUrl) {
    URL.revokeObjectURL(activePreviewUrl);
  }
  activePreviewUrl = URL.createObjectURL(file);
  previewImg.src = activePreviewUrl;
  previewImg.style.display = 'block';
  placeholder.style.display = 'none';
  viewport.classList.add('has-image');

  // Activate scanning animation
  setScanningState(true);

  try {
    // Downscale and compress on client side for instantaneous mobile upload
    const uploadPayload = await optimizeImageForUpload(file);

    const formData = new FormData();
    formData.append('file', uploadPayload);

    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned status ${response.status}`);
    }

    const data = await response.json();
    renderClassificationResult(data);
  } catch (error) {
    console.error('Recognition error:', error);
    showError(`Recognition failed: ${error.message}`);
  } finally {
    setScanningState(false);
  }
}

/**
 * Loads a demo sample image from the server for quick verification.
 */
async function loadSample(denomination) {
  hideError();
  try {
    setScanningState(true);
    const response = await fetch(`/api/sample/${denomination}`);
    if (!response.ok) {
      throw new Error(`Failed to load sample: ${response.statusText}`);
    }
    const blob = await response.blob();
    const file = new File([blob], `sample_${denomination}.jpg`, { type: 'image/jpeg' });
    processImageFile(file);
  } catch (error) {
    showError(`Could not load sample image: ${error.message}`);
    setScanningState(false);
  }
}

/**
 * Toggles the scanning laser, reticle, and loading indicator.
 */
function setScanningState(isScanning) {
  if (isScanning) {
    loadingCard.style.display = 'flex';
    resultCard.style.display = 'none';
    reticle.style.display = 'block';
    scanline.style.display = 'block';
  } else {
    loadingCard.style.display = 'none';
    reticle.style.display = 'none';
    scanline.style.display = 'none';
  }
}

/**
 * Shows error banner notification.
 */
function showError(message) {
  if (errorToast) {
    errorToast.textContent = message;
    errorToast.style.display = 'block';
  }
}

/**
 * Hides error banner notification.
 */
function hideError() {
  if (errorToast) {
    errorToast.style.display = 'none';
  }
}

/**
 * Populates the UI with prediction metrics, banknote specs, and confidence bars.
 */
function renderClassificationResult(data) {
  resultCard.style.display = 'block';

  // 1. Denomination Header & Badge
  const symbolElem = document.getElementById('resSymbol');
  symbolElem.textContent = data.denomination;
  symbolElem.className = 'denom-symbol color-' + data.raw_class;

  document.getElementById('resName').textContent = data.name;
  document.getElementById('resConfidence').textContent = data.confidence.toFixed(1) + '%';

  // 2. Banknote Specifications
  const details = data.details || {};
  document.getElementById('resColor').textContent = details.color || 'N/A';
  document.getElementById('resDimensions').textContent = details.dimensions || 'N/A';
  document.getElementById('resMotif').textContent = details.motif || 'N/A';
  document.getElementById('resBleed').textContent = details.bleed_lines || 'N/A';

  // 3. Probability Distribution Bars
  const probContainer = document.getElementById('probBars');
  probContainer.innerHTML = '';

  if (data.probabilities) {
    // Sort denominations numerically (10, 20, 50, 100, 200, 500)
    const sortedClasses = Object.keys(data.probabilities).sort((a, b) => parseInt(a) - parseInt(b));

    sortedClasses.forEach((cls) => {
      const percentage = data.probabilities[cls];
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-name">₹${cls}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${percentage}%"></div>
        </div>
        <span class="prob-pct">${percentage}%</span>
      `;
      probContainer.appendChild(row);
    });
  }

  // Smooth scroll to the result card for comfortable viewing
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
