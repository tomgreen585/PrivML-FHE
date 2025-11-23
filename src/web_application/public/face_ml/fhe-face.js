//FHE FACE SCRIPT

//run when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log('FHE Face Script loading...');

  //UI elements for file upload, buttons, password fields, logs, canvas, etc.
  const faceFileInput = document.getElementById('faceFileInput');
  const faceDropzone = document.getElementById('faceDropzone');
  const faceEncryptPassword = document.getElementById('faceEncryptPassword');
  const faceEncryptBtn = document.getElementById('faceEncryptBtn');
  const faceEncryptedPreview = document.getElementById('faceEncryptedPreview');
  const faceSendBtn = document.getElementById('faceSendBtn');
  const faceLogBox = document.getElementById('faceLogBox');
  const faceDecryptPassword = document.getElementById('faceDecryptPassword');
  const faceDecryptBtn = document.getElementById('faceDecryptBtn');
  const faceOutputCanvas = document.getElementById('faceOutputCanvas');
  const faceRestartBtn = document.getElementById('faceRestartBtn');
  const faceSaveBtn = document.getElementById('faceSaveBtn');
  const TRUSTED_SERVER_URL = 'http://localhost:4000'; //URL of trusted backend server

  //runtime variables
  let selectedImage = null;
  let encryptedData = null;
  let encryptedResult = null;
  let currentPassword = null;
  let resultDataUrl = null;
  let isLocked = false;

  //log function to append message to logging box and scroll
  function faceLog(msg) {
    faceLogBox.textContent += `\n${msg}`;
    faceLogBox.scrollTop = faceLogBox.scrollHeight;
    console.log('[FHE-FACE]', msg);
  }

  //called when restart button is clicked or at startup
  function clearAll() {
    selectedImage = null;
    encryptedData = null;
    encryptedResult = null;
    currentPassword = null;
    resultDataUrl = null;

    //clear canvas
    if (faceEncryptedPreview) {
      const ctx = faceEncryptedPreview.getContext('2d');
      ctx.clearRect(0, 0, faceEncryptedPreview.width, faceEncryptedPreview.height);
    }

    if (faceOutputCanvas) {
      const ctx = faceOutputCanvas.getContext('2d');
      ctx.clearRect(0, 0, faceOutputCanvas.width, faceOutputCanvas.height);
    }

    //reset input fields and UI
    faceEncryptPassword.value = '';
    faceDecryptPassword.value = '';

    faceLogBox.textContent = '';
    faceLog('[INFO] Waiting for image…');

    //make UI usable
    isLocked = false;
    faceEncryptBtn.disabled = false;
    faceEncryptPassword.disabled = false;
    faceFileInput.disabled = false;
    faceDropzone.classList.remove('locked');
    faceSendBtn.disabled = true;
    faceDecryptBtn.disabled = true;
    faceSaveBtn.disabled = true;

    //re-render drag/drop zone
    faceDropzone.innerHTML = `
      <button type="button" id="facePickBtn" class="chip-btn">Insert Image</button>
      <input id="faceFileInput" type="file" accept="image/*" hidden />
      <div class="hint">Drag & drop image here</div>
    `;
    document.getElementById('facePickBtn').addEventListener('click', () => faceFileInput.click());
    setupDropzone();
  }

  //allows clicking on predefined template image to simulate uploaded image
  document.querySelectorAll('.template-img-encrypted').forEach(img => {
    img.addEventListener('click', async () => {
      if (isLocked) return;

      const imagePath = `/face_ml/template_images/${img.dataset.image}`;
      const response = await fetch(imagePath);
      const blob = await response.blob();
      const file = new File([blob], img.dataset.image, { type: blob.type });
      handleNewFile(file);
    });
  });

  //handle user file if uploaded OR template
  function handleNewFile(file) {
    selectedImage = file;
    faceEncryptBtn.disabled = faceEncryptPassword.value.trim().length === 0;

    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      faceDropzone.innerHTML = '';
      faceDropzone.appendChild(img);
      img.style.maxHeight = '280px';
      img.style.maxWidth = '100%';
      img.style.objectFit = 'contain';
    };
    img.src = url;
  }

  //enable drag and drop functionality
  function setupDropzone() {
    faceFileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file && !isLocked) handleNewFile(file);
    });

    faceDropzone.addEventListener('dragover', e => {
      if (isLocked) return;
      e.preventDefault();
      faceDropzone.classList.add('dragging');
    });

    faceDropzone.addEventListener('dragleave', () => {
      if (isLocked) return;
      faceDropzone.classList.remove('dragging')
    });

    faceDropzone.addEventListener('drop', e => {
      if (isLocked) return;
      e.preventDefault();
      faceDropzone.classList.remove('dragging');
      const file = e.dataTransfer.files[0];
      if (file) handleNewFile(file);
    });
  }

  //displays noise on canvas to visualize 'encryption'
  function displayEncryptedNoise(password) {
    const canvas = faceEncryptedPreview;
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 256;
    const imageData = ctx.createImageData(256, 256);
    let seed = 0;
    for (let i = 0; i < password.length; i++) seed += password.charCodeAt(i);

    function random() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    }

    for (let i = 0; i < 256 * 256; i++) {
      const pixelIndex = i * 4;
      const value1 = Math.floor(random() * 256);
      const value2 = Math.floor(random() * 256);
      const value3 = Math.floor(random() * 256);
      imageData.data[pixelIndex] = value1;
      imageData.data[pixelIndex + 1] = value2;
      imageData.data[pixelIndex + 2] = value3;
      imageData.data[pixelIndex + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
  }

  //enable encrypt button when password is valid
  faceEncryptPassword.addEventListener('input', () => {
    faceEncryptBtn.disabled = !(selectedImage && faceEncryptPassword.value.trim().length > 0);
  });

  //send to trusted server to encrypt image + password 'locally'
  faceEncryptBtn.addEventListener('click', async () => {
    if (!selectedImage) return faceLog('[ERR] No image selected.');
    const password = faceEncryptPassword.value.trim();
    currentPassword = password;

    const formData = new FormData();
    formData.append('password', password);
    formData.append('image', selectedImage);
    
    //lock UI
    isLocked = true;
    faceEncryptBtn.disabled = true;
    faceEncryptPassword.disabled = true;
    faceFileInput.disabled = true;
    faceDropzone.classList.add('locked');

    try {
      //send to trusted server
      const res = await fetch(`${TRUSTED_SERVER_URL}/api/fhe-face-encrypt`, {
        method: 'POST',
        body: formData
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Face encryption failed');
      encryptedData = json;
      faceSendBtn.disabled = false;
      displayEncryptedNoise(password);
    } catch (err) {
      faceLog(`[ERR] ${err.message}`);
    }
  });

  //send encrypted data to untrusted server for inference
  faceSendBtn.addEventListener('click', async () => {
    if (!encryptedData) return faceLog('[ERR] Nothing to send');
    faceSendBtn.disabled = true;

    const formData = new FormData();
    formData.append('fhe_data', JSON.stringify(encryptedData));

    //submit encrypted data
    const res = await fetch('/api/fhe-face/upload', { method: 'POST', body: formData });
    const { jobId } = await res.json();
    const es = new EventSource(`/api/fhe-face/stream/${jobId}`);

    es.addEventListener('log', ev => faceLog(JSON.parse(ev.data).message));
    es.addEventListener('result', ev => {
      const data = JSON.parse(ev.data);
      if (data.encrypted_output) {
        encryptedResult = {
          encrypted_output: data.encrypted_output,
          context: encryptedData.context
        };
        faceLog('[INFO] Received encrypted face result. Ready to decrypt.');
        faceDecryptBtn.disabled = false;
      } else if (data.error) {
        faceLog(`[ERR] Server error: ${data.error}`);
      } else {
        faceLog('[ERR] No encrypted face output received');
      }
      es.close();
    });
    es.addEventListener('error', () => {
      faceLog('[ERR] Stream error');
      faceSendBtn.disabled = false;
      es.close();
    });
  });

  //decrypt encrypted output using original password
  faceDecryptBtn.addEventListener('click', async () => {
    const password = faceDecryptPassword.value.trim();
    if (password !== currentPassword) return faceLog('[ERR] Password mismatch.');

    try {
        const res = await fetch(`${TRUSTED_SERVER_URL}/api/fhe-face-decrypt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operation: 'decrypt', password, encrypted_output: encryptedResult })
        });

        const result = await res.json();
        drawBoundingBox(result.prediction);
        faceSaveBtn.disabled = false;
    } catch (err) {
        faceLog(`[ERR] ${err.message}`);
    }
  });

  //draw bounding box prediction on original image
  function drawBoundingBox(pred) {
    const ctx = faceOutputCanvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      const cw = faceOutputCanvas.width;
      const ch = faceOutputCanvas.height;

      const bounds = faceOutputCanvas.getBoundingClientRect();
      faceOutputCanvas.width = Math.round(bounds.width);
      faceOutputCanvas.height = Math.round(bounds.height);

      const scale = Math.min(faceOutputCanvas.width / img.width, faceOutputCanvas.height / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const x = (faceOutputCanvas.width - w) / 2;
      const y = (faceOutputCanvas.height - h) / 2;

      ctx.clearRect(0, 0, faceOutputCanvas.width, faceOutputCanvas.height);
      ctx.drawImage(img, x, y, w, h);

      const [xc, yc, boxW, boxH] = pred.map(Number);
      const absX = x + (xc - boxW / 2) * w;
      const absY = y + (yc - boxH / 2) * h;
      const absW = boxW * w;
      const absH = boxH * h;

      ctx.strokeStyle = 'green';
      ctx.lineWidth = 3;
      ctx.strokeRect(absX, absY, absW, absH);
    };
    img.src = URL.createObjectURL(selectedImage);
  }

  //save final output image
  faceSaveBtn.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = faceOutputCanvas.toDataURL();
    a.download = 'predicted-face.png';
    a.click();
  });

  //restart UI
  faceRestartBtn.addEventListener('click', clearAll);
  //initialise
  setupDropzone();
  clearAll();
});
