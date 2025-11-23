//FHE BORDER SCRIPT

//run when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log('FHE Border Script loading...');

  //UI elements for file upload, buttons, password fields, logs, canvas, etc.
  const borderFileInput = document.getElementById('borderFileInput');
  const borderDropzone = document.getElementById('borderDropzone');
  const borderEncryptPassword = document.getElementById('borderEncryptPassword');
  const borderEncryptBtn = document.getElementById('borderEncryptBtn');
  const borderEncryptedPreview = document.getElementById('borderEncryptedPreview');
  const borderSendBtn = document.getElementById('borderSendBtn');
  const borderLogBox = document.getElementById('borderLogBox');
  const borderDecryptPassword = document.getElementById('borderDecryptPassword');
  const borderDecryptBtn = document.getElementById('borderDecryptBtn');
  const borderOutputCanvas = document.getElementById('borderOutputCanvas');
  const borderRestartBtn = document.getElementById('borderRestartBtn');
  const borderSaveBtn = document.getElementById('borderSaveBtn');
  const TRUSTED_SERVER_URL = 'http://localhost:4000'; //URL of trusted backend server

  //runtime variables
  let selectedImage = null;
  let encryptedData = null;
  let encryptedResult = null;
  let currentPassword = null;
  let resultDataUrl = null;
  let isLocked = false;

  //log function to append message to logging box and scroll
  function borderLog(msg) {
    borderLogBox.textContent += `\n${msg}`;
    borderLogBox.scrollTop = borderLogBox.scrollHeight;
    console.log('[FHE-BORDER]', msg);
  }

  function sizeCanvasToCSS() {
    const r = borderOutputCanvas.getBoundingClientRect();
    borderOutputCanvas.width = Math.round(r.width);
    borderOutputCanvas.height = Math.round(r.height);
  }

  //called when restart button is clicked or at startup
  function clearAll() {
    selectedImage = null;
    encryptedData = null;
    encryptedResult = null;
    currentPassword = null;
    resultDataUrl = null;

    //clear canvas
    if (borderEncryptedPreview) {
      const ctx = borderEncryptedPreview.getContext('2d');
      ctx.clearRect(0, 0, borderEncryptedPreview.width, borderEncryptedPreview.height);
    }

    if (borderOutputCanvas) {
      sizeCanvasToCSS();
      const ctx = borderOutputCanvas.getContext('2d');
      ctx.clearRect(0, 0, borderOutputCanvas.width, borderOutputCanvas.height);
    }

    //reset input fields and UI
    borderEncryptPassword.value = '';
    borderDecryptPassword.value = '';

    borderLogBox.textContent = '';
    borderLog('[INFO] Waiting for image…');

    //make UI usable
    isLocked = false;
    borderEncryptBtn.disabled = false;
    borderEncryptPassword.disabled = false;
    borderFileInput.disabled = false;
    borderDropzone.classList.remove('locked');
    borderSendBtn.disabled = true;
    borderDecryptBtn.disabled = true;
    borderSaveBtn.disabled = true;

    //re-render drag/drop zone
    borderDropzone.innerHTML = `
      <button type="button" id="borderPickBtn" class="chip-btn">Insert Image</button>
      <input id="borderFileInput" type="file" accept="image/*" hidden />
      <div class="hint">Drag & drop image with borders here</div>
    `;

    document.getElementById('borderPickBtn').addEventListener('click', () => borderFileInput.click());
    setupDropzone();
  }

  //allows clicking on predefined template image to simulate uploaded image
  document.querySelectorAll('.template-img-encrypted').forEach(img => {
    img.addEventListener('click', async () => {
      if(isLocked) return;

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
    borderEncryptBtn.disabled = borderEncryptPassword.value.trim().length === 0;

    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      borderDropzone.innerHTML = '';
      borderDropzone.appendChild(img);
      img.style.maxHeight = '280px';
      img.style.maxWidth = '100%';
      img.style.objectFit = 'contain';
    };
    img.src = url;
  }

  //enable drag and drop functionality
  function setupDropzone() {
    borderFileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) handleNewFile(file);
    });

    borderDropzone.addEventListener('dragover', e => {
      if (isLocked) return;
      e.preventDefault();
      borderDropzone.classList.add('dragging');
    });

    borderDropzone.addEventListener('dragleave', () => {
      if (isLocked) return;
      borderDropzone.classList.remove('dragging')
    });

    borderDropzone.addEventListener('drop', e => {
      if (isLocked) return;
      e.preventDefault();
      borderDropzone.classList.remove('dragging');
      const file = e.dataTransfer.files[0];
      if (file) handleNewFile(file);
    });
  }

  //displays noise on canvas to visualize 'encryption'
  function displayEncryptedNoise(password) {
    const canvas = borderEncryptedPreview;
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
  borderEncryptPassword.addEventListener('input', () => {
    borderEncryptBtn.disabled = !(selectedImage && borderEncryptPassword.value.trim().length > 0);
  });

  //send to trusted server to encrypt image + password 'locally'
  borderEncryptBtn.addEventListener('click', async () => {
    if (!selectedImage) return borderLog('[ERR] No image selected.');
    const password = borderEncryptPassword.value.trim();
    currentPassword = password;

    const formData = new FormData();
    formData.append('password', password);
    formData.append('image', selectedImage);

    //lock UI
    isLocked = true;
    borderEncryptBtn.disabled = true;
    borderEncryptPassword.disabled = true;
    borderFileInput.disabled = true;
    borderDropzone.classList.add('locked');

    try {
      //send to trusted server
      const res = await fetch(`${TRUSTED_SERVER_URL}/api/fhe-border-encrypt`, {
        method: 'POST',
        body: formData
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Border encryption failed');
      encryptedData = json;
      borderSendBtn.disabled = false;
      displayEncryptedNoise(password);
    } catch (err) {
      borderLog(`[ERR] ${err.message}`);
    }
  });

  //send encrypted data to untrusted server for inference
  borderSendBtn.addEventListener('click', async () => {
    if (!encryptedData) return borderLog('[ERR] Nothing to send');
    borderSendBtn.disabled = true;

    const formData = new FormData();
    formData.append('fhe_data', JSON.stringify(encryptedData));

    //submit encrypted data
    const res = await fetch('/api/fhe-border/upload', { method: 'POST', body: formData });
    const { jobId } = await res.json();
    const es = new EventSource(`/api/fhe-border/stream/${jobId}`);

    es.addEventListener('log', ev => borderLog(JSON.parse(ev.data).message));
    es.addEventListener('result', ev => {
      const data = JSON.parse(ev.data);
      if (data.encrypted_output) {
        encryptedResult = {
          encrypted_output: data.encrypted_output,
          context: encryptedData.context
        };
        borderLog('[INFO] Received encrypted border result. Ready to decrypt.');
        borderDecryptBtn.disabled = false;
      } else if (data.error) {
        borderLog(`[ERR] Server error: ${data.error}`);
      } else {
        borderLog('[ERR] No encrypted border output received');
      }
      es.close();
    });
    es.addEventListener('error', () => {
      borderLog('[ERR] Stream error');
      borderSendBtn.disabled = false;
      es.close();
    });
  });

  //decrypt encrypted output using original password
  borderDecryptBtn.addEventListener('click', async () => {
    const password = borderDecryptPassword.value.trim();
    if (password !== currentPassword) return borderLog('[ERR] Password mismatch.');

    try {
      const res = await fetch(`${TRUSTED_SERVER_URL}/api/fhe-border-decrypt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operation: 'decrypt', password, encrypted_output: encryptedResult })
      });
      const result = await res.json();
      displayCleanedImage(result.cleaned_image);
      borderSaveBtn.disabled = false;
    } catch (err) {
      borderLog(`[ERR] ${err.message}`);
    }
  });

  //display output image
  function displayCleanedImage(cleanedImageData) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 32;
    tempCanvas.height = 32;
    const tempCtx = tempCanvas.getContext('2d');
    const imageData = tempCtx.createImageData(32, 32);
    for (let i = 0; i < 32; i++) {
      for (let j = 0; j < 32; j++) {
        const pixelIndex = (i * 32 + j) * 4;
        const value = cleanedImageData[i][j];
        imageData.data[pixelIndex] = value;
        imageData.data[pixelIndex + 1] = value;
        imageData.data[pixelIndex + 2] = value;
        imageData.data[pixelIndex + 3] = 255;
      }
    }
    tempCtx.putImageData(imageData, 0, 0);
    sizeCanvasToCSS();
    const ctx = borderOutputCanvas.getContext('2d');
    const cw = borderOutputCanvas.width, ch = borderOutputCanvas.height;
    const scale = Math.min(cw / 32, ch / 32);
    const w = 32 * scale, h = 32 * scale;
    const x = (cw - w) / 2, y = (ch - h) / 2;
    
    ctx.clearRect(0, 0, cw, ch);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, x, y, w, h);
  }

  //save final output image
  borderSaveBtn.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = borderOutputCanvas.toDataURL();
    a.download = 'border-image.png';
    a.click();
  });

  //restart UI
  borderRestartBtn.addEventListener('click', clearAll);

  //intialize
  sizeCanvasToCSS();
  window.addEventListener('resize', sizeCanvasToCSS);
  setupDropzone();
  clearAll();
});