//FHE MNIST SCRIPT

//run once DOM loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log('FHE MNIST script loading...');
  
  //DOM element references
  const mnistDrawCanvas = document.getElementById('mnistDrawCanvas');
  const mnistClearBtn = document.getElementById('mnistClearBtn');
  const mnistEncryptPassword = document.getElementById('mnistEncryptPassword');
  const mnistEncryptBtn = document.getElementById('mnistEncryptBtn');
  const mnistEncryptedPreview = document.getElementById('mnistEncryptedPreview');
  const mnistSendBtn = document.getElementById('mnistSendBtn');
  const mnistLogBox = document.getElementById('mnistLogBox');
  const mnistDecryptPassword = document.getElementById('mnistDecryptPassword');
  const mnistDecryptBtn = document.getElementById('mnistDecryptBtn');
  const mnistOutputCanvas = document.getElementById('mnistOutputCanvas');
  const mnistRestartBtn = document.getElementById('mnistRestartBtn');
  const TRUSTED_SERVER_URL = 'http://localhost:4000';

  //runtime state variables
  let encryptedData = null;
  let encryptedResult = null;
  let currentPassword = null;
  let hasDrawing = false;
  let fheDrawing = false;
  let fheLastX = 0; 
  let fheLastY = 0;
  let isLocked = false;

  //needs to be removed from pipeline (NOT USED)
  const kernelShape = [1, 1]; 
  const stride = 1;

  //log to textbox and console
  function mnistLog(msg) {
    mnistLogBox.textContent += `\n${msg}`;
    mnistLogBox.scrollTop = mnistLogBox.scrollHeight;
    console.log('[FHE-MNIST]', msg);
  }

  //setup drawing canvas listeners
  function initFheDrawing() {
    console.log('Initializing FHE drawing canvas...');
    const ctx = mnistDrawCanvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, mnistDrawCanvas.width, mnistDrawCanvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = '#fff';

    function getPos(e) {
      const rect = mnistDrawCanvas.getBoundingClientRect();
      if (e.touches && e.touches[0]) {
        const t = e.touches[0];
        return { x: t.clientX - rect.left, y: t.clientY - rect.top };
      } else {
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
      }
    }

    function start(e) {
      if (isLocked) return;
      e.preventDefault();
      const p = getPos(e);
      fheDrawing = true;
      fheLastX = p.x;
      fheLastY = p.y;
      console.log('FHE drawing started at:', p.x, p.y);
    }

    function move(e) {
      if (!fheDrawing || isLocked) return;
      e.preventDefault();
      const p = getPos(e);
      ctx.beginPath();
      ctx.moveTo(fheLastX, fheLastY);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      fheLastX = p.x;
      fheLastY = p.y;
    }

    function end() {
      if (!fheDrawing || isLocked) return;
      fheDrawing = false;
      hasDrawing = true;
      updateEncryptButton();
    }

    //bind mouse and touch events
    mnistDrawCanvas.addEventListener('mousedown', start);
    mnistDrawCanvas.addEventListener('mousemove', move);
    mnistDrawCanvas.addEventListener('mouseup', end);
    mnistDrawCanvas.addEventListener('mouseleave', end);
    mnistDrawCanvas.addEventListener('touchstart', start, { passive: false });
    mnistDrawCanvas.addEventListener('touchmove', move, { passive: false });
    mnistDrawCanvas.addEventListener('touchend', end);
  }

  //enable encrypt button only when drawing + password = true
  function updateEncryptButton() {
    if (!mnistEncryptBtn) return;
    const hasPassword = mnistEncryptPassword ? mnistEncryptPassword.value.trim().length > 0 : false;
    mnistEncryptBtn.disabled = !(hasDrawing && hasPassword);
  }

  //convert canvas content to 28x28 grayscale MNIST input
  function canvasToMNIST(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const normalizedData = [];
    
    for (let i = 0; i < 28 * 28; i++) {
      const pixelIndex = i * 4;
      const gray = (imageData.data[pixelIndex] + imageData.data[pixelIndex + 1] + imageData.data[pixelIndex + 2]) / 3;
      normalizedData.push(gray / 255.0);
    }
    return normalizedData;
  }

  //render fake encrypted image for preview purposes
  function displayEncryptedPreview(_, password) {
    if (!mnistEncryptedPreview) return;
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(28, 28);

    let seed = 0;
    for (let i = 0; i < password.length; i++) {
      seed += password.charCodeAt(i);
    }

    function random() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    }

    for (let i = 0; i < 28 * 28; i++) {
      const pixelIndex = i * 4;
      const value = Math.floor(random() * 256);
      imageData.data[pixelIndex] = value;
      imageData.data[pixelIndex + 1] = value;
      imageData.data[pixelIndex + 2] = value;
      imageData.data[pixelIndex + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    const previewCtx = mnistEncryptedPreview.getContext('2d');
    mnistEncryptedPreview.width = 280;
    mnistEncryptedPreview.height = 280;
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(canvas, 0, 0, 280, 280);
  }

  //encrypts image data using provided password via trusted server
  async function encryptWithPython(imageData, password) {
    try {
      //sends drawing to trusted server for encryption
      const response = await fetch(`${TRUSTED_SERVER_URL}/api/fhe-mnist-encrypt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operation: 'encrypt',
          password: password,
          image_data: imageData,
          kernel_shape: kernelShape,
          stride: stride
        })
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'Encryption failed');
      }
      return result;
    } catch (error) {
      throw new Error(`Encryption failed: ${error.message}`);
    }
  }

  //handles encryption button click
  if (mnistEncryptBtn) {
    mnistEncryptBtn.addEventListener('click', async () => {
      const password = mnistEncryptPassword ? mnistEncryptPassword.value.trim() : '';
      if (!password) {
        mnistLog('[ERR] Please enter a password.');
        return;
      }

      //lock UI and store current password
      currentPassword = password;
      isLocked = true;
      mnistEncryptBtn.disabled = true;
      mnistEncryptPassword.disabled = true;
      mnistClearBtn.disabled = true;
      mnistDrawCanvas.classList.add('locked');

      try {
        //convert cnavas to MNIST array and encrypt
        const mnistData = canvasToMNIST(mnistDrawCanvas);
        encryptedData = await encryptWithPython(mnistData, password);
        displayEncryptedPreview(mnistData, password);
        if (mnistSendBtn) mnistSendBtn.disabled = false;
      } catch (error) {
        mnistLog(`[ERR] Encryption failed: ${error.message}`);
        mnistEncryptBtn.disabled = false;
        isLocked = false;
      }
    });
  }

  //handles sending encrypted data to untrusted server for FHE inference
  if (mnistSendBtn) {
    mnistSendBtn.addEventListener('click', async () => {
      if (!encryptedData) {
        mnistLog('[ERR] No encrypted data to send.');
        return;
      }
      mnistSendBtn.disabled = true;

      try {
        //prepare request with encrypted data
        const formData = new FormData();
        formData.append('fhe_data', JSON.stringify(encryptedData));
        const response = await fetch('/api/fhe-mnist/upload', {
          method: 'POST',
          body: formData
        });

        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || 'Upload failed');
        }

        //listen for log and result events
        const jobId = result.jobId;
        const eventSource = new EventSource(`/api/fhe-mnist/stream/${jobId}`);

        eventSource.addEventListener('log', (event) => {
          const data = JSON.parse(event.data);
          mnistLog(data.message);
        });

        eventSource.addEventListener('result', (event) => {
          const data = JSON.parse(event.data);
          if (data.encrypted_output) {
            encryptedResult = {
              encrypted_output: data.encrypted_output,
              context: encryptedData.context
            };
            mnistLog('[INFO] Received encrypted mnist result. Ready to decrypt.');
            if (mnistDecryptBtn) mnistDecryptBtn.disabled = false;
          } else if (data.error) {
            mnistLog(`[ERR] Server error: ${data.error}`);
          } else {
            mnistLog('[ERR] No encrypted output received');
          }
          eventSource.close();
        });

        eventSource.addEventListener('error', () => {
          mnistLog('[ERR] Server stream error');
          eventSource.close();
          mnistSendBtn.disabled = false;
        });

      } catch (error) {
        mnistLog(`[ERR] ${error.message}`);
        mnistSendBtn.disabled = false;
      }
    });
  }

  //decrypts encrypted prediction result using the password
  async function decryptWithPython(encryptedOutput, password) {
    try {
      //sends response to trusted server to be decrypted 
      const response = await fetch(`${TRUSTED_SERVER_URL}/api/fhe-mnist-decrypt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operation: 'decrypt',
          password: password,
          encrypted_output: encryptedOutput
        })
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'Decryption failed');
      }
      return result;
    } catch (error) {
      throw new Error(`Decryption failed: ${error.message}`);
    }
  }

  //handles decryption button click
  if (mnistDecryptBtn) {
    mnistDecryptBtn.addEventListener('click', async () => {
      const password = mnistDecryptPassword ? mnistDecryptPassword.value.trim() : '';
      if (!password) {
        mnistLog('[ERR] Please enter decryption password.');
        return;
      }
      if (!encryptedResult) {
        mnistLog('[ERR] No encrypted result to decrypt.');
        return;
      }
      if (password !== currentPassword) {
        mnistLog('[ERR] Incorrect decryption password.');
        return;
      }
      mnistDecryptBtn.disabled = true;
      try {
        const result = await decryptWithPython(encryptedResult, password);
        displayFHEResult(result.prediction.toString());
      } catch (error) {
        mnistLog(`[ERR] Decryption failed: ${error.message}`);
        mnistDecryptBtn.disabled = false;
      }
    });
  }

  //display FHE results on output canvas
  function displayFHEResult(digit) {
    if (!mnistOutputCanvas) return;
    const ctx = mnistOutputCanvas.getContext('2d');
    const rect = mnistOutputCanvas.getBoundingClientRect();
    mnistOutputCanvas.width = rect.width;
    mnistOutputCanvas.height = rect.height;
    
    ctx.clearRect(0, 0, mnistOutputCanvas.width, mnistOutputCanvas.height);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, mnistOutputCanvas.width, mnistOutputCanvas.height);
    ctx.fillStyle = '#111';
    ctx.font = 'bold 140px ui-sans-serif, system-ui, -apple-system';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(digit, mnistOutputCanvas.width / 2, mnistOutputCanvas.height / 2);
  }

  function clearFHEAll() {
    if (mnistDrawCanvas) {
      const ctx = mnistDrawCanvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, mnistDrawCanvas.width, mnistDrawCanvas.height);
    }
    
    if (mnistEncryptedPreview) {
      const previewCtx = mnistEncryptedPreview.getContext('2d');
      mnistEncryptedPreview.width = 280;
      mnistEncryptedPreview.height = 280;
      previewCtx.fillStyle = '#222';
      previewCtx.fillRect(0, 0, 280, 280);
    }
    
    if (mnistOutputCanvas) {
      const outputCtx = mnistOutputCanvas.getContext('2d');
      outputCtx.clearRect(0, 0, mnistOutputCanvas.width, mnistOutputCanvas.height);
    }
    
    mnistLogBox.textContent = '';
    mnistLog('[INFO] Waiting for image…');

    hasDrawing = false;
    encryptedData = null;
    encryptedResult = null;
    currentPassword = null;
    
    if (mnistEncryptPassword) mnistEncryptPassword.value = '';
    if (mnistDecryptPassword) mnistDecryptPassword.value = '';

    isLocked = false;
    if (mnistEncryptBtn) mnistEncryptBtn.disabled = false;
    if (mnistEncryptPassword) mnistEncryptPassword.disabled = false;
    if (mnistClearBtn) mnistClearBtn.disabled = false;
    if (mnistSendBtn) mnistSendBtn.disabled = true;
    if (mnistDecryptBtn) mnistDecryptBtn.disabled = true;
  }

  if (mnistEncryptPassword) {
    mnistEncryptPassword.addEventListener('input', updateEncryptButton);
  }

  //handles clear button to clear the drawing pane
  if (mnistClearBtn) {
    mnistClearBtn.addEventListener('click', () => {
      const ctx = mnistDrawCanvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, mnistDrawCanvas.width, mnistDrawCanvas.height);
      hasDrawing = false;
      updateEncryptButton();
      mnistLog('[INFO] Canvas cleared.');
    });
  }

  //handles the restart button to reset the logic and start again
  if (mnistRestartBtn) {
    mnistRestartBtn.addEventListener('click', clearFHEAll);
  }
  initFheDrawing();
  clearFHEAll();
  console.log('FHE MNIST script loaded successfully');
});