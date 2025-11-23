//FACE SCRIPT

//wait until DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  //get dom elements
  const fileInput = document.getElementById('fileInput');
  const pickBtn = document.getElementById('pickBtn');
  const dropzone = document.getElementById('dropzone');
  const confirm = document.getElementById('confirmImg');
  const sendBtn = document.getElementById('sendBtn');
  const logBox = document.getElementById('logBox');
  const canvas = document.getElementById('outputCanvas');
  const saveBtn = document.getElementById('saveBtn');
  const resetBtn = document.getElementById('resetBtn');

  //runtime variables
  let selectedFile = null;
  let latestDataUrl = null;

  //logging function to append to logging output
  function log(line) {
    logBox.textContent += `\n${line}`;
    logBox.scrollTop = logBox.scrollHeight;
  }

  //canvas resizing to CSS dimensions
  function sizeCanvasToCSS() {
    const r = canvas.getBoundingClientRect();
    canvas.width  = Math.round(r.width);
    canvas.height = Math.round(r.height);
  }

  //reset UI and state
  function clearAll() {
    selectedFile = null;
    latestDataUrl = null;
    confirm.src = '';
    confirm.style.display = 'none';
    sizeCanvasToCSS();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    logBox.textContent = '';
    log('[INFO] Waiting for image...')
    sendBtn.disabled = true;
    saveBtn.disabled = true;
  }

  //initialize
  sizeCanvasToCSS();
  window.addEventListener('resize', sizeCanvasToCSS);

  //predefine template images, allows clicking on images to simulate user upload of file
  document.querySelectorAll('.template-img').forEach(img => {
    img.addEventListener('click', async () => {
      const imagePath = `/face_ml/template_images/${img.dataset.image}`;
      const response = await fetch(imagePath);
      const blob = await response.blob();
      const file = new File([blob], img.dataset.image, { type: blob.type });
      handleNewFile(file);
    });
  });

  //file selection event
  pickBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.value = '';
    fileInput.click();
  });

  //clicking anywhere in dropzone opens picker
  dropzone.addEventListener('click', (e) => {
    if (e.target !== fileInput) {
      fileInput.value = '';
      fileInput.click();
    }
  });

  //enter or space triggers file input
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.value = '';
      fileInput.click();
    }
  });

  //drag and drop functionality
  ['dragenter', 'dragover'].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.add('dragging');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropzone.addEventListener(evt, e => {
      e.preventDefault(); e.stopPropagation();
      dropzone.classList.remove('dragging');
    })
  );
  dropzone.addEventListener('drop', e => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleNewFile(file);
  });

  //file picker change event
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) {
      log('[INFO] Picker canceled.');
      return;
    }
    handleNewFile(f);
  });

  //handles new file selection from user
  function handleNewFile(file) {
    selectedFile = file;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      confirm.src = url;
      confirm.style.display = 'block';
      sendBtn.disabled = false;
    };
    img.onerror = () => {
      log('[ERR] Could not preview the selected file as an image.');
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  //sends image to server for processing and model inference
  sendBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      log('[ERR] No file selected.');
      return;
    }

    sendBtn.disabled = true;

    const form = new FormData();
    form.append('image', selectedFile);

    let jobId = null;
    try { //construct request
      const res = await fetch('/api/face/upload', { method: 'POST', body: form });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Upload failed');
      jobId = json.jobId;
    } catch (err) {
      log(`[ERR] ${err.message}`);
      sendBtn.disabled = false;
      return;
    }

    //handles streamed response from server
    const es = new EventSource(`/api/face/stream/${jobId}`);

    es.addEventListener('log', (ev) => {
      const data = JSON.parse(ev.data);
      log(data.message);
    });

    es.addEventListener('result', (ev) => {
      const data = JSON.parse(ev.data);
      latestDataUrl = data.image; //get processed image URL
      drawDataUrlToCanvas(latestDataUrl);
      saveBtn.disabled = false;
      log('[INFO] Received result.');
      es.close();
    });

    es.addEventListener('error', () => {
      log('[ERR] Stream error.');
      es.close();
      sendBtn.disabled = false;
    });
  });

  //draw model output on canvas
  function drawDataUrlToCanvas(dataUrl) {
    sizeCanvasToCSS();
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      const cw = canvas.width, ch = canvas.height;
      const scale = Math.min(cw / img.width, ch / img.height);
      const w = img.width * scale, h = img.height * scale;
      const x = (cw - w) / 2, y = (ch - h) / 2;
      ctx.clearRect(0, 0, cw, ch);
      ctx.drawImage(img, x, y, w, h);
    };
    img.src = dataUrl;
  }

  //save output image as png file
  saveBtn.addEventListener('click', () => {
    if (!latestDataUrl) return;
    const a = document.createElement('a');
    a.href = latestDataUrl;
    a.download = 'detected-face-image.png';
    a.click();
  });

  //reset button to clear UI pipeline
  resetBtn.addEventListener('click', clearAll);

  //initial clear
  clearAll();
});
