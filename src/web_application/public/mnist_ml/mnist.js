//PLAINTEXT MNIST SCRIPT

//runs once DOM loaded
document.addEventListener('DOMContentLoaded', () => {
  const drawCanvas = document.getElementById('drawCanvas');
  const clearBtn = document.getElementById('clearBtn');
  const confirm = document.getElementById('confirmImg');
  const sendBtn = document.getElementById('sendBtn');
  const logBox = document.getElementById('logBox');
  const canvasOut = document.getElementById('outputCanvas');
  const resetBtn = document.getElementById('resetBtn');

  let latestDataUrl = null;
  let drawing = false;
  let lastX = 0, lastY = 0;

  //append log messages to log box and scroll
  function log(line){
    logBox.textContent += `\n${line}`;
    logBox.scrollTop = logBox.scrollHeight;
  }

  //resize the output canvas to match display size
  function sizeOutCanvas(){
    const r = canvasOut.getBoundingClientRect();
    canvasOut.width  = Math.round(r.width);
    canvasOut.height = Math.round(r.height);
  }

  //clears drawing canvas, output, and preview image
  function clearAll(){
    const ctx = drawCanvas.getContext('2d');
    ctx.save();
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0,0,drawCanvas.width, drawCanvas.height);
    ctx.restore();

    if (confirm) {
      confirm.src = '';
      confirm.style.display = 'none';
    }

    latestDataUrl = null;
    sendBtn.disabled = true;
    const octx = canvasOut.getContext('2d');
    octx.clearRect(0,0,canvasOut.width, canvasOut.height);
    logBox.textContent = '[INFO] Waiting for image…';
  }

  //auto-resize canvas on window size changes
  sizeOutCanvas();
  window.addEventListener('resize', sizeOutCanvas);

  //initialize drawing logic on canvas
  (function initDrawing(){
    const ctx = drawCanvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0,0,drawCanvas.width, drawCanvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = '#fff';

    function getPos(e){
      const rect = drawCanvas.getBoundingClientRect();
      if (e.touches && e.touches[0]){
        const t = e.touches[0];
        return { x: t.clientX - rect.left, y: t.clientY - rect.top };
      }
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function start(e){
      e.preventDefault();
      const p = getPos(e);
      drawing = true;
      lastX = p.x; lastY = p.y;
    }

    function move(e){
      if (!drawing) return;
      const p = getPos(e);
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      lastX = p.x; lastY = p.y;
    }

    function end(){
      if (!drawing) return;
      drawing = false;
      //convert canvas to data URL and show preview
      const url = drawCanvas.toDataURL('image/png');
      latestDataUrl = url;

      if (confirm) {
        confirm.src = url;
        confirm.style.display = 'block';
      }
      sendBtn.disabled = false;
    }

    //bind drawing events
    drawCanvas.addEventListener('mousedown', start);
    drawCanvas.addEventListener('mousemove', move);
    drawCanvas.addEventListener('mouseup', end);
    drawCanvas.addEventListener('mouseleave', end);
    drawCanvas.addEventListener('touchstart', start, {passive:false});
    drawCanvas.addEventListener('touchmove', move, {passive:false});
    drawCanvas.addEventListener('touchend', end);

    //bind clear button
    clearBtn.addEventListener('click', clearAll);
  })();

  //handle submit/upload button
  sendBtn.addEventListener('click', async () => {
    sendBtn.disabled = true;

    //convert canvas image to PNG and send it
    const blob = await new Promise(res => drawCanvas.toBlob(res, 'image/png'));
    const form = new FormData();
    form.append('image', blob, 'drawing.png');

    let jobId = null;
    try {
      const res = await fetch('/api/mnist/upload', { method: 'POST', body: form });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Upload failed');
      jobId = json.jobId;
    } catch (err){
      log(`[ERR] ${err.message}`);
      sendBtn.disabled = false;
      return;
    }

    //open server-sent events stream to receive log/results
    const es = new EventSource(`/api/mnist/stream/${jobId}`);
    es.addEventListener('open', () => {
    });
    es.addEventListener('log', (ev) => {
      const data = JSON.parse(ev.data);
      log(data.message);
    });
    es.onmessage = (event) => {
      log(`[INFO] Generic message: ${event.data}`);
    };

    es.addEventListener('result', (ev) => {
      const data = JSON.parse(ev.data);
      if (data.image) {
        latestDataUrl = data.image;
        drawResultImage(latestDataUrl);
      //render label prediction if present
      } else if (typeof data.label !== 'undefined') {
        drawLabelToOutput(String(data.label));
        latestDataUrl = null;
      }
      log('[INFO] Received result.');
    });

    es.addEventListener('error', (ev) => {
      console.error(ev);
      es.close();
      sendBtn.disabled = false;
    });
  });

  //draw a returned image (output from server)
  function drawResultImage(dataUrl){
    sizeOutCanvas();
    const ctx = canvasOut.getContext('2d');
    const img = new Image();
    img.onload = () => {
      const cw = canvasOut.width, ch = canvasOut.height;
      const s = Math.min(cw/img.width, ch/img.height);
      const w = img.width*s, h = img.height*s;
      const x = (cw-w)/2, y = (ch-h)/2;
      ctx.clearRect(0,0,cw,ch);
      ctx.drawImage(img, x, y, w, h);
    };
    img.src = dataUrl;
  }

  //draw a prediction label text to the canvas
  function drawLabelToOutput(text){
    sizeOutCanvas();
    const ctx = canvasOut.getContext('2d');
    const cw = canvasOut.width, ch = canvasOut.height;
    ctx.clearRect(0,0,cw,ch);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0,0,cw,ch);
    ctx.fillStyle = '#111';
    ctx.font = 'bold 140px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, cw/2, ch/2);
  }

  //reset UI state on reset click
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      log('[INFO] Resetting interface.');
      clearAll();
    });
  }
  //clear everything initially
  clearAll();
});
