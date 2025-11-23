//MNIST NAV SCRIPT

//runs once DOM loaded
(function () {
  //grab DOM elements
  const fileInput = document.getElementById('fileInput');
  const pickBtn = document.getElementById('pickBtn');
  const dropzone = document.getElementById('dropzone');
  const preview = document.getElementById('previewImg');
  const confirm = document.getElementById('confirmImg');
  const sendBtn = document.getElementById('sendBtn');
  const logBox = document.getElementById('logBox');
  const canvas = document.getElementById('outputCanvas');
  const saveBtn = document.getElementById('saveBtn');
  const resetBtn = document.getElementById('resetBtn');
  let loadedImage = null; //image uplaoded by the user

  //append log messages to the logging box
  function log(line) {
    logBox.textContent += `\n${line}`;
    logBox.scrollTop = logBox.scrollHeight;
  }

  //reset UI to initial state
  function clearAll() {
    loadedImage = null;
    preview.src = '';
    confirm.src = '';
    preview.style.display = 'none';
    confirm.style.display = 'none';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    logBox.textContent = '[INFO] Waiting for image…';
    sendBtn.disabled = true;
    saveBtn.disabled = true;
  }

  //ensure canvas size matches CSS display dimensions
  function sizeCanvasToCSS() {
    const rect = canvas.getBoundingClientRect();
    canvas.width  = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
  }
  sizeCanvasToCSS();
  window.addEventListener('resize', sizeCanvasToCSS);

  //trigger hidden file input when "pick" button is clicked
  pickBtn.addEventListener('click', (e) => {
  e.preventDefault();
  fileInput.click();
});

  fileInput.addEventListener('change', onFiles);
  //trigger hidden file input when "pick" button is clicked
  pickBtn.addEventListener('click', (e) => {
  e.preventDefault();
  fileInput.click();
});

//handle drag/drop and keyboard accessibility on dropzone
document.addEventListener('DOMContentLoaded', () => {
  const buttons = document.querySelectorAll('.dropbtn');

  //dropdown handling for nav items
  function closeAll(except) {
    document.querySelectorAll('.dropdown-content').forEach(dc => {
      if (dc !== except) dc.style.display = 'none';
    });
  }

  buttons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.getAttribute('data-dropdown');
      const menu = document.getElementById('dropdown-' + id);
      const open = menu.style.display === 'block';
      closeAll(menu);
      menu.style.display = open ? 'none' : 'block';
    });
  });

  document.addEventListener('click', () => closeAll());
});

document.addEventListener('DOMContentLoaded', () => {
  const dropzone  = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const pickBtn   = document.getElementById('pickBtn');

  pickBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
  dropzone.addEventListener('click', (e) => { if (e.target !== fileInput) fileInput.click(); });
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
});

dropzone.addEventListener('click', (e) => {
  if (e.target === fileInput) return;
  fileInput.click();
});

dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

  function onFiles(e) {
    const f = e.target.files[0];
    if (f) handleFile(f);
  }

  //load image file into preview/confirm image and canvas
  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      log('[ERR] Please choose an image file.');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        loadedImage = img;
        preview.src = img.src;
        confirm.src = img.src;
        preview.style.display = 'block';
        confirm.style.display = 'block';
        sendBtn.disabled = false;
        log('[INFO] Image loaded.');
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  }

  //simulated upload button
  sendBtn.addEventListener('click', async () => {
    if (!loadedImage) return;

    log('[INFO] Passing image to model…');
    sendBtn.disabled = true;

    await pause(300);
    log('[INFO] Computing metrics…');

    await pause(400);
    log('[INFO] Time taken to compute: ~0.7s');

    await pause(200);
    log('[INFO] Generating output…');

    saveBtn.disabled = false;
    log('[INFO] Sending back to user.');
  });

  //download canvas as png
  saveBtn.addEventListener('click', () => {
    const a = document.createElement('a');
    a.download = 'bordered-image.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
  });

  //reset button clears everything - resets UI
  resetBtn.addEventListener('click', clearAll);

  //sleep/pause helper
  function pause(ms) { 
    return new Promise(res => setTimeout(res, ms)); 
  }
})();
