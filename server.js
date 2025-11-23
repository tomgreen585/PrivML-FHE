/**
 * Untrusted Server
 
 This server is one of the "untrusted" component of the PrivML-FHE.

 It is responsible for serving the web application frontend and handling client
 interactions with various ML/FHE pipelines. It never sees plaintext data when operating
 in FHE mode where payloads only contain encrypted data.

 Responsibilities
 - Server static frontend files (HTML/CSS/JS) for Border, MNIST, and Face ML demos
 - Provide REST and server event endpoints for running ML pipelines
 - Manage temporary job state using Maps for different tasks
 - Handle plaintext ML tasks
 - Handle encrypted ML tasks (Torch + FHE + TenSEAL)
 - Spawn Python processes (using Python kernel) to run the ML/FHE pipelines
 - Stream logs and results back to the frontend in real time using server events
 - Manage uplaods, temporary storage, and cleanup after job completion.

 NOTE: This server is considered "untrusted" because in the FHE pipelines, it
 cannot decrypt or view sensitive data and only processes/handles encrypted inputs
 and return encrypted results. A separate "trusted server" handles keys and encryption,
 decryption, and data handling for the FHE workflows.
 */

//required core and third party modulues
const express = require('express');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');

//intiialize express app
const app = express();
const PORT = 3000;

const repoRoot = __dirname;
const PYTHON_BIN = '/opt/anaconda3/envs/project/bin/python'; //python kernel path used to run plain/fhe models

//increase payload limits and server static frontend files
app.use(express.json({ limit: '500mb' }));
app.use(express.urlencoded({ extended: true, limit: '500mb' }));
app.use(express.static(path.join(__dirname, 'src', 'web_application', 'public')));

//memory maps for tracking jobs
const jobsBorder = new Map();
const jobsFace   = new Map();
const jobsMnist = new Map();
const jobsFheMnist = new Map();
const jobsFheFace = new Map();
const jobsFheBorder = new Map();

//directory to temporarily store uplaoded files
const uploadDir = path.join(__dirname, 'tmp_uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

//multer config for disk-based file uplaods (not production ready)
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, uploadDir),
  filename: (_, file, cb) => {
    const ext = path.extname(file.originalname || '');
    cb(null, `${uuidv4()}${ext || '.png'}`);
  },
});
const upload = multer({ storage });

//multer config for in-memory uploads (used for encrypted payloads)
const largeUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fieldSize: 500 * 1024 * 1024 }
});

//home page (START PAGE)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'start', 'start.html'));
});

//border_ml page (Border your Image)
app.get('/border_ml', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'border_ml', 'border.html'));
});

//mnist_ml page (Number Detector)
app.get('/mnist_ml', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'mnist_ml', 'mnist.html'));
});

//face_ml page (Face Detection)
app.get('/face_ml', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'face_ml', 'face.html'));
});

//run python scrips and stream stdout/stderr/result 
function sseRunPython(res, scriptPath, inputPath, outputPath) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });

  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  //spawn python process
  const py = spawn(PYTHON_BIN, [scriptPath, inputPath, outputPath], {
    cwd: path.dirname(scriptPath),
    env: {
      ...process.env,
      PYTHONPATH: repoRoot,
      PYTHONUNBUFFERED: '1',
    },
  });

  //stream logs from stdout
  py.stdout.setEncoding('utf8');
  py.stdout.on('data', (chunk) => {
    chunk.split(/\r?\n/).filter(Boolean).forEach(line => sendEvent('log', { message: line }));
  });

  //stream errors from stderr
  py.stderr.setEncoding('utf8');
  py.stderr.on('data', (chunk) => {
    chunk.split(/\r?\n/).filter(Boolean).forEach(line => sendEvent('log', { message: `[stderr] ${line}` }));
  });

  //handle completion and send result
  py.on('close', (code) => {
    try {
      if (code !== 0) {
        sendEvent('error', { message: `Python exited with code ${code}` });
      } else if (fs.existsSync(outputPath)) {
        const buf = fs.readFileSync(outputPath);
        const b64 = `data:image/png;base64,${buf.toString('base64')}`;
        sendEvent('result', { image: b64 });
      } else {
        sendEvent('error', { message: 'No output image produced' });
      }
    } catch (err) {
      sendEvent('error', { message: String(err) });
    } finally {}
  });
  return py;
}

/**
 * Border ML: Upload image + launch python pipeline
 */
app.post('/api/border/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });
  const jobId = uuidv4();
  jobsBorder.set(jobId, { filePath: req.file.path });
  res.json({ jobId });
});

app.get('/api/border/stream/:jobId', (req, res) => {
  //look up job id details
  const jobId = req.params.jobId;
  const job = jobsBorder.get(jobId);
  if (!job) return res.status(404).end('Unknown job');
  
  //paths for python script and output file
  const scriptPath = path.join(__dirname, 'src', 'border_ml_pipeline', 'border.py');
  const outputPath = path.join(uploadDir, `${jobId}-output.png`);
  
  //spawn python process
  const py = sseRunPython(res, scriptPath, job.filePath, outputPath);
  const finish = () => {
    try { fs.unlinkSync(job.filePath); } catch {}
    try { fs.unlinkSync(outputPath); } catch {}
    jobsBorder.delete(jobId);
    res.end();
  };
  //cleanup if client disconnects early
  req.on('close', () => { try { py.kill('SIGTERM'); } catch {} finish(); });
  py.on('close', finish);
});

/**
 * Face ML: Upload image + launch python pipeline
 */
app.post('/api/face/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });
  const jobId = uuidv4();
  jobsFace.set(jobId, { filePath: req.file.path });
  res.json({ jobId });
});

app.get('/api/face/stream/:jobId', (req, res) => {
  //check for job id
  const jobId = req.params.jobId;
  const job = jobsFace.get(jobId);
  if (!job) return res.status(404).end('Unknown job');

  //get ml pipeline script path
  const scriptPath = path.join(__dirname, 'src', 'face_ml_pipeline', 'face.py');
  const outputPath = path.join(uploadDir, `${jobId}-output.png`);
  
  //spawn python process
  const py = sseRunPython(res, scriptPath, job.filePath, outputPath);
  const finish = () => {
    try { fs.unlinkSync(job.filePath); } catch {}
    try { fs.unlinkSync(outputPath); } catch {}
    jobsFace.delete(jobId);
    res.end();
  };
  req.on('close', () => { try { py.kill('SIGTERM'); } catch {} finish(); });
  py.on('close', finish);
});

/**
 * MNIST ML: Upload image + launch python pipeline
 */
app.post('/api/mnist/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });
  const jobId = uuidv4();
  jobsMnist.set(jobId, { filePath: req.file.path });
  res.json({ jobId });
});

app.get('/api/mnist/stream/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobsMnist.get(jobId);
  if (!job) return res.status(404).end('Unknown job');
  
  //server events response headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });
  
  //send server event
  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  //paths for MNIST scrips and output JSON
  const scriptPath = path.join(__dirname, 'src', 'mnist_ml_pipeline', 'mnist.py');
  const outputPath = path.join(uploadDir, `${jobId}-output.json`);
  const repoRoot   = __dirname;

  //spawn python pipeline
  const py = spawn('/opt/anaconda3/envs/project/bin/python', [scriptPath, job.filePath, outputPath], {
    cwd: path.dirname(scriptPath),
    env: { ...process.env, PYTHONPATH: repoRoot }
  });

  //forward stdout
  py.stdout.setEncoding('utf8');
  py.stdout.on('data', chunk => {
    chunk.split(/\r?\n/).filter(Boolean).forEach(line => sendEvent('log', { message: line }));
  });

  //forward stderr
  py.stderr.setEncoding('utf8');
  py.stderr.on('data', chunk => {
    chunk.split(/\r?\n/).filter(Boolean).forEach(line => sendEvent('log', { message: `[stderr] ${line}` }));
  });

  //cleanup
  const finish = () => {
    try { fs.unlinkSync(job.filePath); } catch {}
    try { fs.unlinkSync(outputPath); } catch {}
    jobsMnist.delete(jobId);
    res.end();
  };

  //emit result or error on exit
  py.on('close', (code) => {
    try {
      if (code !== 0) {
        sendEvent('error', { message: `Python exited with code ${code}` });
      } else {
        if (fs.existsSync(outputPath)) {
          const obj = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
          sendEvent('result', obj);
        } else {
          sendEvent('error', { message: 'No output produced' });
        }
      }
    } catch (err) {
      sendEvent('error', { message: String(err) });
    } finally {
      finish();
    }
  });

  //clean on close
  req.on('close', () => { try { py.kill('SIGTERM'); } catch {} finish(); });
});

/**
 * FHE-MNIST: Upload encrypted MNIST data. Expects fhe_data in body. Launch python pipeline
 */
app.post('/api/fhe-mnist/upload', largeUpload.none(), (req, res) => {
  const fheDataString = req.body.fhe_data;

  //check for mandatory fields
  if (!fheDataString) {
    return res.status(400).json({ error: 'Missing FHE data' });
  }

  try {
    const fheData = JSON.parse(fheDataString);

    //ensure mandatory fields exist
    if (!fheData.encrypted_vector || !fheData.context || !fheData.kernel_shape || !fheData.stride) {
      return res.status(400).json({ error: 'Incomplete FHE data' });
    }

    const jobId = uuidv4();
    jobsFheMnist.set(jobId, { fheData });
    res.json({ jobId });

  } catch (error) {
    return res.status(400).json({ error: 'Invalid FHE data format' });
  }
});

//stream inference results, sends logs, cleans after completion
app.get('/api/fhe-mnist/stream/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobsFheMnist.get(jobId);
  if (!job) return res.status(404).end('Unknown job');
  
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });
  
  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  //ml pipeline script path
  const scriptPath = path.join(__dirname, 'src', 'fhe_mnist_ml_pipeline', 'fhe_mnist.py');
  const outputPath = path.join(uploadDir, `${jobId}-fhe-output.json`);
  const inputDataPath = path.join(uploadDir, `${jobId}-fhe-input.json`);
  
  try {
    fs.writeFileSync(inputDataPath, JSON.stringify(job.fheData));
  } catch (err) {
    sendEvent('error', { message: `Failed to write FHE data: ${err.message}` });
    return;
  }

  //spawn python process
  const py = spawn(PYTHON_BIN, [scriptPath, inputDataPath, outputPath], {
    cwd: path.dirname(scriptPath),
    env: { ...process.env, PYTHONPATH: repoRoot, PYTHONUNBUFFERED: '1' }
  });

  py.stdout.setEncoding('utf8');
  py.stdout.on('data', chunk => {
    chunk.split(/\r?\n/).filter(Boolean).forEach(line => sendEvent('log', { message: line }));
  });

  py.stderr.setEncoding('utf8');
  py.stderr.on('data', chunk => {
    chunk.split(/\r?\n/).filter(Boolean).forEach(line => sendEvent('log', { message: `[stderr] ${line}` }));
  });

  //clean on finish
  const finish = () => {
    try { fs.unlinkSync(inputDataPath); } catch {}
    try { fs.unlinkSync(outputPath); } catch {}
    jobsFheMnist.delete(jobId);
    res.end();
  };

  py.on('close', (code) => {
    try {
      if (code !== 0) {
        sendEvent('error', { message: `Python FHE script exited with code ${code}` });
      } else {
        if (fs.existsSync(outputPath)) {
          const obj = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
          sendEvent('result', obj);
        } else {
          sendEvent('error', { message: 'No FHE output produced' });
        }
      }
    } catch (err) {
      sendEvent('error', { message: String(err) });
    } finally {
      finish();
    }
  });

  req.on('close', () => { 
    try { py.kill('SIGTERM'); } catch {} 
    finish(); 
  });
});

/**
 * FHE-Face: Upload encrypted data. Expects fhe_data in body. Launches python pipeline
 */
app.post('/api/fhe-face/upload', largeUpload.none(), (req, res) => {
  const data = req.body.fhe_data;
  if (!data) return res.status(400).json({ error: 'Missing FHE data' });
  try {
  const jobId = uuidv4();
  jobsFheFace.set(jobId, { fheData: JSON.parse(data) });
  res.json({ jobId });
  } catch (err) {
  res.status(400).json({ error: 'Malformed FHE data' });
  }
});

//launches python pipeline, sends logs and result, clean input and state
app.get('/api/fhe-face/stream/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobsFheFace.get(jobId);
  if (!job) return res.status(404).end('Unknown job');

  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  //paths to ml pipelines
  const scriptPath = path.join(__dirname, 'src', 'fhe_face_ml_pipeline', 'fhe_face.py');
  const inputPath = path.join(uploadDir, `${jobId}-fhe-input.json`);
  const outputPath = path.join(uploadDir, `${jobId}-fhe-output.json`);

  fs.writeFileSync(inputPath, JSON.stringify(job.fheData));

  //spawn python process
  const py = spawn(PYTHON_BIN, [scriptPath, inputPath, outputPath], {
    cwd: path.dirname(scriptPath),
    env: { ...process.env, PYTHONPATH: repoRoot, PYTHONUNBUFFERED: '1' }
  });

  //send stdout
  py.stdout.on('data', chunk => {
    chunk.toString().split('\n').forEach(line => line && sendEvent('log', { message: line }));
  });
  
  //send stderr
  py.stderr.on('data', chunk => {
    chunk.toString().split('\n').forEach(line => line && sendEvent('log', { message: `[stderr] ${line}` }));
  });

  //on close clean process
  py.on('close', () => {
  if (fs.existsSync(outputPath)) {
    const result = JSON.parse(fs.readFileSync(outputPath));
    sendEvent('result', result);
  } else {
    sendEvent('error', { message: 'No output produced' });
  }
  fs.unlinkSync(inputPath);
  fs.unlinkSync(outputPath);
  jobsFheFace.delete(jobId);
  res.end();
  });

  req.on('close', () => { try { py.kill('SIGTERM'); } catch {} });
});

/**
 * FHE-Border: Stream encrypted border creation results. Luauches python pipeline
 */
app.post('/api/fhe-border/upload', largeUpload.none(), (req, res) => {
  const data = req.body.fhe_data;
  if (!data) return res.status(400).json({ error: 'Missing FHE border data' });
  try {
    const jobId = uuidv4();
    jobsFheBorder.set(jobId, { fheData: JSON.parse(data) });
    res.json({ jobId });
  } catch (err) {
    res.status(400).json({ error: 'Malformed FHE border data' });
  }
});

//launches python pipeline, sends logs and results, cleans input/output and states
app.get('/api/fhe-border/stream/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobsFheBorder.get(jobId);
  if (!job) return res.status(404).end('Unknown job');

  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  //path for model pipeline script
  const scriptPath = path.join(__dirname, 'src', 'fhe_border_ml_pipeline', 'fhe_border.py');
  const inputPath = path.join(uploadDir, `${jobId}-border-input.json`);
  const outputPath = path.join(uploadDir, `${jobId}-border-output.json`);

  fs.writeFileSync(inputPath, JSON.stringify(job.fheData));

  //spawn python proces
  const py = spawn(PYTHON_BIN, [scriptPath, inputPath, outputPath], {
    cwd: path.dirname(scriptPath),
    env: { ...process.env, PYTHONPATH: repoRoot, PYTHONUNBUFFERED: '1' }
  });

  //send stdout
  py.stdout.on('data', chunk => {
    chunk.toString().split('\n').forEach(line => line && sendEvent('log', { message: line }));
  });

  //send stderr
  py.stderr.on('data', chunk => {
    chunk.toString().split('\n').forEach(line => line && sendEvent('log', { message: `[stderr] ${line}` }));
  });

  //clean on close
  py.on('close', () => {
    if (fs.existsSync(outputPath)) {
      const result = JSON.parse(fs.readFileSync(outputPath));
      sendEvent('result', result);
    } else {
      sendEvent('error', { message: 'No border output produced' });
    }
    try { fs.unlinkSync(inputPath); } catch {}
    try { fs.unlinkSync(outputPath); } catch {}
    jobsFheBorder.delete(jobId);
    res.end();
  });

  req.on('close', () => { try { py.kill('SIGTERM'); } catch {} });
});

//start server
app.listen(PORT, () => {
  console.log(`Connect to server running at http://localhost:${PORT}`);
});