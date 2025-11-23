/**
 * Trusted Server
 * 
 * This server is the "trusted" component of the PrivML-FHE project.
 * 
 * It is responsible for performing sensitive operations such as encryption and decryption of data using
 * TenSEAL contexts. Unlike the untrusted server, this server does handle plaintext inputs and secret keys,
 * making it a secure component in the system.
 * 
 * Responsibilities:
 * - Accepts requests from the client.
 * - Spawn Python helper scrips (fhe_*_client_helper.py) to perform context generation, encryption,
 * and decryption tasks for the different ML pipelines.
 * - Manage input/output data via JSON or temporary files.
 * - Clean up temporary files created during encryption/decryption operations.
 * - Returns results (encrypted vectors, contexts, predictions, outputs, etc.) back to the requesting client.
 * 
 * Differences to Untrusted Server:
 * - Holds and uses secret keys to generate TenSEAL context.
 * - Has access to plaintext inputs (e.g. raw images) for encryption.
 * - Performs decryption of encrypted model outputs into usable predictions.
 * Must be secured and isolated since compromise would expose plaintext data.
 * 
 * NOTE: This server is considered "trusted" because it has access to plaintext and private cryptographic keys. It should
 * be kept isolated from untrusted enviornments and exposed only over secure channels.
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const tmp = require('tmp');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');

const app = express();
const PORT = 4000;
const PYTHON_BIN = '/opt/anaconda3/envs/project/bin/python';
const repoRoot = __dirname;

//allow requests from the untrusted server/frontend
app.use(cors({ origin: 'http://localhost:3000' }));
app.use(express.json({ limit: '500mb' }));

//configure upload storage
const upload = multer({ dest: path.join(__dirname, 'tmp_uploads') });

//paths to Python FHE helper scripts
const mnist_fhe_path = path.join(__dirname, 'src', 'web_application', 'public', 'mnist_ml', 'fhe_mnist_client_helper.py');
const face_fhe_path  = path.join(__dirname, 'src', 'web_application', 'public', 'face_ml', 'fhe_face_client_helper.py');
const border_fhe_path = path.join(__dirname, 'src', 'web_application', 'public', 'border_ml', 'fhe_border_client_helper.py');

/**
 * MNIST-FHE: Encrypt image data, expects mnist data, spawns helper to generate encrypted vector + context
 */
app.post('/api/fhe-mnist-encrypt', (req, res) => {
    const { operation, password, image_data, kernel_shape, stride } = req.body;
    
    //validate required parameters
    if (operation !== 'encrypt' || !password || !image_data || !kernel_shape || !stride) {
        return res.status(400).json({ error: 'Missing required parameters for encryption' });
    }

    //convert to JSOn string to pass to python script
    const imageDataJson = JSON.stringify(image_data);

    //spawn python process
    const py = spawn(PYTHON_BIN, [
        mnist_fhe_path,
        'encrypt',
        password,
        imageDataJson,
        kernel_shape[0].toString(),
        kernel_shape[1].toString(),
        stride.toString()
    ], {
        cwd: path.dirname(mnist_fhe_path),
        env: { ...process.env, PYTHONPATH: repoRoot }
    });

    //collect stdout and stderr outputs
    let stdout = '', stderr = '';
    py.stdout.on('data', chunk => stdout += chunk.toString());
    py.stderr.on('data', chunk => stderr += chunk.toString());

    //clean on close
    py.on('close', code => {
        if (code !== 0) {
            console.error(`[MNIST ENCRYPT ERROR] Exit ${code}: ${stderr}`);
            return res.status(500).json({ error: `Encryption failed: ${stderr}` });
        }
        try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
        } catch (e) {
            console.error(`[MNIST ENCRYPT] Failed to parse JSON: ${e.message}`);
            console.error("STDOUT:", stdout);
            res.status(500).json({ error: 'Failed to parse encryption result' });
        }
    });
});

/**
 * MNIST-FHE: Decrypt model output, expects output data, writes encrypted output to a temp file for python helper
 */
app.post('/api/fhe-mnist-decrypt', (req, res) => {
    const { operation, password, encrypted_output } = req.body;
    
    //ensure required parameters are present
    if (operation !== 'decrypt' || !password || !encrypted_output?.encrypted_output || !encrypted_output?.context) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    //write encrypted data to temp file
    const tempFile = tmp.fileSync({ postfix: '.json' });
    fs.writeFileSync(tempFile.name, JSON.stringify(encrypted_output));

    //spawn python helper script
    const py = spawn(PYTHON_BIN, [mnist_fhe_path, 'decrypt', password, tempFile.name], {
        cwd: path.dirname(mnist_fhe_path),
        env: { ...process.env, PYTHONPATH: repoRoot }
    });

    //collect stdout/stderr outputs
    let stdout = '', stderr = '';
    py.stdout.on('data', chunk => stdout += chunk.toString());
    py.stderr.on('data', chunk => stderr += chunk.toString());

    //clean on close
    py.on('close', code => {
        fs.unlinkSync(tempFile.name);
        if (code !== 0) return res.status(500).json({ error: `Decryption failed: ${stderr}` });
        try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
        } catch {
            res.status(500).json({ error: '3Failed to parse decryption result' });
        }
    });
});

/**
 * Face-FHE: Encrypt face image, expects face data, spawns helper to generate encrypted tensor + context
 */
app.post('/api/fhe-face-encrypt', upload.single('image'), (req, res) => {
  const { password } = req.body;
  
  //check for required arguments
  if (!password || !req.file) {
    return res.status(400).json({ error: 'Missing password or image file' });
  }

  const imagePath = req.file.path;

  //spawn python helper script
  const py = spawn(PYTHON_BIN, [face_fhe_path, 'encrypt', password, imagePath], {
    cwd: path.dirname(face_fhe_path),
    env: { ...process.env, PYTHONPATH: repoRoot }
  });

  //collect stdout/stderr outputs
  let stdout = '', stderr = '';
  py.stdout.on('data', chunk => { stdout += chunk.toString(); });
  py.stderr.on('data', chunk => { stderr += chunk.toString(); });

  //clean on close
  py.on('close', code => {
    try { fs.unlinkSync(imagePath); } catch {}
    
    if (code !== 0) {
      console.error(`[PY ERROR] Exit ${code}: ${stderr}`);
      return res.status(500).json({ error: stderr || 'Python failed' });
    }
    
    if (stderr) {
      console.log('[PYTHON LOGS]', stderr);
    }
    
    try {
      //helpers print logs before JSON - clea to valid JSON
      let cleanedOutput = stdout.trim();
      let jsonStart = cleanedOutput.lastIndexOf('{');
      if (jsonStart !== -1) {
        cleanedOutput = cleanedOutput.substring(jsonStart);
      }

      let braceCount = 0;
      let jsonEnd = -1;
      for (let i = 0; i < cleanedOutput.length; i++) {
        if (cleanedOutput[i] === '{') braceCount++;
        else if (cleanedOutput[i] === '}') {
          braceCount--;
          if (braceCount === 0) {
            jsonEnd = i + 1;
            break;
          }
        }
      }
      if (jsonEnd !== -1) {
        cleanedOutput = cleanedOutput.substring(0, jsonEnd);
      }
      
      const result = JSON.parse(cleanedOutput);
      return res.json(result);
      
    } catch (e) { //catch errors
      console.error("[ERR] Failed to parse JSON output:", e.message);
      console.error("STDOUT length:", stdout.length);
      console.error("STDOUT preview (first 500 chars):", stdout.slice(0, 500));
      console.error("STDOUT preview (last 500 chars):", stdout.slice(-500));
      return res.status(500).json({ error: 'Failed to parse encryption result' });
    }
  });
});

/**
 * Face-FHE: Decrypt result, expects model outputs, writes encrypted output file for Python helper
 */
app.post('/api/fhe-face-decrypt', (req, res) => {
    const { password, encrypted_output } = req.body;
    
    //check for mandatory arguments
    if (!password || !encrypted_output?.encrypted_output || !encrypted_output?.context) {
        return res.status(400).json({ error: 'Missing fields' });
    }

    const tempFile = tmp.fileSync({ postfix: '.json' });
    fs.writeFileSync(tempFile.name, JSON.stringify(encrypted_output));

    //spawn python helper script
    const py = spawn(PYTHON_BIN, [face_fhe_path, 'decrypt', password, tempFile.name], {
        cwd: path.dirname(face_fhe_path),
        env: { ...process.env, PYTHONPATH: repoRoot }
    });

    //collect stdout/stderr outputs
    let stdout = '', stderr = '';
    py.stdout.on('data', chunk => stdout += chunk.toString());
    py.stderr.on('data', chunk => stderr += chunk.toString());

    //clean on close
    py.on('close', code => {
        fs.unlinkSync(tempFile.name);
        if (code !== 0) return res.status(500).json({ error: stderr });
        try {
            res.json(JSON.parse(stdout));
        } catch {
            res.status(500).json({ error: '1Failed to parse decryption result' });
        }
    });
});

/**
 * Border FHE: encrypt image, expects border data, spawns python helper to produce encrypted border input + context
 */
app.post('/api/fhe-border-encrypt', upload.single('image'), (req, res) => {
  const { password } = req.body;
  
  //check for mandatory arugments
  if (!password || !req.file) {
    return res.status(400).json({ error: 'Missing password or image file' });
  }

  const imagePath = req.file.path;

  //spawn python helper script
  const py = spawn(PYTHON_BIN, [border_fhe_path, 'encrypt', password, imagePath], {
    cwd: path.dirname(border_fhe_path),
    env: { ...process.env, PYTHONPATH: repoRoot }
  });

  //collect stdout/stderr outputs
  let stdout = '', stderr = '';
  py.stdout.on('data', chunk => { stdout += chunk.toString(); });
  py.stderr.on('data', chunk => { stderr += chunk.toString(); });

  //clean on close
  py.on('close', code => {
    try { fs.unlinkSync(imagePath); } catch {}
    
    if (code !== 0) {
      console.error(`[BORDER PY ERROR] Exit ${code}: ${stderr}`);
      return res.status(500).json({ error: stderr || 'Border encryption failed' });
    }
    
    if (stderr) {
      console.log('[BORDER PYTHON LOGS]', stderr);
    }
    
    try {
      //clean stdout and parse JSON
      let cleanedOutput = stdout.trim();
      let jsonStart = cleanedOutput.lastIndexOf('{');
      if (jsonStart !== -1) {
        cleanedOutput = cleanedOutput.substring(jsonStart);
      }
      
      let braceCount = 0;
      let jsonEnd = -1;
      for (let i = 0; i < cleanedOutput.length; i++) {
        if (cleanedOutput[i] === '{') braceCount++;
        else if (cleanedOutput[i] === '}') {
          braceCount--;
          if (braceCount === 0) {
            jsonEnd = i + 1;
            break;
          }
        }
      }
      
      if (jsonEnd !== -1) {
        cleanedOutput = cleanedOutput.substring(0, jsonEnd);
      }
      
      //return json result
      const result = JSON.parse(cleanedOutput);
      return res.json(result);
      
    } catch (e) {
      console.error("[BORDER ERR] Failed to parse JSON output:", e.message);
      console.error("STDOUT length:", stdout.length);
      console.error("STDOUT preview (first 500 chars):", stdout.slice(0, 500));
      console.error("STDOUT preview (last 500 chars):", stdout.slice(-500));
      return res.status(500).json({ error: 'Failed to parse border encryption result' });
    }
  });
});

/**
 * Border FHE: Decrypt result, expects model outputs, writes encrypted output to temp file for python helper
 */
app.post('/api/fhe-border-decrypt', (req, res) => {
    const { password, encrypted_output } = req.body;
    //check for mandatory arguments
    if (!password || !encrypted_output?.encrypted_output || !encrypted_output?.context) {
        return res.status(400).json({ error: 'Missing fields for border decryption' });
    }

    const tempFile = tmp.fileSync({ postfix: '.json' });
    fs.writeFileSync(tempFile.name, JSON.stringify(encrypted_output));
    
    //get path of helper python script
    const border_fhe_path = path.join(__dirname, 'src', 'web_application', 'public', 'border_ml', 'fhe_border_client_helper.py');

    //spawn python process
    const py = spawn(PYTHON_BIN, [border_fhe_path, 'decrypt', password, tempFile.name], {
        cwd: path.dirname(border_fhe_path),
        env: { ...process.env, PYTHONPATH: repoRoot }
    });

    //collect stdout/stderr outputs
    let stdout = '', stderr = '';
    py.stdout.on('data', chunk => stdout += chunk.toString());
    py.stderr.on('data', chunk => stderr += chunk.toString());

    //clean on close
    py.on('close', code => {
        fs.unlinkSync(tempFile.name);
        if (code !== 0) {
            console.error(`[BORDER DECRYPT ERROR] Exit ${code}: ${stderr}`);
            return res.status(500).json({ error: stderr });
        }
        try {
            const result = JSON.parse(stdout);
            res.json(result);
        } catch (e) {
            console.error("[BORDER DECRYPT ERR] Failed to parse JSON:", e.message);
            res.status(500).json({ error: 'Failed to parse border decryption result' });
        }
    });
});

//start trusted server
app.listen(PORT, () => {
  console.log(`Trusted server running at http://localhost:${PORT}`);
});
