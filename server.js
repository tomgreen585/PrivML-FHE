const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

// Serve static assets
app.use(express.static(path.join(__dirname, 'src', 'web_application', 'public')));

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'start', 'start.html'));
});

app.get('/border_ml', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'border_ml', 'border.html'));
});

app.get('/mnist_ml', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'mnist_ml', 'mnist.html'));
});

app.get('/face_ml', (req, res) => {
  res.sendFile(path.join(__dirname, 'src', 'web_application', 'public', 'face_ml', 'face.html'));
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
