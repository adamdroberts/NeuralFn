const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

let backendProcess = null;
let mainWindow = null;
let backendPort = 8000;

// Resolve directories
const projectRoot = path.dirname(__dirname);
const userDataPath = app.getPath('userData');

console.log('NeuralFn Desktop launching...');
console.log('Project Root:', projectRoot);
console.log('User Data Path:', userDataPath);

// Function to find a free port
async function getFreePort() {
  const { getPort } = require('get-port-please');
  try {
    const port = await getPort({ port: 8000, host: '127.0.0.1' });
    console.log(`Resolved free port: ${port}`);
    return port;
  } catch (err) {
    console.error('Failed to resolve port, using fallback 8000:', err);
    return 8000;
  }
}

// Function to check if the backend is ready
function checkBackendReady(port, callback) {
  const options = {
    hostname: '127.0.0.1',
    port: port,
    path: '/api/bootstrap',
    method: 'GET',
    timeout: 1000
  };

  const req = http.request(options, (res) => {
    console.log(`Backend ping response code: ${res.statusCode}`);
    callback(true);
  });

  req.on('error', () => {
    callback(false);
  });

  req.on('timeout', () => {
    req.destroy();
    callback(false);
  });

  req.end();
}

// Poll the backend until it is ready
function waitForBackend(port, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      checkBackendReady(port, (ready) => {
        if (ready) {
          clearInterval(interval);
          resolve();
        } else if (Date.now() - startTime > timeoutMs) {
          clearInterval(interval);
          reject(new Error('Backend server failed to start in time.'));
        }
      });
    }, 200);
  });
}

// Spawns the FastAPI backend process
function spawnBackend(port) {
  // Ensure folders exist in userData
  const snapshotsDir = path.join(userDataPath, 'session_snapshots');
  const artifactsDir = path.join(userDataPath, 'artifacts');
  fs.mkdirSync(snapshotsDir, { recursive: true });
  fs.mkdirSync(artifactsDir, { recursive: true });

  const env = {
    ...process.env,
    NEURALFN_DATABASE_URL: `sqlite:///${path.join(userDataPath, 'neuralfn.db')}`,
    NEURALFN_SNAPSHOTS_DIR: snapshotsDir,
    NEURALFN_ARTIFACTS_DIR: artifactsDir,
    NEURALFN_REDIS_URL: '', // Force MemoryLiveStateStore (Offline Mode)
    NEURALFN_ALLOW_ORIGINS: '*',
    PYTHONUNBUFFERED: '1'
  };

  let pythonExecutable = 'python';
  let args = [];

  // Check if we are running a packaged production build
  const isPackaged = app.isPackaged;
  const packagedBackend = process.platform === 'win32'
    ? path.join(process.resourcesPath, 'backend', 'server.exe')
    : path.join(process.resourcesPath, 'backend', 'server');

  if (isPackaged && fs.existsSync(packagedBackend)) {
    console.log('Spawning packaged production backend:', packagedBackend);
    pythonExecutable = packagedBackend;
    args = ['--port', port.toString()];
  } else {
    console.log('Running in development mode. Spawning server using environment Python...');
    // We can also try python3 as a fallback if python is not available
    pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    args = ['-m', 'uvicorn', 'server.app:app', '--host', '127.0.0.1', '--port', port.toString()];
  }

  backendProcess = spawn(pythonExecutable, args, {
    cwd: projectRoot,
    env: env
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend]: ${data.toString().trim()}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend-Error]: ${data.toString().trim()}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
  });
}

function createMainWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    title: 'NeuralFn',
    backgroundColor: '#030712', // Matches gray-950
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true
    }
  });

  // Load the backend SPA (which serves our built Vite frontend at the root)
  mainWindow.loadURL(`http://127.0.0.1:${port}`);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App lifecycle handlers
app.whenReady().then(async () => {
  backendPort = await getFreePort();
  spawnBackend(backendPort);

  try {
    await waitForBackend(backendPort);
    createMainWindow(backendPort);
  } catch (err) {
    console.error('Initialization error:', err);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null && backendPort) {
    createMainWindow(backendPort);
  }
});

// Clean up processes on exit
app.on('will-quit', () => {
  if (backendProcess) {
    console.log('Stopping backend subprocess...');
    backendProcess.kill('SIGINT');
  }
});
