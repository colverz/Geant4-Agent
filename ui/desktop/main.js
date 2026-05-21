const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron/main");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const HOST = process.env.G4_DESKTOP_HOST || "127.0.0.1";
const PORT = process.env.G4_DESKTOP_PORT || "8088";
const START_URL = `http://${HOST}:${PORT}/?desktop=1`;
const REPO_ROOT = path.resolve(__dirname, "..", "..");

let bridgeProcess = null;
let mainWindow = null;

function pythonCandidates() {
  const candidates = [];
  const envPython = process.env.G4_DESKTOP_PYTHON;
  if (envPython) candidates.push(envPython);
  const venvPython = path.resolve(REPO_ROOT, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venvPython)) candidates.push(venvPython);
  candidates.push("python");
  candidates.push("py");
  return [...new Set(candidates)];
}

function spawnBridge(command) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      command,
      ["-m", "ui.desktop.runtime_bridge", "--host", HOST, "--port", PORT],
      {
        cwd: REPO_ROOT,
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      },
    );
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.stdout.on("data", (chunk) => process.stdout.write(`[geant4-bridge] ${chunk}`));
      child.stderr.on("data", (chunk) => process.stderr.write(`[geant4-bridge] ${chunk}`));
      child.on("exit", (code) => {
        process.stderr.write(`[geant4-bridge] exited with code ${code}\n`);
        if (bridgeProcess === child) bridgeProcess = null;
      });
      resolve(child);
    }, 700);
    child.once("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.once("exit", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`bridge exited during startup with code ${code}`));
    });
  });
}

async function startBridge() {
  if (bridgeProcess) return bridgeProcess;
  let lastError = null;
  for (const candidate of pythonCandidates()) {
    try {
      bridgeProcess = await spawnBridge(candidate);
      return bridgeProcess;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`Unable to start Python UI bridge. ${lastError || ""}`);
}

async function waitForBridge(timeoutMs = 20000) {
  const base = `http://${HOST}:${PORT}`;
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(`${base}/api/runtime`);
      if (response.ok) return;
    } catch (_) {
      // Bridge not ready yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 350));
  }
  throw new Error(`UI bridge did not become ready within ${timeoutMs} ms.`);
}

async function createMainWindow() {
  await startBridge();
  await waitForBridge();
  const win = new BrowserWindow({
    width: 1480,
    height: 980,
    minWidth: 1120,
    minHeight: 760,
    backgroundColor: "#081018",
    autoHideMenuBar: true,
    title: "Geant4-Agent",
    titleBarStyle: "hidden",
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  mainWindow = win;
  win.on("closed", () => {
    if (mainWindow === win) mainWindow = null;
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  await win.loadURL(START_URL);
}

ipcMain.handle("window:minimize", () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  if (win) win.minimize();
});

ipcMain.handle("window:toggle-maximize", () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  if (!win) return;
  if (win.isMaximized()) win.unmaximize();
  else win.maximize();
});

ipcMain.handle("window:close", () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  if (win) win.close();
});

app.whenReady().then(async () => {
  try {
    await createMainWindow();
  } catch (error) {
    dialog.showErrorBox("Geant4-Agent Desktop Startup Failed", String(error));
    app.quit();
    return;
  }
  app.on("activate", async () => {
    if (BrowserWindow.getAllWindows().length === 0) await createMainWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (bridgeProcess) {
    bridgeProcess.kill();
    bridgeProcess = null;
  }
});
