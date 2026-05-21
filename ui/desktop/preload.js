const { contextBridge, ipcRenderer } = require("electron/renderer");

contextBridge.exposeInMainWorld("geant4Desktop", {
  shell: "chromium",
  bridgeMode: "localhost-http",
  version: "0.2.0",
  minimize: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximize: () => ipcRenderer.invoke("window:toggle-maximize"),
  close: () => ipcRenderer.invoke("window:close"),
});
