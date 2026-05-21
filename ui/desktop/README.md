`ui/desktop/` is the Chromium desktop shell for the local Geant4-Agent UI.

Startup:

```powershell
powershell -ExecutionPolicy Bypass -File ui\desktop\start_desktop.ps1
```

What it does:
- starts the Python local UI bridge on `127.0.0.1:8088`
- opens the current `ui/web/` frontend inside Electron/Chromium
- passes `?desktop=1`, enabling the desktop visual treatment
- shuts down the bridge process when the desktop window quits

No Geant4 run is triggered during startup. Runtime execution still requires an explicit run request.
