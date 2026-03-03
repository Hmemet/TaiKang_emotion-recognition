param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

function Stop-ProcessTree {
    param([int]$ProcessId)

    try {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
        foreach ($child in $children) {
            Stop-ProcessTree -ProcessId $child.ProcessId
        }

        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
    }
}

function Resolve-BrowserPath {
    $candidates = @(
        (Get-Command msedge -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        (Get-Command chrome -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }

    if ($candidates.Count -gt 0) {
        return $candidates[0]
    }

    return $null
}

function Resolve-PythonPath {
    param(
        [string]$ScriptDir,
        [string]$Preferred
    )

    $candidates = @()

    if ($Preferred) {
        $candidates += $Preferred
    }

    if ($env:BACKEND_PYTHON) {
        $candidates += $env:BACKEND_PYTHON
    }

    $candidates += "F:\miniconda\envs\transformers\python.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    return $null
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = Split-Path -Parent $scriptDir
$frontendPath = Join-Path $workspaceDir "frontend\index.html"

if (-not (Test-Path $frontendPath)) {
    throw "Frontend page not found: $frontendPath"
}

$resolvedPython = Resolve-PythonPath -ScriptDir $scriptDir -Preferred $PythonPath
if (-not $resolvedPython) {
    throw "Python not found. Install Python and add it to PATH, or run from an activated environment."
}

Write-Host "[1/4] Starting backend service..."
$logDir = Join-Path $scriptDir "logs"
New-Item -Path $logDir -ItemType Directory -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logDir ("backend_stdout_" + $stamp + ".log")
$stderrLog = Join-Path $logDir ("backend_stderr_" + $stamp + ".log")

Write-Host "Using Python: $resolvedPython"
$backendProcess = Start-Process -FilePath $resolvedPython -ArgumentList "app.py" -WorkingDirectory $scriptDir -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

$healthUrl = "http://$BackendHost`:$BackendPort/api/health"
$maxWaitSeconds = 90
$ready = $false

Write-Host "[2/4] Waiting for backend health check: $healthUrl"
for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
    Start-Sleep -Seconds 1

    $running = Get-Process -Id $backendProcess.Id -ErrorAction SilentlyContinue
    if (-not $running) {
        $stderrTail = ""
        if (Test-Path $stderrLog) {
            $stderrTail = (Get-Content -Path $stderrLog -Tail 20 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }

        if ([string]::IsNullOrWhiteSpace($stderrTail)) {
            $stderrTail = "(no stderr output captured)"
        }

        throw "Backend process exited unexpectedly.`nPython: $resolvedPython`nStdout: $stdoutLog`nStderr: $stderrLog`nLast stderr lines:`n$stderrTail"
    }

    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
            $ready = $true
            break
        }
    }
    catch {
    }
}

if (-not $ready) {
    Stop-ProcessTree -ProcessId $backendProcess.Id
    throw "Backend was not ready within $maxWaitSeconds seconds. Process stopped.`nStdout: $stdoutLog`nStderr: $stderrLog"
}

$frontendUrl = ([System.Uri]$frontendPath).AbsoluteUri
$browserPath = Resolve-BrowserPath
$tempProfile = Join-Path $env:TEMP ("emotion_ui_profile_" + [Guid]::NewGuid().ToString("N"))

New-Item -Path $tempProfile -ItemType Directory -Force | Out-Null

Write-Host "[3/4] Opening frontend page: $frontendUrl"

if ($browserPath) {
    $browserArgs = @(
        "--new-window",
        "--app=$frontendUrl",
        "--user-data-dir=$tempProfile",
        "--no-first-run",
        "--disable-session-crashed-bubble"
    )

    try {
        $browserProcess = Start-Process -FilePath $browserPath -ArgumentList $browserArgs -PassThru
        Write-Host "[4/4] Close the browser app window to stop backend automatically..."
        Wait-Process -Id $browserProcess.Id
    }
    catch {
        Write-Host "Browser app-window launch failed. Falling back to default browser."
        Start-Process -FilePath $frontendPath
        Write-Host "Press Enter in this window when done to stop backend."
        [void](Read-Host)
    }
}
else {
    Write-Host "Edge/Chrome not found. Opening in default browser."
    Start-Process -FilePath $frontendPath
    Write-Host "Press Enter in this window when done to stop backend."
    [void](Read-Host)
}

Write-Host "Stopping backend..."
Stop-ProcessTree -ProcessId $backendProcess.Id

if (Test-Path $tempProfile) {
    Remove-Item -Path $tempProfile -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Done."