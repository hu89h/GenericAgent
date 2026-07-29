<#
Prepare the self-contained Python used by a GenericAgent source checkout.

This is deliberately separate from any Conda environment. It downloads the
same Python-build-standalone distribution used by the desktop release workflow,
creates .portable\uv-python, fills an exact offline wheelhouse when needed, and
installs only the desktop/Agent runtime dependencies into that interpreter.

Examples:
  powershell -ExecutionPolicy Bypass -File .\prepare_project_runtime.ps1
  powershell -ExecutionPolicy Bypass -File .\prepare_project_runtime.ps1 -PythonArchive C:\cache\python.tar.gz
  powershell -ExecutionPolicy Bypass -File .\prepare_project_runtime.ps1 -PipIndexUrl https://pypi.org/simple
#>

param(
    [string]$ProjectDir = "",
    [string]$RuntimeDir = "",
    [string]$WheelDir = "",
    [string]$PythonArchive = "",
    [string]$PipIndexUrl = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"

function Fail([string]$message) { throw "[ERROR] $message" }

function Resolve-ProjectRoot {
    if ($ProjectDir) {
        $resolved = Resolve-Path -LiteralPath $ProjectDir -ErrorAction Stop
        if (Test-Path -LiteralPath (Join-Path $resolved.Path "agentmain.py")) { return $resolved.Path }
        Fail "ProjectDir does not contain agentmain.py: $ProjectDir"
    }
    $candidate = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..\")).Path
    if (Test-Path -LiteralPath (Join-Path $candidate "agentmain.py")) { return $candidate }
    Fail "Cannot locate GenericAgent project root. Pass -ProjectDir <path>."
}

function Assert-Within([string]$root, [string]$target) {
    $r = (Resolve-Path -LiteralPath $root).Path.TrimEnd('\') + '\'
    $t = [IO.Path]::GetFullPath($target).TrimEnd('\')
    if (-not $t.StartsWith($r, [StringComparison]::OrdinalIgnoreCase)) {
        Fail "Refusing to modify a path outside the project: $t"
    }
}

function Get-PythonAsset {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" -Headers @{ Accept = "application/vnd.github+json" }
    $asset = $release.assets | Where-Object {
        $_.name -match '^cpython-3\.12\.[0-9]+\+.*-x86_64-pc-windows-msvc-install_only\.tar\.gz$'
    } | Select-Object -First 1
    if (-not $asset) { Fail "No Python 3.12 Windows standalone asset found in the latest release." }
    return $asset
}

function Invoke-Pip([string]$python, [string[]]$arguments) {
    & $python -m pip @arguments
    if ($LASTEXITCODE -ne 0) { Fail "Python command failed: -m pip $($arguments -join ' ')" }
}

$root = Resolve-ProjectRoot
$runtime = if ($RuntimeDir) { [IO.Path]::GetFullPath($RuntimeDir) } else { Join-Path $root ".portable\uv-python" }
$wheels = if ($WheelDir) { [IO.Path]::GetFullPath($WheelDir) } else { Join-Path $root "artifacts\offline-wheels\windows-py312" }
Assert-Within $root $runtime
Assert-Within $root $wheels

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("genericagent-python-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
try {
    $python = Join-Path $runtime "python.exe"
    if (Test-Path -LiteralPath $python) {
        if (-not (Test-Path -LiteralPath (Join-Path $runtime "python312.dll"))) {
            Fail "The existing project Python runtime is incomplete: $runtime"
        }
        Write-Host "Reusing project Python: $python"
    } else {
        $archive = $PythonArchive
        if ($archive) {
            $archive = (Resolve-Path -LiteralPath $archive -ErrorAction Stop).Path
        } else {
            $asset = Get-PythonAsset
            $archive = Join-Path $tempRoot $asset.name
            Write-Host "Downloading $($asset.name) ..."
            Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archive
        }

        $extract = Join-Path $tempRoot "extract"
        New-Item -ItemType Directory -Force -Path $extract | Out-Null
        if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) { Fail "tar.exe is required to extract Python-build-standalone." }
        & tar.exe -xzf $archive -C $extract
        if ($LASTEXITCODE -ne 0) { Fail "Failed to extract Python archive: $archive" }

        $pythonFile = Get-ChildItem -LiteralPath $extract -Recurse -Filter python.exe -File | Select-Object -First 1
        if (-not $pythonFile) { Fail "The Python archive does not contain python.exe." }
        $pythonRoot = $pythonFile.Directory.FullName
        if (-not (Test-Path -LiteralPath (Join-Path $pythonRoot "python312.dll"))) {
            Fail "The extracted Python runtime is incomplete: $pythonRoot"
        }

        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runtime) | Out-Null
        Assert-Within $root $runtime
        Move-Item -LiteralPath $pythonRoot -Destination $runtime
        $python = Join-Path $runtime "python.exe"
    }

    $requirements = @(
        "requests>=2.28", "beautifulsoup4>=4.12", "bottle>=0.12",
        "simple-websocket-server>=0.4", "aiohttp>=3.9", "psutil",
        "zvec>=0.6,<0.7", "pypdf>=5.0", "Pillow>=9.0", "PySocks>=1.7",
        "fastapi", "uvicorn", "websockets", "pydantic", "setuptools", "wheel"
    )
    New-Item -ItemType Directory -Force -Path $wheels | Out-Null
    if (-not (Get-ChildItem -LiteralPath $wheels -Filter *.whl -File -ErrorAction SilentlyContinue)) {
        Write-Host "Downloading exact offline dependency wheelhouse ..."
        $indexArgs = if ($PipIndexUrl) { @("-i", $PipIndexUrl) } else { @() }
        Invoke-Pip $python (@("download", "--dest", $wheels) + $indexArgs + $requirements)
    }

    $installer = Join-Path $root "frontends\desktop\packaging\scripts\windows\install_windows.ps1"
    $installerArgs = @(
        "-ProjectDir", $root, "-PythonPath", $python, "-WheelDir", $wheels,
        "-Mode", "PrepareOnly", "-NoVenv"
    )
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer @installerArgs
    if ($LASTEXITCODE -ne 0) { Fail "Offline dependency installation failed." }

    & $python -c "import aiohttp, fastapi, PIL, pydantic, requests, socks, uvicorn, websockets, zvec; print('project runtime smoke test: ok')"
    if ($LASTEXITCODE -ne 0) { Fail "Project runtime dependency smoke test failed." }
    Write-Host "Project runtime ready: $python"
    Write-Host "Wheelhouse: $wheels"
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
