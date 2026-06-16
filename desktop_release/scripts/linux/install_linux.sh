#!/usr/bin/env bash
set -euo pipefail

# GenericAgent Desktop Linux setup script
#
# Usage examples:
#   ./install_linux.sh
#   ./install_linux.sh --project-dir /path/to/GenericAgent_Desktop --mode PrepareOnly
#   ./install_linux.sh --mode BridgeOnly
#   ./install_linux.sh --mode AppImageOnly
#
# What this script does:
#   1. Locate GenericAgent project dir, which must contain agentmain.py.
#   2. Locate a supported Python, or create/use .venv.
#   3. Install minimal Python dependencies for desktop bridge.
#   4. Copy mykey_template.py to mykey.py if mykey.py is missing.
#   5. Write ~/.ga_desktop_settings.json for the Tauri shell.
#   6. Optionally start the bridge or packaged AppImage.

PROJECT_DIR=""
PYTHON_PATH=""
MODE="Auto"
NO_VENV=0
SKIP_PIP_INSTALL=0

log_step() { printf '\n==> %s\n' "$*" >&2; }
log_ok() { printf '[OK] %s\n' "$*" >&2; }
log_warn() { printf '[WARN] %s\n' "$*" >&2; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '1,28p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir|-p)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      PROJECT_DIR="$2"; shift 2 ;;
    --python-path)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      PYTHON_PATH="$2"; shift 2 ;;
    --mode|-m)
      [[ $# -ge 2 ]] || fail "Missing value for $1"
      MODE="$2"; shift 2 ;;
    --no-venv)
      NO_VENV=1; shift ;;
    --skip-pip-install)
      SKIP_PIP_INSTALL=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      fail "Unknown argument: $1" ;;
  esac
done

case "$MODE" in
  Auto|PrepareOnly|BridgeOnly|AppImageOnly) ;;
  *) fail "Unsupported --mode '$MODE'. Expected Auto, PrepareOnly, BridgeOnly, or AppImageOnly." ;;
esac

resolve_script_root() {
  local src="${BASH_SOURCE[0]}"
  while [[ -L "$src" ]]; do
    local dir
    dir="$(cd -P "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" && pwd
}

has_agentmain() {
  [[ -f "$1/agentmain.py" ]]
}

abs_path() {
  python3 - "$1" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

find_project_root() {
  local script_root="$1"
  if [[ -n "$PROJECT_DIR" ]]; then
    local p
    p="$(abs_path "$PROJECT_DIR")"
    has_agentmain "$p" || fail "ProjectDir does not contain agentmain.py: $PROJECT_DIR"
    printf '%s\n' "$p"
    return
  fi

  local candidates=(
    "$PWD"
    "$script_root"
    "$script_root/.."
    "$script_root/../.."
    "$script_root/../../.."
    "$HOME/GenericAgent_Desktop"
    "$HOME/GenericAgent_Desktop_test"
  )
  local c p
  for c in "${candidates[@]}"; do
    [[ -e "$c" ]] || continue
    p="$(abs_path "$c")"
    if has_agentmain "$p"; then
      printf '%s\n' "$p"
      return
    fi
  done
  fail "Cannot locate GenericAgent project root. Pass --project-dir <path>."
}

python_supported() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 10) and sys.version_info[:2] < (3, 14) else 1)
PY
}

python_exe() {
  local py="$1"
  "$py" - <<'PY'
import sys
print(sys.executable)
PY
}

find_python() {
  local root="$1"
  if [[ -n "$PYTHON_PATH" ]]; then
    python_supported "$PYTHON_PATH" || fail "Specified Python is not supported. Need Python >=3.10,<3.14: $PYTHON_PATH"
    python_exe "$PYTHON_PATH"
    return
  fi

  local candidates=(
    "$root/.portable/uv-python/bin/python3"
    "$root/.portable/python/bin/python3"
    "$root/python/bin/python3"
    python3.13
    python3.12
    python3.11
    python3.10
    python3
    python
  )
  local py exe
  for py in "${candidates[@]}"; do
    command -v "$py" >/dev/null 2>&1 || [[ -x "$py" ]] || continue
    if python_supported "$py"; then
      exe="$(python_exe "$py")"
      printf '%s\n' "$exe"
      return
    fi
  done
  fail "No supported Python found. Install Python 3.10-3.13, or pass --python-path."
}

ensure_venv() {
  local root="$1"
  local base_python="$2"
  if [[ "$NO_VENV" -eq 1 ]]; then
    printf '%s\n' "$base_python"
    return
  fi

  local venv_dir="$root/.venv"
  local venv_py="$venv_dir/bin/python"
  if [[ ! -x "$venv_py" ]]; then
    log_step "Create virtual environment: $venv_dir"
    "$base_python" -m venv "$venv_dir" || fail "Failed to create venv. On Debian/Ubuntu install python3-venv if missing."
  fi
  python_supported "$venv_py" || fail "Venv Python is invalid: $venv_py"
  python_exe "$venv_py"
}

install_dependencies() {
  local root="$1"
  local py="$2"
  if [[ "$SKIP_PIP_INSTALL" -eq 1 ]]; then
    log_warn "SkipPipInstall is set; dependencies are not installed."
    return
  fi
  log_step "Upgrade pip"
  "$py" -m pip install --upgrade pip || fail "pip upgrade failed."

  log_step "Install GenericAgent minimal package and desktop bridge extras"
  "$py" -m pip install -e "$root" psutil || fail "pip install failed."
}

ensure_mykey() {
  local root="$1"
  local mykey="$root/mykey.py"
  local tpl="$root/mykey_template.py"
  if [[ -f "$mykey" ]]; then
    log_ok "mykey.py exists"
    return
  fi
  if [[ -f "$tpl" ]]; then
    cp "$tpl" "$mykey"
    log_warn "Created mykey.py from mykey_template.py. User still needs to fill API keys."
  else
    log_warn "mykey.py and mykey_template.py are missing. Model config may not work."
  fi
}

write_desktop_settings() {
  local root="$1"
  local py="$2"
  local settings_path="$HOME/.ga_desktop_settings.json"
  local bridge="$root/frontends/desktop_bridge.py"
  [[ -f "$bridge" ]] || fail "desktop_bridge.py not found: $bridge"

  "$py" - "$settings_path" "$py" "$root" "$bridge" <<'PY'
import json, pathlib, sys
settings_path, python_path, project_dir, bridge_script = sys.argv[1:5]
obj = {
    "python_path": python_path,
    "project_dir": project_dir,
    "bridge_script": bridge_script,
}
pathlib.Path(settings_path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
PY
  log_ok "Wrote desktop settings: $settings_path"
}

find_appimage() {
  local root="$1"
  local candidates=(
    "$root/frontends/GenericAgent.AppImage"
    "$root/frontends/desktop/src-tauri/target/release/bundle/appimage/GenericAgent_0.1.0_amd64.AppImage"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return
    fi
  done
  find "$root/frontends/desktop/src-tauri/target/release/bundle/appimage" -maxdepth 1 -type f -name '*.AppImage' 2>/dev/null | head -n 1 || true
}

start_bridge() {
  local root="$1"
  local py="$2"
  local bridge="$root/frontends/desktop_bridge.py"
  [[ -f "$bridge" ]] || fail "desktop_bridge.py not found: $bridge"
  log_step "Start bridge: $bridge"
  log_warn "This keeps running in current console. Browse http://127.0.0.1:14168/status to check."
  export PYTHONPATH="$root:$root/frontends:${PYTHONPATH:-}"
  exec "$py" "$bridge"
}

start_appimage() {
  local root="$1"
  local appimage
  appimage="$(find_appimage "$root")"
  [[ -n "$appimage" ]] || fail "Packaged AppImage not found. Use --mode BridgeOnly or build Tauri first."
  chmod +x "$appimage"
  log_step "Start packaged AppImage"
  nohup "$appimage" >/tmp/genericagent-desktop-appimage.log 2>&1 &
  log_ok "Started: $appimage"
  log_ok "Log: /tmp/genericagent-desktop-appimage.log"
}

log_step "Resolve project root"
SCRIPT_ROOT="$(resolve_script_root)"
ROOT="$(find_project_root "$SCRIPT_ROOT")"
log_ok "Project root: $ROOT"

log_step "Resolve Python"
BASE_PY="$(find_python "$ROOT")"
log_ok "Base Python: $BASE_PY"

PY="$(ensure_venv "$ROOT" "$BASE_PY")"
log_ok "Runtime Python: $PY"

install_dependencies "$ROOT" "$PY"
ensure_mykey "$ROOT"
write_desktop_settings "$ROOT" "$PY"

case "$MODE" in
  PrepareOnly)
    log_ok "Preparation finished. No app started because --mode PrepareOnly was used."
    ;;
  BridgeOnly)
    start_bridge "$ROOT" "$PY"
    ;;
  AppImageOnly)
    start_appimage "$ROOT"
    ;;
  Auto)
    if [[ -n "$(find_appimage "$ROOT")" ]]; then
      start_appimage "$ROOT"
    else
      start_bridge "$ROOT" "$PY"
    fi
    ;;
esac
