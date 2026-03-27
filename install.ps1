#Requires -Version 5
<#
.SYNOPSIS
  MYCONEX Windows installer - delegates to install.sh via WSL, or runs natively.
.EXAMPLE
  .\install.ps1 --role hub
  .\install.ps1 --unattended answers.yaml
#>
param(
    [string]$Role         = "",
    [string]$Unattended   = "",
    [switch]$SaveAnswers,
    [string]$AnswersOut   = ".\myconex-answers.yaml",
    [switch]$NoTui,
    [switch]$Reinstall,
    [switch]$SkipVerify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param([string]$Msg); Write-Host "[install] $Msg" -ForegroundColor Cyan }

# --- Build arg list to pass through ---
$passArgs = @()
if ($Role)        { $passArgs += "--role", $Role }
if ($Unattended)  { $passArgs += "--unattended", $Unattended }
if ($SaveAnswers) { $passArgs += "--save-answers" }
if ($AnswersOut)  { $passArgs += "--answers-out", $AnswersOut }
if ($NoTui)       { $passArgs += "--no-tui" }
if ($Reinstall)   { $passArgs += "--reinstall" }
if ($SkipVerify)  { $passArgs += "--skip-verify" }

# --- WSL path ---
function Get-WslStatus {
    try {
        $out = wsl --status 2>&1 | Out-String
        if ($out -match "Default Version: 1") { return "wsl1" }
        if ($out -match "WSL" -or (Get-Command wsl -ErrorAction SilentlyContinue)) { return "wsl2" }
    } catch {}
    return "none"
}

$wslStatus = Get-WslStatus

if ($wslStatus -eq "wsl1") {
    Write-Step "WSL1 detected. WSL2 is required. Upgrading..."
    wsl --set-default-version 2
    Write-Step "Please restart your terminal and re-run this installer."
    exit 0
}

if ($wslStatus -eq "wsl2") {
    Write-Step "WSL2 detected - running Linux installer inside WSL"
    # Convert Windows path to WSL path
    $repoRoot = (wsl wslpath "'$PSScriptRoot'").Trim()
    $installSh = "${repoRoot}/install.sh"
    $wslArgs = @("bash", $installSh) + $passArgs
    wsl @wslArgs
    exit $LASTEXITCODE
}

# --- Native Windows path (no WSL) ---
Write-Step "No WSL detected - running native Windows install"

# 1. Chocolatey
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Step "Installing Chocolatey"
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 2. Core deps
foreach ($pkg in @("python", "git", "docker-desktop")) {
    if (-not (Get-Command $pkg.Split('-')[0] -ErrorAction SilentlyContinue)) {
        Write-Step "Installing $pkg"
        choco install $pkg -y --no-progress
    }
}

# 3. Wait for Docker Desktop engine
Write-Step "Waiting for Docker Desktop engine..."
$retries = 30
while ($retries -gt 0) {
    if (docker info 2>$null) { break }
    Start-Sleep 5; $retries--
}
if ($retries -eq 0) { Write-Error "Docker Desktop did not start in time."; exit 1 }

# 4. Pip deps
Write-Step "Installing Python dependencies"
python -m pip install --upgrade -r "$PSScriptRoot\requirements.txt"

# 5. Task Scheduler entry for auto-start
$taskName = "MYCONEX-$($Role.ToUpper() -replace '-','')"
Write-Step "Registering Task Scheduler entry: $taskName"
$action  = New-ScheduledTaskAction -Execute "python" `
           -Argument "-m myconex --mode $(if ($Role -eq 'hub') {'api'} else {'worker'})" `
           -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Highest -Force | Out-Null

Write-Step "Windows install complete. Start manually: python -m myconex --mode $(if ($Role -eq 'hub') {'api'} else {'worker'})"
