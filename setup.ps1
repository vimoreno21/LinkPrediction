# Ensure Python 3.13 is used (assuming it's installed and in PATH)
$pythonCmd = "python3.13"

# Check if specified Python version exists
try {
    & $pythonCmd --version
} catch {
    Write-Error "Python 3.13 not found. Make sure it's installed and in your PATH."
    exit 1
}

# Create virtual environment with specific Python version
& $pythonCmd -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Upgrade pip using the recommended method
& $pythonCmd -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
