# Project Setup

This project uses Python 3 and all dependencies are managed via a `requirements.txt` file.

## Automated Setup Scripts

To streamline environment creation and installation, two helper scripts are provided:

- **setup.sh** (macOS/Linux)
- **setup.ps1** (Windows PowerShell)

### setup.sh
```bash
#!/usr/bin/env bash
set -e
# Create virtual environment
python3 -m venv .venv
# Activate
source .venv/bin/activate
# Upgrade pip
pip install --upgrade pip
# Install dependencies
pip install -r requirements.txt
```

### setup.ps1
```powershell
# Create virtual environment
python -m venv .venv
# Activate
.\.venv\Scripts\Activate.ps1
# Upgrade pip
pip install --upgrade pip
# Install dependencies
pip install -r requirements.txt
```

## Manual Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Run the automated script**
   - On macOS/Linux:
     ```bash
     chmod +x setup.sh
     ./setup.sh
     ```
   - On Windows PowerShell:
     ```powershell
     ./setup.ps1
     ```

3. **Verify installation**
   ```bash
   python -c "import networkx, matplotlib, numpy, pandas, tqdm, seaborn, sklearn; print('All dependencies installed')"
   ```

4. **Run your script**
   ```bash
   python your_script.py
   ```

## Managing Dependencies

- **Generate `requirements.txt`**  
  ```bash
  pip freeze > requirements.txt
  ```

- **Add new packages**  
  1. Activate your environment:
     ```bash
     source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
     ```
  2. Install the package:
     ```bash
     pip install <package-name>
     ```
  3. Update `requirements.txt`:
     ```bash
     pip freeze > requirements.txt
     ```

