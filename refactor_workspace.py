import argparse
import shutil
import re
import subprocess
from pathlib import Path

# Define base path
BASE_DIR = Path(__file__).resolve().parent

# Define directories and file moves
MIGRATIONS = [
    # Create target directories
    {"type": "mkdir", "dest": "src/subsumption/layers"},
    {"type": "mkdir", "dest": "src/hardware/sensors"},
    {"type": "mkdir", "dest": "src/hardware/actuators"},
    {"type": "mkdir", "dest": "src/arm"},
    {"type": "mkdir", "dest": "src/visual_servoing"},
    {"type": "mkdir", "dest": "firmware/esp32"},
    {"type": "mkdir", "dest": "legacy/FYProbot"},
    
    # Move firmware components out of legacy targets first
    {"type": "move", "src": "FYProbot/esp32", "dest": "firmware/esp32"},
    
    # Move legacy modules
    {"type": "move", "src": "FYProbot", "dest": "legacy/FYProbot"},
    
    # Move driver files
    {"type": "move", "src": "src/drivers/pca9685_driver.py", "dest": "src/hardware/actuators/pca9685_driver.py"},
    {"type": "move", "src": "src/drivers/serial_interface.py", "dest": "src/hardware/sensors/serial_interface.py"},
    {"type": "move", "src": "src/drivers", "dest": "legacy/old_drivers"}, # Catch-all wrapper for remaining drivers
]

# Define files to touch (scaffold)
SCAFFOLDS = [
    "src/subsumption/arbitrator.py",
    "src/subsumption/layers/__init__.py",
    "src/subsumption/layers/layer0_idle.py",
    "src/subsumption/layers/layer1_scan.py",
    "src/subsumption/layers/layer2_approach.py",
    "src/subsumption/layers/layer3_collect.py",
    "src/subsumption/layers/layer4_intercept.py",
    "src/subsumption/layers/layer5_emergency.py",
    "src/subsumption/layers/layer6_low_power.py",
    "src/hardware/__init__.py",
    "src/hardware/sensors/__init__.py",
    "src/hardware/actuators/__init__.py",
]

# Define regex patterns for updating imports
IMPORT_REPLACEMENTS = [
    (re.compile(r'from\s+src\.drivers(\.[\w_]+)?\s+import'), r'from src.hardware.actuators\1 import'),
    (re.compile(r'import\s+src\.drivers'), r'import src.hardware.actuators')
]

def parse_args():
    parser = argparse.ArgumentParser(description="Refactor Workspace to Subsumption Architecture")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")
    return parser.parse_args()

def execute_migrations(dry_run):
    print("=== Start Workspace Migration ===\n")
    for task in MIGRATIONS:
        if task["type"] == "mkdir":
            dest = BASE_DIR / task["dest"]
            if not dest.exists():
                print(f"[MKDIR] {'(DRY RUN)' if dry_run else ''} Creating directory -> {dest.relative_to(BASE_DIR)}")
                if not dry_run:
                    dest.mkdir(parents=True, exist_ok=True)
            
        elif task["type"] == "move":
            src = BASE_DIR / task["src"]
            dest = BASE_DIR / task["dest"]
            
            if src.exists() and not dest.exists():
                print(f"[MOVE ] {'(DRY RUN)' if dry_run else ''} Moving: {src.relative_to(BASE_DIR)} -> {dest.relative_to(BASE_DIR)}")
                if not dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest))
            elif not src.exists():
                print(f"[SKIP ] Source does not exist: {src.relative_to(BASE_DIR)}")
            else:
                print(f"[SKIP ] Destination already exists: {dest.relative_to(BASE_DIR)}")

def execute_scaffolding(dry_run):
    print("\n=== Scaffolding Files ===")
    for path_str in SCAFFOLDS:
        target = BASE_DIR / path_str
        if not target.exists():
            print(f"[TOUCH] {'(DRY RUN)' if dry_run else ''} Creating empty file -> {target.relative_to(BASE_DIR)}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()

def replace_imports(dry_run):
    print("\n=== Scanning & Updating Imports in src/ ===")
    src_dir = BASE_DIR / "src"
    if not src_dir.exists():
        print("No src/ directory found.")
        return

    py_files = list(src_dir.rglob("*.py"))
    
    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception as e:
            continue

        new_content = content
        modified = False
        
        for pattern, replacement in IMPORT_REPLACEMENTS:
            if pattern.search(new_content):
                new_content = pattern.sub(replacement, new_content)
                modified = True
                
        if modified:
            print(f"[SED  ] {'(DRY RUN)' if dry_run else ''} Updated imports in -> {py_file.relative_to(BASE_DIR)}")
            if not dry_run:
                py_file.write_text(new_content, encoding="utf-8")

def run_pytest(dry_run):
    print("\n=== Validating with Pytest ===")
    if dry_run:
        print("[TEST ] (DRY RUN) Would execute: python -m pytest --collect-only")
    else:
        print("[TEST ] Executing: python -m pytest --collect-only")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--collect-only"],
                cwd=BASE_DIR,
                text=True,
                capture_output=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print("Pytest collection warnings/errors detected.")
                print(result.stderr)
        except Exception as e:
            print(f"Failed to run pytest (is pytest installed?): {e}")

if __name__ == "__main__":
    args = parse_args()
    execute_migrations(args.dry_run)
    execute_scaffolding(args.dry_run)
    replace_imports(args.dry_run)
    run_pytest(args.dry_run)
    print("\n=== Refactoring Complete ===")