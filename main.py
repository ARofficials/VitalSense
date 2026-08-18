import os
import sys
import subprocess
import platform

def is_venv():
    """Checks if running inside a virtual environment"""
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def get_venv_python():
    """Returns the path to the python executable within the venv"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")

def bootstrap():
    """
    Bootstrap process:
    1. Checks if 'venv' exists. If not, creates it.
    2. Installs requirements.txt.
    3. Relaunches the script inside the venv.
    """
    venv_dir = "venv"
    venv_python = get_venv_python()
    req_file = "requirements.txt"

    print("==================================================")
    print("  🚀 VITALSENSE BOOTSTRAPPER")
    print("==================================================")

    # 1. Create Venv if missing
    if not os.path.exists(venv_dir):
        print(f"[INFO] Creating virtual environment in '{venv_dir}'...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    
    # 2. Install Requirements
    if os.path.exists(req_file):
        print("[INFO] Checking/Installing dependencies...")
        # We use the venv python to install pip packages
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req_file, "--quiet"])
    else:
        print("[WARNING] requirements.txt not found! Skipping install.")

    # 3. Relaunch Application
    print("[INFO] Launching Application...\n")
    
    # Pass all arguments (like -f video.mp4) to the inner script
    args = [venv_python, __file__] + sys.argv[1:]
    
    # On Windows, we need to use subprocess.call to avoid permission locking
    # On Unix, we can use execv to replace the process
    if platform.system() == "Windows":
        subprocess.call(args)
    else:
        os.execv(venv_python, args)

def main():
    # If we are NOT in a venv, we run the bootstrap process
    if not is_venv():
        bootstrap()
        sys.exit()

    # --- ACTUAL APPLICATION LOGIC STARTS HERE ---
    # Only imports complex libraries AFTER bootstrap ensures they exist
    import argparse
    from vital_monitor import ProductionVitalMonitor

    parser = argparse.ArgumentParser(description="Production Vital Signs Monitor")
    parser.add_argument("-f", "--file", type=str, help="Path to video file", default=0)
    args = parser.parse_args()
    
    source = int(args.file) if str(args.file).isdigit() else args.file
    
    app = ProductionVitalMonitor()
    app.run(source=source)

if __name__ == "__main__":
    main()