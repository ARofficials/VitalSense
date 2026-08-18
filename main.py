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
        print("[INFO] Checking dependencies...")
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req_file, "--quiet"])
    else:
        print("[WARNING] requirements.txt not found! Skipping install.")

    # 3. Relaunch Application
    print("[INFO] Launching Application...\n")
    args = [venv_python, __file__] + sys.argv[1:]
    
    if platform.system() == "Windows":
        subprocess.call(args)
    else:
        os.execv(venv_python, args)

def main():
    # If we are NOT in a venv, we run the bootstrap process
    # (Skip this check if running via PyInstaller frozen app)
    if not getattr(sys, 'frozen', False) and not is_venv():
        bootstrap()
        sys.exit()

    print("[INFO] Starting Launcher...")
    
    # --- LAUNCHER INTEGRATION ---
    # Instead of starting gui_app directly, we start the Launcher
    try:
        import launcher
        app = launcher.LauncherApp()
        app.mainloop()
    except ImportError as e:
        print(f"[ERROR] Could not import launcher: {e}")
        print("Ensure 'launcher.py' is in the same directory.")
    except Exception as e:
        print(f"[CRITICAL] App crashed: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()