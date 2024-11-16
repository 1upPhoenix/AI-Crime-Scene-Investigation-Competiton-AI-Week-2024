import sys
import subprocess
import platform

libraries = [
    'pandas',
    'joblib',
    'imbalanced-learn',  
    'scikit-learn',  
    'matplotlib',  
    'numpy'
]

def install_libraries():
    try:
        for lib in libraries:
            print(f"Installing {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print("All libraries installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing libraries: {e}")

def install_for_windows():
    current_os = platform.system().lower()
    
    if current_os == 'windows':
        install_libraries()
    else:
        print("This script is intended for Windows only.")

if __name__ == "__main__":
    install_for_windows()
