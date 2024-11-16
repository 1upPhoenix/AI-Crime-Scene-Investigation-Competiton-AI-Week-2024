import subprocess
import sys
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
            subprocess.check_call([sys.executable, "-m", "pip3", "install", lib])
        print("All libraries installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing libraries: {e}")


def install_for_mac():
    current_os = platform.system().lower()
    
    if current_os == 'darwin':
        install_libraries()
    else:
        print("This script is intended for macOS only.")


if __name__ == "__main__":
    install_for_mac()
