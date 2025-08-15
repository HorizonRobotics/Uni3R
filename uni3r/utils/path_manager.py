import sys
import os
import os.path as path

# Get the normalized path of the current file's directory
HERE_PATH = path.normpath(path.dirname(__file__))

# Get the project root path (two levels up from current directory)
PROJECT_ROOT = path.normpath(path.join(HERE_PATH, '../..'))

# Path to submodules directory
SUBMODULES_PATH = path.join(PROJECT_ROOT, 'submodules')

def add_submodule_to_path(submodule_name):
    """
    Add a specified submodule to the Python path
    
    Args:
        submodule_name (str): Name of the submodule
    
    Raises:
        ImportError: If the submodule directory doesn't exist
    """
    submodule_path = path.join(SUBMODULES_PATH, submodule_name)
    if path.isdir(submodule_path):
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
    else:
        raise ImportError(
            f"Submodule {submodule_name} is not initialized, could not find: {submodule_path}\n"
            "Did you forget to run 'git submodule update --init --recursive' ?"
        )

def init_all_submodules():
    """
    Initialize paths for all submodules
    """
    if not path.isdir(SUBMODULES_PATH):
        raise ImportError(
            f"Submodules directory not found at: {SUBMODULES_PATH}\n"
            "Did you forget to run 'git submodule update --init --recursive' ?"
        )
    
    # Get all subdirectories in the submodules directory
    submodules = [d for d in os.listdir(SUBMODULES_PATH) 
                  if path.isdir(path.join(SUBMODULES_PATH, d))]
    
    for submodule in submodules:
        add_submodule_to_path(submodule) 
    
    # Add the project root to the Python path
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
