import os
import glob

def find_arcgis_pro_python():
    # Possible installation paths for ArcGIS Pro
    possible_installation_paths = [
        "C:\\Program Files\\ArcGIS\\Pro",
        "C:\\Program Files (x86)\\ArcGIS\\Pro"  # For 32-bit installations on 64-bit Windows
        # Add more paths as needed
    ]
    
    # Define the pattern to search for python.exe
    pattern = os.path.join("bin", "Python", "envs", "*", "python.exe")
    
    # Search for python.exe in each possible installation path
    for path in possible_installation_paths:
        python_exe_paths = glob.glob(os.path.join(path, pattern))
        if python_exe_paths:
            # Assuming there's only one Python interpreter, return the first one found
            return python_exe_paths[0]
    
    # Return None if python.exe is not found in any of the paths
    return None

# Example usage:
python_exe_path = find_arcgis_pro_python()
if python_exe_path:
    print("Path to ArcGIS Pro Python interpreter:", python_exe_path)
else:
    print("ArcGIS Pro Python interpreter not found.")
