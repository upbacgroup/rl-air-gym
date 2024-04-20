import os
import zipfile

def unzip_directories(directories):
    current_directory = os.getcwd()

    for directory in directories:
        directory_path = os.path.join(current_directory, directory)

        # Get a list of all files in the current directory
        files = os.listdir(directory_path)

        # Iterate through each file
        for file in files:
            # Check if the file is a zip file
            if file.endswith(".zip"):
                file_path = os.path.join(directory_path, file)

                # Open the zip file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all contents to the output directory
                    zip_ref.extractall(directory_path)

                print(f"Unzipped {file} to {directory_path}")

                # Remove the original zip file
                os.remove(file_path)
                print(f"Removed {file}")

if __name__ == "__main__":
    # List of directories
    directories = [
        "m500_urdf/meshes/dae/",
        # "description/m500_jetson_zed_urdf/meshes/dae/",
        # "description/m500_jetson_rplidar_urdf/meshes/dae/",
        # "description/m500_jetson_zed_rplidar_urdf/meshes/dae/",
    ]

    unzip_directories(directories)
