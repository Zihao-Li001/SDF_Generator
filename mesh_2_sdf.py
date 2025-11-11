import trimesh
from mesh_to_sdf import mesh_to_voxels
import time, glob, os, multiprocessing, io, zlib
import numpy as np

# Set input and output directories
INPUT_DIR = "./dataset/stl/"
OUTPUT_DIR = "./dataset/sdf/"

# Set SDF parameters
VOXEL_RESOLUTION = 64
PAD_VOXELS = True

# CPU core control
# Set to None to use all available cores
# Set to a sepcific integer to limie the core count
NUM_CORES = None

COMPRESSION_LEVEL = 6

# Log File
LOG_FILE = "processing_log.txt"


def process_file(stl_path):
    """
    Worker function to process a single STL file.
    Loads the mesh, computes the SDF, and save it as a .npy file.
    """
    # --- Logging ---
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"STARTING: {stl_path}\n")
    except Exception:
        pass
    # --- End Log ---

    try:
        # Define output path
        base_name = os.path.basename(stl_path)
        file_name = os.path.splitext(base_name)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_name}.npy.z")

        # Optional: Skip if the file has already been processed
        if os.path.exists(output_path):
            # --- Log ---
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as log:
                    log.write(f"SKIPPED: {stl_path}\n")
            except Exception:
                pass
            # --- End Log ---

            return (stl_path, "skipped")

        # Load the mesh using trimesh
        mesh = trimesh.load(stl_path)

        # Convert mesh to SDF voxels
        voxels = mesh_to_voxels(
            mesh,
            voxel_resolution=VOXEL_RESOLUTION,
            pad=PAD_VOXELS,
            sign_method="normal",
        )

        # --- Save as compressed .npy.z ---
        # Save the array to an in-memory bytes buffer
        buffer = io.BytesIO()
        np.save(buffer, voxels)
        buffer.seek(0)

        # Read the .npy data from the buffer and compress it
        npy_data = buffer.read()
        compressed_data = zlib.compress(npy_data, level=COMPRESSION_LEVEL)

        # Write the compressed bytes to the final .npy.z file
        with open(output_path, "wb") as f:
            f.write(compressed_data)
        # --- End of save logic ---

        # --- Log ---
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"SUCCESS: {stl_path}\n")
        except Exception:
            pass
        # --- End Log ---

        return (stl_path, "success")

    except Exception as e:
        # --- Log ---
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"FAIL: {stl_path}-{e}\n")
        except Exception:
            pass
        # --- End Log ---

        return (stl_path, f"failed: {e}")


def main():
    """
    Main function to find files and manage the parallel processing pool.
    """
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    print(f"Log will be written to {LOG_FILE}")
    start_time = time.time()

    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all .stl files in the input directory
    stl_files = glob.glob(os.path.join(INPUT_DIR, "*.stl"))

    if not stl_files:
        print(f"Error: No .stl files found in '{INPUT_DIR}'")
        return

    print(f"Found {len(stl_files)} STL files to process.")
    print(f"Output will be saved as .npy.z files in '{OUTPUT_DIR}'.")
    print(f"Using {VOXEL_RESOLUTION}^3 resolution.")

    # --- CPU Core Logic ---
    available_cores = multiprocessing.cpu_count()
    if NUM_CORES is None:
        num_processes = available_cores
        print(
            f"Starting parallel processing with all {num_processes} avaliable_cores..."
        )
    else:
        num_processes = NUM_CORES
        if num_processes > available_cores:
            print(
                f"Warning: Requested {num_processes} cores, but only {available_cores}"
            )
            num_processes = available_cores
        else:
            print(f"Starting parallel processing with {num_processes} cores...")
    # --- End CPU Core Logic ---

    with multiprocessing.Pool(processes=num_processes) as pool:
        print("Processing... Please wait.")
        results = list(pool.imap_unordered(process_file, stl_files))

    # --- Print Summary ---
    end_time = time.time()
    success_count = 0
    skipped_count = 0
    failed_files = []

    for path, status in results:
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_files.append((path, status))

    print("\n--- Processing Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for path, error in failed_files:
            print(f"  - {os.path.basename(path)}: {error}")


if __name__ == "__main__":
    main()
