import trimesh
import time, glob, os, multiprocessing, io, zlib
import numpy as np
from itertools import islice
from representation.voxel_generator import process_voxel

# Set input and output directories
INPUT_DIR = "./dataset/stl/"
OUTPUT_DIR = "./dataset/voxel/"

# Set SDF parameters
VOXEL_RESOLUTION = 64
PAD_VOXELS = False

# --- CPU core control ---
# Set to None to use all available cores
# Set to a sepcific integer to limie the core count
NUM_CORES = 12
# --- End Control ---

# Log File
LOG_FILE = "processing_voxel_log.txt"

BATCH_SIZE = 200
CHUNKSIZE = 1


def process_file(stl_path):
    """
    Worker function to process a single STL file.
    Loads the mesh, computes the binary occupancy voxel,
    and save it as a .npy file.
    """
    # Define output path
    base_name = os.path.basename(stl_path)
    file_name = os.path.splitext(base_name)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{file_name}.npy")

    # --- Log helper ---
    def write_log(msg: str):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(msg + "\n")
        except Exception:
            pass

    # --- Start log ---
    write_log(f"START: {stl_path}")

    # --- Skip case ---
    if os.path.exists(output_path):
        write_log(f"SKIP: {stl_path} (already processed)")
        return

    try:
        # --- Load mesh (using trimesh) ---
        mesh = trimesh.load(stl_path)

        # --- Binary Occupancy Voxel generate ---
        process_voxel(stl_path, output_path, resolution=VOXEL_RESOLUTION)

        # Explicitly delete mesh, release memory
        del mesh

        # --- Log: success ---
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"SUCCESS: {stl_path}\n")
        except Exception:
            pass
        # --- End Log ---
        return

    except Exception as e:
        # --- Log ---
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"FAIL: {stl_path}-{e}\n")
        except Exception:
            pass
        # --- End Log ---
        return


def chunk_list(iterable, size):
    """
    Divide the large list into small batchs.
    To prevent excessive task backlog in the pool.
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


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
    print(f"Output will be saved as .npy files in '{OUTPUT_DIR}'.")

    # --- CPU Core Logic ---
    available_cores = multiprocessing.cpu_count()
    if NUM_CORES is None:
        num_processes = available_cores - 1
    else:
        num_processes = min(NUM_CORES, available_cores)
    print(f"Starting parallel processing with {num_processes} cores...")
    # --- End CPU Core Logic ---

    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_id, subset in enumerate(chunk_list(stl_files, BATCH_SIZE), start=1):
            print(f"\n Process batch {batch_id}: {len(subset)} file")
            pool.map(process_file, subset)

            print(f"Batch {batch_id} complete")
    print("\nAll batches finished. Generating summary from log file...")

    end_time = time.time()
    success_count = 0
    skipped_count = 0
    failed_lines = []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("SUCCESS:"):
                    success_count += 1
                elif line.startswith("SKIPPED:"):
                    skipped_count += 1
                elif line.startswith("FAIL:"):
                    failed_lines.append(line)

    except FileNotFoundError:
        print(f"Error: Log file '{LOG_FILE}' not found. Cannot generate summary")
        return

    # --- Print Summary ---
    print("\n--- Processing Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {len(failed_lines)}")

    if failed_lines:
        print("\nFailed files:")
        for line in failed_lines:
            print(f"  - {line[len('FAIL:'):].strip()}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
