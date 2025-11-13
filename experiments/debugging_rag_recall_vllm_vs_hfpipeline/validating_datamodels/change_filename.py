import os
import sys

def rename_feather_files():
    """
    Scans a specified folder and renames all .feather files
    based on the logic defined within the script.
    """
    
    # --- 1. SET YOUR FOLDER PATH ---
    # IMPORTANT: Replace this with the full, absolute path to your folder.
    #
    # Windows Example: r"C:\Users\YourUser\Documents\MyData"
    # macOS Example:   r"/Users/youruser/Documents/MyData"
    # Linux Example:   r"/home/youruser/Documents/MyData"
    #
    # Use a raw string (r"...") especially on Windows to handle backslashes.
    
    folder_path = "experiment_1/instruction_0/datamodels/collections/test"

    # --- Safety Check ---
    if folder_path == "REPLACE_WITH_YOUR_FOLDER_PATH":
        print("Error: Please open the script and set the 'folder_path' variable.")
        sys.exit(1)
        
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at path: {folder_path}")
        print("Please check the 'folder_path' variable in the script.")
        return

    print(f"Scanning folder: {folder_path}...")
    
    files_renamed = 0
    files_scanned = 0
    
    # --- 2. LOOP THROUGH ALL FILES IN THE FOLDER ---
    for filename in os.listdir(folder_path):
        
        # Check if it's a file and ends with .feather
        full_old_path = os.path.join(folder_path, filename)
        if filename.endswith(".feather") and os.path.isfile(full_old_path):
            files_scanned += 1
            
            new_name = "" # Initialize new_name
            try:
                # 1. Split filename into base and suffix (at the last '_')
                # e.g., "instruction-0-1" and "360000.feather"
                parts = filename.rsplit('_', 1)
                if len(parts) != 2:
                    raise ValueError("Filename does not contain an underscore '_' separator.")
                
                base_part, suffix_part = parts


                base_parts = base_part.rsplit('-', 1)
                if len(base_parts) != 2:
                    raise ValueError("Filename base part does not contain a hyphen '-' separator.")
                
                main_part, experiment_num = base_parts

                # 3. Construct the new name
                new_name = f"{main_part}_experiment-{experiment_num}_evaluator-Judge_{suffix_part}"

            except ValueError as e:
                print(f"Skipping '{filename}': Does not match expected format. Error: {e}")
                continue
            except Exception as e:
                print(f"Skipping '{filename}': An unexpected error occurred. Error: {e}")
                continue

            # --- 4. PREPARE TO RENAME (SAFELY) ---
            
            # Ensure the logic actually changed the name
            if new_name == filename:
                print(f"Skipping '{filename}': Renaming logic resulted in no change.")
                continue

            full_new_path = os.path.join(folder_path, new_name)

            # Safety check: Don't overwrite an existing file
            if os.path.exists(full_new_path):
                print(f"Skipping '{filename}': Target file '{new_name}' already exists.")
            else:
                # --- 5. PERFORM THE RENAME ---
                try:
                    os.rename(full_old_path, full_new_path)
                    print(f"SUCCESS: {filename}  ->  {new_name}")
                    files_renamed += 1
                except OSError as e:
                    print(f"ERROR renaming {filename}: {e}")

    print("\n--- Renaming Complete ---")
    if files_scanned == 0:
        print("No .feather files were found in the specified folder.")
    else:
        print(f"Scanned {files_scanned} .feather files.")
        print(f"Successfully renamed {files_renamed} files.")

# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    # WARNING: This script modifies files on your computer.
    # PLEASE BACK UP YOUR FILES before running.
    rename_feather_files()