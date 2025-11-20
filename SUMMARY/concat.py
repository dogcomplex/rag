import os
import sys
import json
import re
from datetime import datetime
try:
    # Allow optional in-process sanitization if enabled via config
    from sanitize_for_gemini import aggressive_sanitize_text_content as _gem_sanitize
except Exception:
    _gem_sanitize = None

# Define encodings globally for use in multiple functions
encodings = ['utf-8', 'latin-1', 'cp1252']

if len(sys.argv) not in [2, 3]:
    print("Usage: python concat.py <config_key> [--batch]")
    sys.exit(1)

if len(sys.argv) == 3 and sys.argv[2] != '--batch':
    print(f"Error: Unknown argument '{sys.argv[2]}'")
    print("Usage: python concat.py <config_key> [--batch]")
    sys.exit(1)

config_path = sys.argv[1]

if os.path.isdir(config_path):
    print(f"Error: Configuration path '{config_path}' is a directory. Please provide a path to a JSON configuration file.")
    sys.exit(1)

with open(config_path, 'r') as config_file:
    configs = json.load(config_file)

config = configs['concat']

# New configuration for file sizes
max_file_size_mb = config.get("max_file_size_mb", 2.8)
total_size_hard_cap_mb = config.get("total_size_hard_cap_mb", 20.0)
max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
total_size_hard_cap_bytes = int(total_size_hard_cap_mb * 1024 * 1024)
max_individual_file_size_mb = config.get("max_individual_file_size_mb", 1.0)
max_individual_file_size_bytes = int(max_individual_file_size_mb * 1024 * 1024)
skip_binary_files = config.get("skip_binary_files", True)
sanitize_for_gemini = config.get("sanitize_for_gemini", False)

excluded_files = config["excluded_files"]
excluded_files_with_tree = config["excluded_files_with_tree"]
excluded_file_types = tuple(config["excluded_file_types"])
# Convert excluded folder names to lowercase for case-insensitive comparison
excluded_folders = [ef.lower() for ef in config.get("excluded_folders", [])]
excluded_contents_folder = [ecf.lower() for ecf in config.get("excluded_contents_folder", [])]
target_folder = configs["target"]
output_folder = "SUMMARY/summaries" # Hardcoded to ensure output to SUMMARY/summaries
original_config_output_folder = configs.get("output") # Keep original for potential logging if needed
if original_config_output_folder != output_folder:
    print(f"INFO: Overriding output folder from config ('{original_config_output_folder}') to '{output_folder}'")
name = configs["name"]

# Helper function to get file size in bytes
def get_file_size_bytes(file_path):
    """Gets the size of a file in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def is_file_excluded(file_path, filename):
    """Checks if a file is excluded from concatenation based on global configs."""
    global excluded_files, excluded_files_with_tree, excluded_file_types, skip_binary_files, max_individual_file_size_bytes
    
    if filename in excluded_files or filename in excluded_files_with_tree:
        return True
    if any(filename.endswith(ext) for ext in excluded_file_types):
        return True
    if skip_binary_files and is_binary(file_path):
        return True
    
    if get_file_size_bytes(file_path) > max_individual_file_size_bytes:
        return True
        
    return False

def is_binary(file_path):
    """Checks if a file is likely binary by looking for null bytes."""
    try:
        with open(file_path, 'rb') as f:
            return b'\x00' in f.read(2048) # Read first 2KB is enough
    except IOError:
        return False

# Recursive helper for generating tree summary
def _generate_summary_recursive_helper(current_path, indent, parent_excluded=False):
    global excluded_folders, excluded_contents_folder
    
    child_lines = []
    current_dir_total_bytes = 0

    try:
        items = sorted(os.listdir(current_path))
    except OSError:
        return [], 0

    for item_name in items:
        item_full_path = os.path.join(current_path, item_name)
        is_dir = os.path.isdir(item_full_path)
        
        item_is_cause_of_exclusion = False
        if is_dir:
            item_name_lower = item_name.lower()
            path_components_lower = [comp.lower() for comp in item_full_path.split(os.path.sep)]
            combined_excluded_folders_lower = [f.lower() for f in (excluded_folders + excluded_contents_folder)]
            # Check if directory name or any part of its path is in the combined exclusion list
            if item_name_lower in combined_excluded_folders_lower or \
               any(excluded_path_part in path_components_lower for excluded_path_part in combined_excluded_folders_lower):
                item_is_cause_of_exclusion = True
        else: # It's a file
            if is_file_excluded(item_full_path, item_name):
                item_is_cause_of_exclusion = True

        is_excluded_for_concatenation = parent_excluded or item_is_cause_of_exclusion
        prefix = "X--" if is_excluded_for_concatenation else "+--"
        
        if is_dir:
            # If excluded, do NOT recurse into the directory tree to avoid huge traversals
            if item_is_cause_of_exclusion:
                child_lines.append(f"{' ' * indent}{prefix} {item_name}/ (skipped)")
            else:
                sub_child_lines, sub_dir_bytes = _generate_summary_recursive_helper(
                    item_full_path, indent + 4, parent_excluded=is_excluded_for_concatenation
                )
                child_lines.append(f"{' ' * indent}{prefix} {item_name}/ ({sub_dir_bytes} bytes)")
                if sub_child_lines:
                    child_lines.extend(sub_child_lines)
                current_dir_total_bytes += sub_dir_bytes
        else:  # It's a file
            bytes_in_file = get_file_size_bytes(item_full_path)
            child_lines.append(f"{' ' * indent}{prefix} {item_name} ({bytes_in_file} bytes)")
            current_dir_total_bytes += bytes_in_file

    return child_lines, current_dir_total_bytes

def generate_tree_summary(root_dir_path):
    if not os.path.isdir(root_dir_path):
        file_name = os.path.basename(root_dir_path)
        is_excluded = is_file_excluded(root_dir_path, file_name)
        prefix = "X--" if is_excluded else "+--"
        bytes_in_file = get_file_size_bytes(root_dir_path)
        return f"{prefix} {file_name} ({bytes_in_file} bytes)\n"

    root_folder_name = os.path.basename(os.path.abspath(root_dir_path))
    child_lines, total_project_bytes = _generate_summary_recursive_helper(root_dir_path, 0)
    
    summary_string = f"+-- {root_folder_name}/ ({total_project_bytes} bytes)\n"
    summary_string += "\n".join(child_lines)
    return summary_string

def filter_base64(content):
    """
    More robust filter to remove lines that appear to contain large base64 encoded data blobs.
    It checks for long strings of base64 characters, often found as values in key-value pairs.
    """
    # Regex to match a string that is composed almost entirely of base64 characters.
    # Allows for some non-base64 characters for robustness.
    base64_pattern = re.compile(r"^[A-Za-z0-9+/=]+$")
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Heuristic 1: The entire line is just a very long base64 string.
        if len(stripped_line) > 200 and base64_pattern.match(stripped_line):
            continue

        # Heuristic 2: The line contains a key-value pair where the value is a base64 string.
        # This is designed to catch patterns like: "key": "iVBORw0KGgo..."
        try:
            # Find the position of the last colon, which typically separates key from value.
            last_colon_pos = stripped_line.rfind(':')
            if last_colon_pos > 0:
                # Extract the potential value part after the colon.
                value_part = stripped_line[last_colon_pos + 1:].strip()
                
                # Clean up potential surrounding characters like quotes and commas.
                if value_part.startswith('"'):
                    value_part = value_part[1:]
                if value_part.endswith('"') or value_part.endswith('",'):
                    value_part = value_part[:-1]
                    if value_part.endswith('"'):
                         value_part = value_part[:-1]
                
                # Check if the cleaned value part is a long base64 string.
                if len(value_part) > 200 and base64_pattern.match(value_part):
                    continue # Skip the entire line if it matches.
        except Exception:
            # If any string processing fails, just keep the line.
            pass

        filtered_lines.append(line)
            
    return '\n'.join(filtered_lines)

def concatenate_files_and_split(root_dir, tree_content, base_prompt_text, output_name):
    global encodings, max_file_size_bytes, total_size_hard_cap_bytes, output_folder

    files_to_process = get_files_to_process(root_dir)

    file_data = []
    total_size = 0
    for file_path in files_to_process:
        size = get_file_size_bytes(file_path)
        if size > max_individual_file_size_bytes:
            print(f"Skipping file {file_path} because its size ({size / (1024*1024):.2f}MB) exceeds the individual file limit of {max_individual_file_size_mb}MB.")
            continue

        relative_path = os.path.relpath(file_path, start=os.getcwd())
        header_path = os.path.join(target_folder, relative_path)
        header_size = len(f"{header_path}\n\n\n".encode('utf-8'))
        
        file_data.append({'path': file_path, 'size': size, 'header_size': header_size})
        total_size += size + header_size

    if total_size > total_size_hard_cap_bytes:
        print(f"Error: Total estimated size of content (~{total_size / (1024*1024):.2f}MB) exceeds hard cap of {total_size_hard_cap_mb}MB. Aborting.")
        sys.exit(1)
        
    header_and_tree_content = base_prompt_text + tree_content
    header_and_tree_size = len(header_and_tree_content.encode('utf-8'))

    chunks = []
    current_chunk_files = []
    current_chunk_size = header_and_tree_size

    for f_data in file_data:
        file_total_size = f_data['size'] + f_data['header_size']
        if current_chunk_size + file_total_size > max_file_size_bytes and current_chunk_files:
            chunks.append(current_chunk_files)
            current_chunk_files = []
            current_chunk_size = 0
        
        current_chunk_files.append(f_data['path'])
        current_chunk_size += file_total_size

    if current_chunk_files:
        chunks.append(current_chunk_files)
    
    total_parts = len(chunks)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    output_filename_base = f"{timestamp}_{output_name}"
    
    os.makedirs(output_folder, exist_ok=True)

    for i, chunk_files in enumerate(chunks, 1):
        if total_parts > 1:
            output_filename = f"{output_filename_base}_part{i}.txt"
        else:
            output_filename = f"{output_filename_base}.txt"
        
        output_path = os.path.join(output_folder, output_filename)
        print(f"Writing part {i}/{total_parts} to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as outfile:
            if i == 1:
                outfile.write(header_and_tree_content)

            for file_path in chunk_files:
                relative_path = os.path.relpath(file_path, start=os.getcwd())
                header_path_str = os.path.join(target_folder, relative_path)
                
                file_content = None
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            file_content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if file_content is not None:
                    # Filter out base64 content before writing
                    filtered_content = filter_base64(file_content)
                    outfile.write(f"{header_path_str}\n{filtered_content}\n\n")
                else:
                    outfile.write(f"{header_path_str}\n```\nUnable to decode file content.\n```\n")

        # Optional sanitize pass to produce a Gemini-friendly variant
        if sanitize_for_gemini:
            try:
                if _gem_sanitize is not None:
                    with open(output_path, 'r', encoding='utf-8', errors='ignore') as _rf:
                        _raw = _rf.read()
                    _clean = _gem_sanitize(_raw)
                    _sanitized_path = os.path.splitext(output_path)[0] + "_sanitized.txt"
                    with open(_sanitized_path, 'w', encoding='utf-8') as _wf:
                        _wf.write(_clean)
                    print(f"Sanitized file written to: {_sanitized_path}")
                else:
                    # Fallback: attempt to run the sanitizer script as a subprocess
                    os.system(f"python SUMMARY/sanitize_for_gemini.py \"{output_path}\"")
            except Exception as _e:
                print(f"WARN: sanitize_for_gemini failed on {output_path}: {_e}")

def get_files_to_process(root_dir):
    files = []
    if not os.path.isdir(root_dir):
        # It's a single file path
        if skip_binary_files and is_binary(root_dir):
            print(f"Skipping binary file: {root_dir}")
            return []
        filename = os.path.basename(root_dir)
        if not any(filename.endswith(ext) for ext in excluded_file_types) and filename not in excluded_files + excluded_files_with_tree:
            return [root_dir]
        return []

    combined_excluded_folders = excluded_folders + excluded_contents_folder

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d.lower() not in combined_excluded_folders]

        path_components_lower = [comp.lower() for comp in dirpath.split(os.path.sep)]
        if any(excluded_folder in path_components_lower for excluded_folder in combined_excluded_folders):
            continue

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if skip_binary_files and is_binary(file_path):
                print(f"Skipping binary file: {file_path}")
                continue

            if not any(filename.endswith(ext) for ext in excluded_file_types) and filename not in excluded_files + excluded_files_with_tree:
                files.append(file_path)
    return files


base_prompt = """Here is the file tree and contents of all files in the project:
"""

def run_single_summary(root_dir, output_name):
    # For single files, ensure they are not excluded before proceeding
    if not os.path.isdir(root_dir):
        filename = os.path.basename(root_dir)
        if is_file_excluded(root_dir, filename):
            print(f"Skipping excluded or binary file in batch mode: {root_dir}")
            return

    tree_summary_content = generate_tree_summary(root_dir)
    tree_header = "target_folder: " + root_dir + ("/" if os.path.isdir(root_dir) else "") + "\n"
    full_tree_for_concat = tree_header + tree_summary_content
    
    concatenate_files_and_split(root_dir, full_tree_for_concat, base_prompt, output_name)


if __name__ == "__main__":
    batch_mode = len(sys.argv) == 3 and sys.argv[2] == '--batch'
    
    if batch_mode:
        print("Running in batch mode...")
        # In batch mode, iterate over items in the target folder
        for item_name in sorted(os.listdir(target_folder)):
            item_path = os.path.join(target_folder, item_name)
            
            # Skip items that are in the excluded folders list
            if item_name.lower() in [f.lower() for f in excluded_folders]:
                print(f"Skipping excluded directory in batch mode: {item_path}")
                continue

            run_single_summary(item_path, f"{name}_{item_name}")
    else:
        # Default behavior: run on the entire target folder
        print("Running in single mode...")
        run_single_summary(target_folder, name)