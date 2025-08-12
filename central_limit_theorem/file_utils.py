import os
from pathlib import Path
from typing import Dict, List


def read_text_files_recursive(root_folder: str) -> Dict[str, str]:
    """
    Recursively reads all text files from a folder structure.

    Args:
        root_folder: Path to the root folder to search

    Returns:
        Dictionary where keys are hierarchical paths (folder/subfolder/filename)
        and values are file contents
    """
    file_contents = {}
    root_path = Path(root_folder)

    # Check if root folder exists
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder '{root_folder}' does not exist")

    # Recursively find all text files
    for file_path in root_path.rglob("*.md"):
        try:
            # Get relative path from root folder
            relative_path = file_path.relative_to(root_path)

            # Create hierarchical key (folder/subfolder/filename)
            key = str(relative_path).replace(os.sep, "_")

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                file_contents[key] = f.read()

        except (UnicodeDecodeError, PermissionError) as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            continue

    return file_contents


def read_all_text_files_recursive(root_folder: str, extensions: List[str] | None = None) -> Dict[str, str]:
    """
    Recursively reads all text files with specified extensions from a folder structure.

    Args:
        root_folder: Path to the root folder to search
        extensions: List of file extensions to include (default: common text extensions)

    Returns:
        Dictionary where keys are hierarchical paths and values are file contents
    """
    if extensions is None:
        extensions = [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".docx", ".doc"]

    file_contents = {}
    root_path = Path(root_folder)

    if not root_path.exists():
        raise FileNotFoundError(f"Root folder '{root_folder}' does not exist")

    # Recursively find all files with specified extensions
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                # Get relative path from root folder
                relative_path = file_path.relative_to(root_path)

                # Create hierarchical key
                key = str(relative_path).replace(os.sep, "_")

                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    file_contents[key] = f.read()

            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Warning: Could not read file {file_path}: {e}")
                continue

    return file_contents


# Example usage:
if __name__ == "__main__":
    # Example 1: Read only .txt files
    try:
        txt_files = read_text_files_recursive("./dataset/repository/")
        print("Text files found:")
        print(txt_files)
    except FileNotFoundError as e:
        print(e)

    # Example 2: Read multiple file types
    # try:
    #     all_files = read_all_text_files_recursive("./", [".py", ".md", ".txt"])
    #     print(f"\nFound {len(all_files)} files")
    #     for key in sorted(all_files.keys()):
    #         print(f"  {key}")
    # except FileNotFoundError as e:
    #     print(e)
