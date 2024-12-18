import os


def get_file_path(filename: str, directory: str) -> str:
    return os.path.join(directory, filename)


def is_file_in_directory(file_path: str) -> bool:
    return os.path.isfile(file_path)


def is_yaml_or_json_file(file_path: str) -> bool:
    return file_path.lower().endswith((".yaml", ".yml", "json"))
