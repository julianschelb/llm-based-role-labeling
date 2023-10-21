# ===========================================================================
#                            File Operations
# ===========================================================================

from bson import json_util
import json
import os


def readJSON(file_path, encoding='utf-8'):
    """
    Imports a JSON file from the given path and returns the data as a Python object.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data


def parseJSON(data):
    """Returns a JSON string from the given data."""
    return json.loads(json_util.dumps(data))


def exportAsJSON(export_filename: str, output):
    """Exports the given data as a JSON file."""
    if export_filename.endswith('.json') is not True:
        raise Exception(f"{export_filename} should be a .json file")

    with open(export_filename, "w") as outfile:
        json.dump(output, outfile, default=str, indent=4)


def validatePath(path: str) -> str:
    """Returns a valid path or prints an error message"""

    if not os.path.exists(path):
        print(f"Directory or file {path} does not exist.")
        exit()

    return path
