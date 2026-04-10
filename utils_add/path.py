import os
import glob

def code_name(task, ttype, dtype, idx):
    if idx:
        filename = f"{task}-{ttype}-{dtype}-{idx:04d}.npy"
    else:
        filename = f"{task}-{ttype}-{dtype}.npy"
    return filename

def get_log_filepath(filename, idx=None):
    filepath = os.path.join(os.getcwd(), f"assets/logs/{filename}")
    return filepath

def get_model_path(filename, idx=None):
    if idx:
        filepath = os.path.join(os.getcwd(), f"assets/models/{os.path.splitext(filename)[0]}-{idx:04d}.pt")
    else:
        filepath = os.path.join(os.getcwd(), f"/assets/models/{filename}")
    return filepath