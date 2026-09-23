import os
import hashlib
import json

class IndexMeta:
    def __init__(self, data_dir, meta_file):
        self.data_dir = data_dir
        self.meta_file = meta_file

    def get_dir_hash(self):
        """Calculates a combined MD5 hash of all raw data files in the data directory."""
        hasher = hashlib.md5()
        # Sort files to ensure the hash is consistent regardless of OS reading order
        for root, _, files in os.walk(self.data_dir):
            for file in sorted(files):
                # Hash JSONs, TXTs, and PDFs
                if file.endswith('.json') or file.endswith('.txt') or file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        # Read and hash the actual file contents
                        buf = f.read()
                        hasher.update(buf)
        return hasher.hexdigest()

    def is_stale(self):
        """Returns True if the data directory has been modified since the last build."""
        current_hash = self.get_dir_hash()
        
        # If there is no previous build metadata, it is definitely stale
        if not os.path.exists(self.meta_file):
            return True
            
        with open(self.meta_file, 'r') as f:
            try:
                saved_meta = json.load(f)
            except json.JSONDecodeError:
                return True
            
        return current_hash != saved_meta.get("data_hash")

    def update_meta(self):
        """Saves the current data directory hash to the meta file after a successful build."""
        current_hash = self.get_dir_hash()
        os.makedirs(os.path.dirname(self.meta_file), exist_ok=True)
        
        with open(self.meta_file, 'w') as f:
            json.dump({"data_hash": current_hash}, f)
