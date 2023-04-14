from google.cloud import storage
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class GCPyClient():

    WAYMO_DIR = ROOT_DIR / 'waymo_open_dataset' / 'data' / 'tf_example'

    def __init__(self, project_id, bucket_name):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.client = storage.Client(project=self.project_id)
    
    def get_bucket(self):
        return self.client.get_bucket(self.bucket_name)
    
    def get_blob(self, blob_name):
        bucket = self.get_bucket()
        return bucket.get_blob(blob_name)
    
    def upload_blob(self, blob_name, file_path):
        bucket = self.get_bucket()
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
    
    def download_blob(self, blob_name, file_path):
        bucket = self.get_bucket()
        blob = bucket.blob(blob_name)
        blob.download_to_filename(file_path)

    def delete_local_file(self, file_path):
        os.remove(file_path)