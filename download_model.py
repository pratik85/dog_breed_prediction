"""
Download model from Google Drive

This script downloads the pre-trained dog breed model from Google Drive
and saves it to the local model/ directory.
"""

import os
import requests
from pathlib import Path

def download_model_from_google_drive(file_id, output_path):
    """
    Download a file from Google Drive.
    
    Args:
        file_id (str): Google Drive file ID from the share link
        output_path (str): Local path where the file will be saved
    """
    # Create model directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Google Drive download URL
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    print(f"Downloading model from Google Drive...")
    print(f"File ID: {file_id}")
    print(f"Saving to: {output_path}")
    
    try:
        # Use requests to download
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Handle Google Drive's warning page for large files
        if 'text/html' in response.headers.get('content-type', ''):
            # Extract real download link from warning page
            cookies = response.cookies
            params = {'id': file_id, 'confirm': 't'}
            response = requests.get(url, params=params, stream=True, cookies=cookies, timeout=300)
        
        # Save the file
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Downloaded: {percent:.1f}%", end='\r')
        
        print(f"\n✓ Model downloaded successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {str(e)}")
        return False


if __name__ == '__main__':
    # Example usage:
    # Replace 'YOUR_FILE_ID' with your Google Drive file ID
    FILE_ID = 'YOUR_FILE_ID'  # Get this from: https://drive.google.com/file/d/1YitdmVzf4FJ6Oq5-x5QGl87ZSlGQXh0o/view?usp=drive_link
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'model', 'dog_breed_model.h5')
    
    download_model_from_google_drive(FILE_ID, OUTPUT_PATH)
