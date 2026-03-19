import os
import re

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(old_path):
            continue
        
        new_name = re.sub(r'[^a-z0-9._-]', '_', filename.lower())
        new_path = os.path.join(folder_path, new_name)
        
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_name}')
        else:
            print(f'Skipped (already exists): {new_name}')

target_folder = r'C:\Users\joaov_zm1q2wh\projects\quantization\docs\references\ml_heathcare'
rename_files(target_folder)