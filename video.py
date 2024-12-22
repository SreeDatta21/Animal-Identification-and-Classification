import os
from ultralytics import YOLO
import subprocess

model = YOLO(r"Animal\besttt.pt")

input_video_path = path_to_video


results = model.predict(source=input_video_path, save=True)

output_dir = results[0].save_dir

output_file_path = os.path.join(output_dir, os.path.basename(input_video_path))

if os.path.exists(output_file_path):
    os.startfile(output_file_path)  
else:
    print(f"Output file not found at {output_file_path}")
