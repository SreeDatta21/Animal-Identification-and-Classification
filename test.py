import os
from ultralytics import YOLO
import subprocess


model = YOLO(r"Animal\besttt.pt")


results = model.predict(source=r"Animal\cow.jpg", save=True)


output_dir = results[0].save_dir


output_file_path = os.path.join(output_dir, 'cow.jpg')


if os.path.exists(output_file_path):
    os.startfile(output_file_path)  
else:
    print(f"Output file not found at {output_file_path}")
 