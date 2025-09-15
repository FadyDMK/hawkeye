import os
import math
import csv


input_path = "ball_positions.csv"
output_path = "distances_from_pipeline.csv"

distances = []

with open(input_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row["Frame"])
        x = float(row["X"]) if row["X"] != "None" else None
        y = float(row["Y"]) if row["Y"] != "None" else None
        z = float(row["Z"]) if row["Z"] != "None" else None

        if None in (x, y, z):
            distances.append((frame, None))
            
        else:
            distance = math.sqrt(x**2 + y**2 + z**2)
            distances.append((frame, distance))

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Distance"])
    for frame, distance in distances:
        writer.writerow([frame, distance])

print(f"Distances saved to {output_path}.")