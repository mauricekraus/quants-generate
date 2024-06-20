import os
import csv
import json
from pathlib import Path

# Path to the root directory containing the JSON files and folders
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = (Path(".") / "generated-dataset").resolve().absolute()

# Create the "csv_data" folder if it doesn't exist
csv_folder = os.path.join(root_dir, "csv_model_data")
os.makedirs(csv_folder, exist_ok=True)

# Find all the numbered folders in the root directory
folders = [folder for folder in os.listdir(root_dir) if folder.isdigit()]

# Iterate over the numbered folders
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    json_path = os.path.join(folder_path, "data.JSON")

    # Check if the JSON file exists in the current folder
    if os.path.isfile(json_path):
        # Load the JSON data
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        # Extract the "prompt_sequence" data
        prompt_sequence = json_data.get("prompt_sequence", [])

        # Create a CSV file path in the "csv_data" folder
        csv_path = os.path.join(csv_folder, f"{folder}.csv")

        # Write the data to the CSV file
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["text", "length"])  # Write header

            # Iterate over the prompt_sequence and write each entry to the CSV file
            for entry in prompt_sequence:
                start = entry.get("start", 0.0)
                end = entry.get("end", 0.0)
                action = entry.get("action", "")
                length = int(
                    (end - start) * 21.05
                )  # Convert length to the desired format

                writer.writerow([action, length])

        print(f"CSV file created: {csv_path}")
    else:
        print(f"JSON file not found in folder: {folder_path}")
