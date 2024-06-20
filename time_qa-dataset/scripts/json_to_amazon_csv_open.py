import os
import csv
import json
import random
from pathlib import Path

# This can only be executed when the generated dataset only consists of open questions

# Path to the root directory containing the numbered folders
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = (Path(".") / "generated-dataset").resolve().absolute()

# Create a CSV file path
csv_path = os.path.join(root_dir, "mturk_open.csv")

# Write the data to the CSV file
with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["video_url", "question"])  # Write header

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

            # Extract the "qa_pairs" data
            qa_pairs = json_data.get("qa_pairs", [])

            # Extract the relevant information from the qa_pairs and write it to the CSV file
            for entry in qa_pairs:
                question = entry.get("question", "")

                video_url = (
                    "https://github.com/Patrerror/TimeQAvideos/raw/main/open/"
                    + folder
                    + ".mp4"
                )

                # Adds sanity questions randomly
                if random.randint(0, 10) == 0:
                    writer.writerow(
                        [
                            "https://github.com/Patrerror/TimeQAvideos/raw/main/default_video.mp4",
                            "Write the word 'Table'",
                        ]
                    )

                writer.writerow([video_url, question])

            print(f"CSV data added for folder: {folder}")
        else:
            print(f"JSON file not found in folder: {folder_path}")

print(f"CSV file created: {csv_path}")
