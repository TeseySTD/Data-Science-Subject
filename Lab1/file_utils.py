import os
import subprocess
import zipfile
import pandas as pd

zip_path = "coffee-sales-dataset.zip"
extract_path = "./coffee_data/"


def load_data():
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        script = "get_data.sh"
        res = subprocess.run(['./' + script], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print("Failed to download dataset.")
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
            raise RuntimeError(f"{script} failed with code {res.returncode}")
        print("Dataset downloaded")
    
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print("ZIP file successfully unpacked")

    files = os.listdir(extract_path)
    print(f"Files in dataset: {files}")

    # Finding CSV file
    csv_files = [f for f in files if f.endswith(".csv")]
    if csv_files:
        csv_file = csv_files[0]  # Take the first CSV file
        df = pd.read_csv(os.path.join(extract_path, csv_file))
        print(f"File loaded: {csv_file}")
    else:
        print("No CSV file found")

    return df
