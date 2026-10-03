import csv
import h5py
import numpy as np
from pathlib import Path

INPUT = Path("simulation/dataset/output/train.h5")
OUTPUT = Path("simulation/dataset/output/aurora_rsi_dataset.csv")


def flatten_dataset(arr):
    arr = np.asarray(arr)

    if arr.ndim == 0:
        return [arr.item()]

    return arr.reshape(-1)


with h5py.File(INPUT, "r") as f:

    print("\nHDF5 DATASETS:")
    for key in f.keys():
        print(f"{key:35s} shape={f[key].shape} dtype={f[key].dtype}")

    # ONE-SAMPLE dataset:
    # Every HDF5 field is flattened into columns.
    row = {}

    for key in f.keys():

        arr = np.asarray(f[key][()])

        if np.iscomplexobj(arr):
            flat = arr.reshape(-1)

            for i, value in enumerate(flat):
                row[f"{key}_real_{i}"] = float(np.real(value))
                row[f"{key}_imag_{i}"] = float(np.imag(value))

        else:
            flat = arr.reshape(-1)

            for i, value in enumerate(flat):
                row[f"{key}_{i}"] = value.item()

# Write CSV
with open(OUTPUT, "w", newline="") as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=["sample_id"] + list(row.keys())
    )

    writer.writeheader()

    writer.writerow({
        "sample_id": 0,
        **row
    })

print("\n==============================================")
print("CSV DATASET CREATED")
print("==============================================")
print("Samples : 1")
print(f"Columns : {len(row) + 1}")
print(f"Output  : {OUTPUT}")
print("==============================================")
