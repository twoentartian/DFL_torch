import argparse
import torch

# Setup argument parser
parser = argparse.ArgumentParser(description="Update model_name in a pickle file")
parser.add_argument("file", help="Path to the pickle file")
parser.add_argument("model_name", help="New model name to set")

args = parser.parse_args()

# Load pickle file
with open(args.file, "rb") as f:
    data = torch.load(f, map_location=torch.device("cpu"))

# Update value
data["model_name"] = args.model_name

# Save back
with open(f'{args.file}.new.pt', "wb") as f:
    torch.save(data, f)

print(f'Updated "model_name" to {data["model_name"]} in {args.file}')
