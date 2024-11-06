import os
import glob
import torch

def find_latest_best_model():
    return "mosXTTS/models/best_model_28574.pth";
    search_path = "mosXTTS/models/best_model_28574.pth"
    files = glob.glob(search_path, recursive=True)
    latest_file = max(files, key=os.path.getctime, default=None)
    return latest_file

model_path = find_latest_best_model()
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
del checkpoint["optimizer"]
for key in list(checkpoint["model"].keys()):
    if "dvae" in key:
        del checkpoint["model"][key]
torch.save(checkpoint, "model_to_deploy.pth")
model_dir = os.path.dirname(model_path)
print(model_dir)