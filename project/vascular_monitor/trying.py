import torch
CHECKPOINT = r"D:\3IA\Semestre 2\AI_Project\Deployment\projetai-debut\project\vascular_monitor\trying.py"   # adjust if different
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

print("=== STATE DICT KEYS & SHAPES ===")
for key, val in state.items():
    print(f"{key:30s}  shape = {tuple(val.shape)}")