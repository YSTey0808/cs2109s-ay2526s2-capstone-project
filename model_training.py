import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import generate_torch_loader_snippet


class TileClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

# --- 2. ASSET COMPOSITING DATASET ---
class AssetSyntheticDataset(Dataset):
    def __init__(self, assets_dir="data/assets", num_samples=5000):
        self.num_samples = num_samples
        
        # Map your exact folder names to the class indices
        self.folder_to_idx = {
            'human': 0, 'wall': 1, 'exit': 2, 'gem': 3,
            'coin': 4, 'key': 5, 'locked': 6, 'lava': 7,
            'box': 8, 'boots': 9, 'shield': 10, 
            'ghost': 11, 'opened': 12, 'floor': 13
        }

        # Load all images into memory for super-fast training
        self.assets = {k: [] for k in self.folder_to_idx.keys()}
        print(f"Loading raw assets from {assets_dir}...")
        
        for folder, idx in self.folder_to_idx.items():
            folder_path = os.path.join(assets_dir, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder '{folder}' not found in {assets_dir}!")
                continue
                
            for file_path in glob.glob(os.path.join(folder_path, "*.png")):
                # Ensure RGBA (alpha channel is required for pasting)
                img = Image.open(file_path).convert("RGBA")
                # Resize to standard 48x48
                img = img.resize((48, 48), Image.Resampling.NEAREST)
                self.assets[folder].append(img)

    def __len__(self): 
        # Since we generate them dynamically, we decide how long an "epoch" is
        return self.num_samples

    def __getitem__(self, idx):
        label_vector = torch.zeros(14)
        
        # 1. Pick a base tile (Usually a floor, sometimes a wall)
        base_type = 'floor' if random.random() < 0.85 else 'wall'
        
        # Ensure the folder actually had images
        if len(self.assets[base_type]) > 0:
            base_img = random.choice(self.assets[base_type]).copy()
            label_vector[self.folder_to_idx[base_type]] = 1.0
        else:
            # Fallback if floor images are missing
            base_img = Image.new("RGBA", (48, 48), (0,0,0,255))

        # 2. If it's a floor, maybe paste an entity on top of it!
        if base_type == 'floor':
            # 70% chance to put something on the floor
            if random.random() < 0.70:
                # Pick a random entity that goes on floors
                fg_classes = [k for k in self.folder_to_idx.keys() if k not in ['floor', 'wall']]
                fg_choice = random.choice(fg_classes)
                
                if len(self.assets[fg_choice]) > 0:
                    fg_img = random.choice(self.assets[fg_choice])
                    
                    # The Magic: Paste the transparent entity onto the floor!
                    base_img.paste(fg_img, (0, 0), fg_img)
                    label_vector[self.folder_to_idx[fg_choice]] = 1.0

        # Convert the final composited PIL Image to a PyTorch Tensor
        img_array = np.array(base_img)
        tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return tensor, label_vector

# --- 3. TRAINING LOOP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We request 10,000 synthetic tile generations per epoch
dataset = AssetSyntheticDataset(assets_dir="data/assets", num_samples=10000)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TileClassifier(num_classes=14).to(device)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
print("Starting training on synthetic asset composites...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_tiles, batch_labels in dataloader:
        batch_tiles, batch_labels = batch_tiles.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_tiles)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

model = model.cpu()
print("Training Complete!")

# --- 4. SNIPPET GENERATION ---
dummy_input = torch.zeros(1, 4, 48, 48)
snippet = generate_torch_loader_snippet(model, example_inputs=dummy_input, prefer="auto", compression="zlib")
print("\n" + "="*50)
print("COPY THIS SNIPPET INTO YOUR AGENT CODE:")
print("="*50)
print(snippet)