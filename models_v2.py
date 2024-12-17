import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
from typing import List
import random 
import math

#import matplotlib.pyplot as plt

#########################
# Augmentations
#########################

def shift_augmentation(states, actions):
    """
    Example augmentation function for the TrajectoryDataset.
    Args:
        states (Tensor): Tensor of shape (T, 2, 64, 64).
        actions (Tensor): Tensor of shape (T-1, 2).
    
    Returns:
        Tuple[Tensor, Tensor]: Augmented states and actions.
    """
    # Check for edges of the agent
    _, _, width_non_zeros = torch.nonzero((states[:, 0] != 0), as_tuple=True)
    width_min = width_non_zeros.min().item()
    width_max = width_non_zeros.max().item()

    # Check for edges of the walls
    wall_non_zeros = torch.nonzero(states[-1, 1, 0, 5:-5] != 0)
    wall_min = wall_non_zeros.min().item()+5
    wall_max = wall_non_zeros.max().item()+5

    wall_pos = int((wall_min+wall_max)/2)


    # Identify range of the data (lowest and highest index where it is not empty space)
    global_min_all = min(width_min, width_max, wall_min, wall_max)
    global_max_all = max(width_min, width_max, wall_min, wall_max)

    # Randomly determine shift (without breaking out of the box)
    min_shift = 5 - global_min_all
    max_shift = 59 - global_max_all
    if global_min_all < 5 or global_max_all > 59:
        shift = 0
    elif min_shift is not max_shift+1 or min_shift is not max_shift:
        try:
            shift = torch.randint(min_shift, max_shift + 1, size=(1,))
        except:
            shift = 0
    else:
        shift = min_shift

    if isinstance(shift, torch.Tensor):
        shift = shift.item()

    # In-place shift for the first channel (primary state)
    states[:, 0].copy_(torch.roll(states[:, 0], shifts=shift, dims=2))
    
    # Special handling for walls (second channel)
    # Separate the edges and core
    left_edge = states[:, 1, :, 0:5].clone()
    right_edge = states[:, 1, :, -5:].clone()
    core = states[:, 1, :, 5:-5]
    
    # In-place shift of the core part
    shifted_core = torch.roll(core, shifts=shift, dims=2)
    
    # Reconstruct the wall channel
    states[:, 1, :, 5:-5] = shifted_core
    
    return states, actions


def random_cropping_augmentation(states, actions):
    T, channels, height, width = states.shape
    
    # Random crop parameters
    crop_height = random.randint(48, 64)  # Crop height between 48 and 64 pixels
    crop_width = random.randint(48, 64)   # Crop width between 48 and 64 pixels
    
    # Randomly determine crop start coordinates
    start_h = random.randint(0, height - crop_height)
    start_w = random.randint(0, width - crop_width)
    
    # Perform random cropping
    cropped_states = states[:, :, start_h:start_h+crop_height, start_w:start_w+crop_width]
    
    # Resize cropped states back to original size using interpolation
    cropped_states = torch.nn.functional.interpolate(
        cropped_states, 
        size=(height, width), 
        mode='bilinear', 
        align_corners=False
    )
    
    return cropped_states, actions

def flip_augmentation(states, actions):
    """
    Example augmentation function for the TrajectoryDataset.
    Args:
        states (Tensor): Tensor of shape (T, 2, 64, 64).
        actions (Tensor): Tensor of shape (T-1, 2).
    
    Returns:
        Tuple[Tensor, Tensor]: Augmented states and actions.
    """
    if random.random() > 0.5:
        states = torch.flip(states, dims=[-1])  # Flip along the width
        actions[:, 0] = -actions[:, 0]  # Invert x-axis action

    # Random vertical flip
    if random.random() > 0.5:
        states = torch.flip(states, dims=[-2])  # Flip along the height
        actions[:, 1] = -actions[:, 1]  # Invert y-axis action
    return states, actions
#########################
# Dataset and Dataloader
#########################

class TrajectoryDataset(Dataset):
    def __init__(self, states_path, actions_path, augmentations=None):
        """
        Args:
            states_path (str): Path to the states .npy file.
            actions_path (str): Path to the actions .npy file.
            augmentations (callable, optional): A function or transform to apply to the states and actions.
        """
        self.states = np.load(states_path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')
        # self.states = torch.tensor(self.states, dtype=torch.float32)
        # self.actions = torch.tensor(self.actions, dtype=torch.float32)
        
        self.augmentations = augmentations
        self.wall_positions = []

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        states = torch.tensor(self.states[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.float32)
        # states, actions = self.states[idx], self.actions[idx]
        
        # Apply augmentations if specified
        if self.augmentations:
            for augmentation in self.augmentations:
                states, actions = augmentation(states, actions)
        return states, actions

#########################
# Model Components
#########################

#########################
# Encoder
#########################

# class Encoder(nn.Module):
#     def __init__(self, in_channels=2, state_dim=128):
#         super().__init__()
#         # Simple CNN encoder
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.GELU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.GELU(),
#         )
#         # After downsampling 64x64 -> approximately 8x8 feature map
#         self.fc = nn.Linear(128 * 8 * 8, state_dim)

#     def forward(self, x):
#         if x.ndimension() == 5:  # (B, T, C, H, W) 
#             B, T, C, H, W = x.shape
#             x = x.view(B * T, C, H, W)  # Flatten batch and sequence dims
#             h = self.conv(x) # B * T, 128, 8, 8
#             h = h.view(h.size(0), -1)
#             s = self.fc(h)
#             s = s.view(B*T,2,8,-1)
#             # s = s.view(B, T, -1)  # Restore batch and sequence dims # (B,T,D)
#         else:  # (B, C, H, W) 
#             h = self.conv(x) #B, 128, 8, 8
#             h = h.view(h.size(0), -1)  # Flatten for FC layer
#             s = self.fc(h) # (B,D)
#             s = s.view(B,2,8,-1)
#         return s

# class Encoder(nn.Module):
#     def __init__(self, in_channels=2, state_dim=128):
#         super().__init__()
#         self.vision_transformer = VisionTransformer(
#             in_channels=in_channels, 
#             embed_dim=state_dim,
#             img_size=64,  # Assuming 64x64 input 
#             patch_size=8
#         )
    
#     def forward(self, x):
#         # x shape: [B, T, C, H, W]
#         if len(x.shape) == 5:
#             B, T, C, H, W = x.shape
#             x = x.view(B*T, C, H, W)
        
#         # Extract features using Vision Transformer
#         features = self.vision_transformer(x)
        
#         return features

#########################
# Recurrent CNN Predictor
#########################

class RecurrentPredictor(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128, cnn_channels=64):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(2 * 2, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cnn_channels, 2, kernel_size=3, padding=1),
        )

    def forward(self, prev_state, action):
        """
        Args:
            prev_state: Tensor of shape (B, state_dim, H, W)
            action: Tensor of shape (B, action_dim)
        Returns:
            next_state: Tensor of shape (B, state_dim, H, W)
        """
        B, D, H, W = prev_state.size()
        # print(prev_state.shape)
        
        # Pass action through MLP and reshape for spatial dimensions
        action_embedding = self.action_mlp(action)
        # print(f'1:{action_embedding.shape}')
        action_embedding = action_embedding.view(B, D, H, W)
        # print(f'2:{action_embedding.shape}')
        # action_embedding = action_embedding.expand(-1, -1, H, W)
        # print(f'3:{action_embedding.shape}')
        
        # Concatenate state and action embeddings
        x = torch.cat([prev_state, action_embedding], dim=1)  # (B, 2 * state_dim, H, W)
        # print(f'3:{x.shape}')
        next_state = self.cnn(x)  # (B, state_dim, H, W)
        # print(f'4:{next_state.shape}')
        
        return next_state


#########################
# ViT
#########################
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=2, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional embedding to match original encoder's structure
        self.proj = nn.Conv2d(in_channels, embed_dim//4, 
                              kernel_size=4, 
                              stride=2, 
                              padding=1)
        
        # Additional convolutions to match the downsampling
        self.additional_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU()
        )
    
    def forward(self, x):
        # x shape: [batch_size, in_channels, img_size, img_size]
        x = self.proj(x)
        x = self.additional_conv(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, num_patches, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_patches, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_patches, self.num_heads, self.head_dim)
        
        # Transpose for multi-head attention
        q = q.transpose(1, 2)  # [batch_size, num_heads, num_patches, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, embed_dim)
        
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=64,
                 patch_size=8, 
                 in_channels=2,
                 state_dim=128):
        super().__init__()
        
        # Patch embedding with convolutions
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=state_dim
        )
        
        # Final projection to match original encoder's output
        self.final_proj = nn.Linear(state_dim * 8 * 8, state_dim)
    
    def forward(self, x):
        # Handle both (B, T, C, H, W) and (B, C, H, W) input formats
        if x.ndimension() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)  # Flatten batch and sequence dims
            
            # Convolutional feature extraction
            h = self.patch_embed(x)  # B*T, state_dim, 8, 8
            
            # Flatten and project
            h = h.view(h.size(0), -1)
            s = self.final_proj(h)
            
            # Reshape to match original encoder's output
            s = s.view(B*T, 2, 8, -1)
        else:
            # Single image input
            h = self.patch_embed(x)  # B, state_dim, 8, 8
            h = h.view(h.size(0), -1)
            s = self.final_proj(h)
            s = s.view(x.size(0), 2, 8, -1)
        
        return s
    
class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=128):
        super().__init__()
        self.vision_transformer = VisionTransformer(
            in_channels=in_channels, 
            state_dim=state_dim,
            img_size=64,
            patch_size=8
        )
    
    def forward(self, x):
        return self.vision_transformer(x)

#########################
# JEPA Model (Recurrent)
#########################

class JEPA(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128, ema_rate=0.99, cnn_channels=64):
        super().__init__()
        self.repr_dim = state_dim

        # Online encoder (learned)
        self.online_encoder = Encoder(in_channels=2, state_dim=state_dim)

        # Target encoder (EMA copy of online encoder)
        self.target_encoder = deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Recurrent CNN Predictor
        self.predictor = RecurrentPredictor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, cnn_channels=cnn_channels)

        # EMA update rate
        self.ema_rate = ema_rate

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder using exponential moving average (EMA)."""
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data

    def forward(self, states, actions):
        """
        Args:
            states: Tensor of shape (B, T, 2, 64, 64)
            actions: Tensor of shape (B, T-1, 2)

        Returns:
            predicted_states: Predicted latent states (B, T-1, D)
            target_next_states: Target latent states (B, T-1, D)
            all_states: All latent states including the first online state (B, T, D)
        """
        B, T, _, _, _ = states.shape 

        encoded_states = self.online_encoder(states) # Shape: (B*T, 128, 8, 8) or B, 128, 8, 8 at inference
        
        H,W = 8, 8 
        encoded_states = encoded_states.view(B, T, -1, H, W)  # Shape: (B, T, 128, 8, 8)
        

        initial_state = encoded_states[:, 0] # Shape: (B, 128, 8, 8)
        predicted_states = []
        prev_state = initial_state

        for t in range(actions.size(1)):  # T-1 iterations
            action = actions[:, t]  # (B, action_dim)
            next_state = self.predictor(prev_state, action)  # (B, D, H, W)
            predicted_states.append(next_state.view(B, -1))  # Flatten spatial dims for final output
            prev_state = next_state

        predicted_states = torch.stack(predicted_states, dim=1)  # (B, T-1, D)
        
        if T > 1:  # Training scenario
            target_next_states = encoded_states[:, 1:].view(B, T-1, -1)  # (B, T-1, D)
        else:  # Inference scenario
            target_next_states = 0  # Placeholder value for inference

        all_states = torch.cat([initial_state.view(B, 1, -1), predicted_states], dim=1)  # Shape: (B, T, 128*8*8)

        return predicted_states, target_next_states, all_states

#########################
# Regularization Utilities
#########################

def variance_regularization(latents, epsilon=1e-4):
    var = torch.var(latents, dim=0)
    return torch.mean(torch.clamp(epsilon - var, min=0))

def covariance_regularization(latents):
    latents = latents - latents.mean(dim=0)
    latents = latents.view(latents.size(0), -1)  # Flatten all dimensions except the batch dimension
    cov = torch.mm(latents.T, latents) / (latents.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.sum(off_diag ** 2)

def normalize_latents(latents):
    return latents / (torch.norm(latents, dim=-1, keepdim=True) + 1e-8)

def contrastive_loss(predicted_states, target_states, temperature=0.1):
    """
    Compute contrastive loss between predicted and target states.
    Args:
        predicted_states: Tensor of shape (B, T-1, D)
        target_states: Tensor of shape (B, T-1, D)
        temperature: Temperature scaling factor for contrastive loss
    Returns:
        loss: Contrastive loss value
    """
    B, T_minus_1, D = predicted_states.shape
    predicted_states = predicted_states.reshape(-1, D)
    target_states = target_states.reshape(-1, D)

    # Normalize the embeddings
    predicted_states = normalize_latents(predicted_states)
    target_states = normalize_latents(target_states)

    # Compute similarity scores
    logits = torch.mm(predicted_states, target_states.T) / temperature
    labels = torch.arange(B * T_minus_1, device=predicted_states.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

#########################
# Training Loop Example
#########################

if __name__ == "__main__":
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    # Hyperparams
    batch_size = 64
    lr = 3e-4
    epochs = 10
    state_dim = 128
    action_dim = 2
    hidden_dim = 128
    cnn_channels = 64
    initial_accumulation_steps = 4  # Initial number of steps to accumulate gradients
    final_accumulation_steps = 4    # Final number of steps to accumulate gradients
    
    # Load data
    # augmentations = [flip_augmentation, shift_augmentation, random_cropping_augmentation]
    # train_dataset = TrajectoryDataset("/scratch/DL24FA/train/states.npy", "/scratch/DL24FA/train/actions.npy", augmentations=augmentations)   
    train_dataset = TrajectoryDataset("/scratch/DL24FA/train/states.npy", "/scratch/DL24FA/train/actions.npy")   
    # train_dataset = TrajectoryDataset("data/states.npy", "data/actions.npy")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, cnn_channels=cnn_channels).to(device)
    if device == 'cuda':
        model = torch.compile(model)

    torch.set_float32_matmul_precision('high')

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=lr*0.1)
    
    loss_history = []

    model.train()
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1}/{epochs} - Before Epoch Start")
        # print(torch.cuda.memory_summary(device=device))

        total_loss = 0.0
        optimizer.zero_grad()
        
        accumulation_steps = max(final_accumulation_steps, initial_accumulation_steps - (initial_accumulation_steps - final_accumulation_steps) * epoch // epochs)
        for step, (states, actions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # print(f"Step {step+1} - After Data Loading")
            # print(torch.cuda.memory_summary(device=device))

            t0 = time.time()
            states = states.to(device)
            actions = actions.to(device)

            # Compute losses
            with torch.autocast(device_type=device, dtype=torch.float16):
                # print(f"Step {step+1} - Before Forward Pass")
                # print(torch.cuda.memory_summary(device=device))
            
                predicted_states, target_states, _ = model(states, actions)

                # print(f"Step {step+1} - After Forward Pass")
                # print(torch.cuda.memory_summary(device=device))

                mse_loss = criterion(predicted_states, target_states)

                # Add variance and covariance regularization
                mse_loss += 0.01 * variance_regularization(predicted_states)
                mse_loss += 0.01 * covariance_regularization(predicted_states)

                # Add contrastive loss
                contrast_loss = contrastive_loss(predicted_states, target_states)
                loss = mse_loss + contrast_loss

            # print(f"Step {step+1} - Before Backward Pass")
            # print(torch.cuda.memory_summary(device=device))

            loss.backward()

            # print(f"Step {step+1} - After Backward Pass")
            # print(torch.cuda.memory_summary(device=device))

            dt=0
            if (step + 1) % accumulation_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update target encoder
                with torch.no_grad():
                    model.update_target_encoder()

                if device == 'mps':
                    torch.mps.synchronize()
                elif device == 'cuda':
                    torch.cuda.synchronize()

                t1 = time.time()
                dt = (t1 - t0) * 1000

            total_loss += loss.item()
            loss_history.append(loss.item())
            print(f"loss {loss.item()}, dt {dt:.2f}ms")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot the loss over time
    """
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('training_loss.png')
    #plt.show()
    """
    # Save the trained model
    torch.save(model.state_dict(), "trained_recurrent_jepa_flip_shift_nowalls_rcrop.pth")
