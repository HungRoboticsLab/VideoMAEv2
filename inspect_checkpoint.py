import torch
import models
from models.modeling_pretrain import pretrain_videomae_base_patch16_224


def inspect_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("Checkpoint keys:")
    for key in checkpoint.keys():
        print(key)

    print("\nCheckpoint summary:")
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"{key}: dict with {len(value)} keys")
            for subkey in value.keys():
                print(f"  - {subkey}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: Tensor of shape {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    model = pretrain_videomae_base_patch16_224()
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        print("\nNo 'model' key found in checkpoint.")
        return 

    print("\nInpsecting model parameters:")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf in parameters: {name}")

if __name__ == "__main__":
    #checkpoint_path = '/home/dani/data/results/vit_b_pt_300e_scratch_10000/checkpoint-14.pth'
    checkpoint_path = '/mnt/hdd1/common/data/results/vit_b_pt_300e_dani/checkpoint-14.pth'
    inspect_checkpoint(checkpoint_path)

