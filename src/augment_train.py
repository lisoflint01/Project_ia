import random
from pathlib import Path
import json

import torch
from PIL import Image

from image_transforms import is_image, to_rgb, save_jpg, TRANSFORMS

# Return all real images in directory
def list_images(class_dir: Path, only_real: bool = False) -> list[Path]:
    
    imgs = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
    
    if only_real:    
        imgs = [p for p in imgs if "aug_" not in p.stem]

    return imgs

# Count real images
def count_real_images(class_dir: Path) -> int:
    return len(list_images(class_dir, only_real=True))

# Number of synthetic images to generate
def compute_num_synthetic(num_real: int, target_ratio: float) -> int:
    return round((target_ratio * num_real) / (1 - target_ratio))

# Generate sintetic image
def synthesize_images_inplace(class_dir: Path, num_synthetic: int, seed: int = 42) -> None:

    torch.manual_seed(seed)
    random.seed(seed)
    
    real_imgs = list_images(class_dir, only_real=True)
    
    # Check error in list
    if not real_imgs:
        return
    
    # Generate image while not reach target number
    generated = 0
    while generated < num_synthetic:
        
        src_img = random.choice(real_imgs)
        

        try:
            with Image.open(src_img) as image:
                image = to_rgb(image)
                num_transforms = random.randint(1, min(3, len(TRANSFORMS)))
                
                # Apply transformation
                for transform in random.sample(TRANSFORMS, num_transforms):
                    image = transform(image)
                    
                img_name = f"aug_{class_dir.name}_{generated:04d}.jpg"
                save_jpg(image, class_dir / img_name)
                generated += 1
                
                print(generated)
            
        except Exception as error:
            print(f"[WARN] {src_img.name}: {error}")

# Start generate synthetic images
def main():
    
    with open(Path(__file__).parent / "config_augment.json") as f:
        json_cfg = json.load(f)
        
    train_dir = Path(json_cfg["train_dir"])
    target_ratio = json_cfg["target_ratio"]
    seed = json_cfg["seed"]
    
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
    
        num_real = count_real_images(class_dir)
        num_synthetic = compute_num_synthetic(num_real, target_ratio)
    
        synthesize_images_inplace(class_dir, num_synthetic, seed)
    
if __name__ == "__main__":
    main()
    