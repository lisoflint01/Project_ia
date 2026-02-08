from pathlib import Path
from PIL import Image
from torchvision import transforms

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

# Check photo is in correct format
def is_image(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in IMG_EXTS
    
# Img in rgb
def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
    
# Save images
def save_jpg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality = 95, optimize = True)

# Rotation
def mod_img_rotate(img: Image.Image) -> Image.Image:
    return transforms.RandomRotation((-12, 12))(img)

# Flip
def mod_img_flip(img: Image.Image) -> Image.Image:
    return transforms.RandomHorizontalFlip(p=1.0)(img)

# Brightness ±25%
def mod_img_brightness(img: Image.Image) -> Image.Image:
    return transforms.ColorJitter(brightness=0.25)(img)

# Contrast ±25%
def mod_img_contrast(img:Image.Image) -> Image.Image:
    return transforms.ColorJitter(contrast=0.25)(img)

# Saturation ±25%
def mod_img_saturation(img:Image.Image) -> Image.Image:
    return transforms.ColorJitter(saturation=0.25)(img)

# List of trasformation
TRANSFORMS = [mod_img_rotate, mod_img_flip, mod_img_brightness, mod_img_contrast, mod_img_saturation]

