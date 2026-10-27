from PIL import Image
from torchvision import transforms

TARGET_SIZE = (224, 224)


def preprocess_image(image_path, target_size=TARGET_SIZE):
    with Image.open(image_path) as img:
        rgba = img.convert("RGBA")

        background = Image.new("RGBA", img.size, "white")
        image = Image.alpha_composite(
            background, rgba
        ).convert("RGB")

    # Calculate scale while preserving aspect ratio
    scale = min(
        target_size[0] / image.width,
        target_size[1] / image.height
    )

    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale))
    )

    image = image.resize(new_size, Image.Resampling.LANCZOS)

    padded = Image.new("RGB", target_size, "white")

    x = (target_size[0] - new_size[0]) // 2
    y = (target_size[1] - new_size[1]) // 2

    padded.paste(image, (x, y))

    return padded


def get_train_transform(normalize=None):
    steps = [
        transforms.RandomRotation(
            degrees=5,
            fill=(255, 255, 255)
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=(255, 255, 255)
        ),
        transforms.ToTensor()
    ]

    # normalize = [mean, std] -> standardize each channel: (x - mean) / std
    if normalize:
        steps.append(transforms.Normalize(*normalize))

    return transforms.Compose(steps)


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])