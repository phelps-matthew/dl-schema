"""
Torch Datasets serve to retrieve features and labels one sample at a time. Accordingly,
the dataset must implement __len__ and __getitem__.

Torch Dataloaders are iterables that abstract batching, shuffling, and multiprocessing.
"""
from torch.utils.data import Dataset
from torchvision.io import read_image
from dl_schema.utils import load_yaml
from pathlib import Path
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    """Sample torch Dataset to be used with torch DataLoader."""

    def __init__(
        self,
        root="/home/mphelps/datasets/micro-speed_1e2",
        split="train",
        transform=None,
        target_transform=None,
    ):
        assert split in {"train", "val"}
        self.root = root
        self.split = split
        self.img_root = (Path(self.root) / split) / "images"
        self.transform = transform
        self.target_transform = target_transform
        if split == "train":
            label_list = load_yaml(self.img_root.parent / "keypoints.json")
        if split == "val":
            label_list = load_yaml(self.img_root.parent / "keypoints.json")

        # Image id's follow format imgxxxxxx.jpg
        self.img_ids = [label["filename"] for label in label_list]
        # Hash map for quick retrieval
        self.labels = {label["filename"]: {"keypoints": label["keypoints"]} for label in label_list}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.img_root / img_id

        # Convert from grayscale to 3-channel to comply with pre-trained networks.
        # Duplicates image across all channels.
        x = Image.open(img_path).convert('RGB')  # PIL image.

        if self.split in {"train", "val"}:
            # Sort to a canonical form.
            keypoints = [v for k,v in sorted(self.labels[img_id]["keypoints"].items())]
            # Python floats are double precision by default.
            y = np.array(keypoints).astype(np.float32)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


if __name__ == "__main__":
    """Test Dataset"""
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Form dataloaders. ToTensor scales image pixels to [0.0, 1.0] floats.
    train_data = MyDataset(split="train", transform=ToTensor())
    test_data = MyDataset(split="val", transform=ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    # Swap channel dimension.
    img = train_features[0].permute(1, 2, 0).numpy()
    label = train_labels[0]

    # Plot with target bounding box
    print(f"Label: {label}")
