from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from src.preprocessing import (
    preprocess_image,
    get_train_transform,
    get_eval_transform
)


class MedicineDataset(Dataset):

    def __init__(self, df, root_dir, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.root_dir / row["relative_image_path"]

        image = preprocess_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[row["MEDICINE_NAME"]]

        return image, label


def create_dataloaders(
    saved_splits,
    root_dir,
    batch_size=32,
    num_workers=2,
    augment=True
):
    # One mapping shared by all three datasets
    class_names = sorted(
        saved_splits["Training"]["MEDICINE_NAME"].unique()
    )

    class_to_idx = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    datasets = {
        "Training": MedicineDataset(
            saved_splits["Training"],
            root_dir,
            class_to_idx,
            transform=get_train_transform() if augment else get_eval_transform()
        ),

        "Validation": MedicineDataset(
            saved_splits["Validation"],
            root_dir,
            class_to_idx,
            transform=get_eval_transform()
        ),

        "Testing": MedicineDataset(
            saved_splits["Testing"],
            root_dir,
            class_to_idx,
            transform=get_eval_transform()
        )
    }

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "Training"),
            num_workers=num_workers
        )
        for name, dataset in datasets.items()
    }

    return loaders, class_to_idx