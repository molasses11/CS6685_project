import os.path

import matplotlib.pyplot as plt

import torchvision as tv


class ButterflyDataset(tv.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        datasplit: str = "train",
        **kwargs,
    ):
        """Initialize the dataset.

        This differs from the base `ImageFolder` dataset class primarily by
        automatically joining "/butterflies" to the given root, and using the
        datasplit parameter to specify training vs test vs validation data.
        """
        folder = "butterflies"
        base_root = os.path.join(root, folder)
        new_root = os.path.join(base_root, datasplit)
        if not os.path.exists(base_root):
            raise ValueError(
                f"The dataset does not exist at {base_root}. "
                "It can be downloaded at: "
                "https://www.kaggle.com/datasets/gpiosenka/"
                "butterfly-images40-species/download?datasetVersionNumber=12 "
                f"and extracted to {base_root}."
            )
        if not os.path.exists(new_root):
            raise ValueError(
                f"The specified datasplit {datasplit} does not exist at {new_root}."
                "Maybe you need to download it from "
                "https://www.kaggle.com/datasets/gpiosenka/"
                "butterfly-images40-species/download?datasetVersionNumber=12"
            )
        super().__init__(new_root, **kwargs)


# class ButterflyDatasetOld(data.Dataset):
#     CSV_PATH = "BUTTERFLIES.csv"

#     def __init__(self, root, datasplit="train", transform=None) -> None:
#         """Initialize.

#         Args:
#             root: The path to the root for data. The actual butterfly dataset
#                 will be in <root>/butterflies.
#             datasplit: The portion of the data to load/use. One of train, test,
#                 or valid.
#         """
#         root = os.path.expanduser(os.path.normpath(root))
#         self.root = os.path.join(root, "butterflies")
#         if not os.path.exists(self.root) or not os.path.exists(
#             os.path.join(self.root, self.CSV_PATH)
#         ):
#             raise ValueError(
#                 f"The dataset could not be found in {self.root}. "
#                 "Download from: "
#                 "https://www.kaggle.com/datasets/gpiosenka/"
#                 "butterfly-images40-species/download?datasetVersionNumber=12 "
#                 f"and extract to the folder {self.root}."
#             )
#         index = np.genfromtxt(
#             os.path.join(self.root, self.CSV_PATH),
#             dtype=None,
#             delimiter=",",
#             names=True,
#             encoding=None,
#         )
#         datasplit_idxs = index["data_set"] == datasplit
#         self.index = index[datasplit_idxs]
#         self.classes = self._get_classes()
#         self.transform = transform

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         row = self.index[idx]
#         image = io.imread(os.path.join(self.root, row["filepaths"]))
#         image = Image.fromarray(image)
#         target = row["class_index"]
#         if self.transform is not None:
#             image = self.transform(image)
#             target = torch.Tensor(target)
#         return image, target

#     def _get_classes(self):
#         idxs = self.index["class_index"]
#         labels = self.index["labels"]
#         idx_label_set = set()
#         for idx, label in zip(idxs, labels):
#             if (idx, label) in idx_label_set:
#                 continue
#             idx_label_set.add((idx, label))

#         classes = [""] * len(idx_label_set)
#         for idx, label in idx_label_set:
#             classes[int(idx)] = label.lower()
#         return classes


if __name__ == "__main__":
    import random

    butterflies_data = ButterflyDataset(
        root=os.path.join(os.path.dirname(__file__), "../data")
    )
    print(len(butterflies_data.classes))
    rand_idxs = (random.randint(0, len(butterflies_data) - 1) for _ in range(25))
    plt.subplots(5, 5, layout="constrained", figsize=(12, 8))
    plt.suptitle("Sample Butterfly Images and Labels")
    for i, idx in enumerate(rand_idxs):
        image, label = butterflies_data[idx]
        plt.subplot(5, 5, i + 1)
        plt.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
        # print(image.shape, label)
        plt.title(butterflies_data.classes[label])
        plt.imshow(image)
    plt.show()
