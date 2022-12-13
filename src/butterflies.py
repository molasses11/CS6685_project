import os.path

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


if __name__ == "__main__":
    import itertools
    import random
    import matplotlib.gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from dwt_transform import DWT2Numpy

    butterflies_data = ButterflyDataset(
        root=os.path.join(os.path.dirname(__file__), "../data")
    )
    print(len(butterflies_data.classes))
    random.seed(42)
    rand_idxs = (random.randint(0, len(butterflies_data) - 1) for _ in range(9))
    fig, axs = plt.subplots(3, 3, layout="constrained", figsize=(12, 8))
    # dwt_fig, dwt_axs = plt.subplots(3, 3, layout="constrained", figsize=(12, 8))
    dwt_fig = plt.figure(layout="constrained", figsize=(12, 8))
    dwt_gridspec = matplotlib.gridspec.GridSpec(3, 3, figure=dwt_fig)
    fig.suptitle("Sample Butterfly Images and Labels")
    dwt_fig.suptitle("DWT Sample Butterfly Images and Labels")

    dwt = DWT2Numpy("haar")
    # dwt = DWT2Numpy("db2")

    def plt_im(fig, idx, image, title):
        plt.figure(fig)
        plt.subplot(3, 3, idx + 1)
        plt.title(title)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    for i, idx in enumerate(rand_idxs):
        image, label = butterflies_data[idx]
        plt_im(fig, i, image, butterflies_data.classes[label])
        transforms = dwt(image, normalize=True)

        dwt_subfig = dwt_fig.add_subfigure(dwt_gridspec[i])
        dwt_sub_axs = dwt_subfig.subplots(2, 2, gridspec_kw={"wspace":0, "hspace":0})
        dwt_sub_axs = itertools.chain.from_iterable(dwt_sub_axs)  # flatten
        for dwt_sub_ax, tf_idx in zip(dwt_sub_axs, range(4)):
            dwt_sub_ax.imshow(transforms[..., tf_idx::4])
            plt.sca(dwt_sub_ax)
            plt.xticks([])
            plt.yticks([])
            dwt_subfig.suptitle(butterflies_data.classes[label])

    plt.show()
