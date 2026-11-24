"""Export the selected train/val/test data without LMDB or source archives."""
import _bootstrap  # noqa: F401
from ovrs_segmentation.preprocessing.huggingface_export import main


if __name__ == "__main__":
    main()
