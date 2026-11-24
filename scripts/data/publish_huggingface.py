"""Publish the selected data to a private Hugging Face dataset repository."""
import _bootstrap  # noqa: F401
from ovrs_segmentation.preprocessing.huggingface_publish import main


if __name__ == "__main__":
    main()
