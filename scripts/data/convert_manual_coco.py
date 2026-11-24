#!/usr/bin/env python
"""Convert reviewed COCO annotation workspaces to segmentation test datasets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ovrs_segmentation.preprocessing.manual_coco import main


if __name__ == "__main__":
    main()
