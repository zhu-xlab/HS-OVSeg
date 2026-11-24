#!/usr/bin/env python3
"""Download and prepare all public EarthVLSet semantic annotations."""

import _bootstrap  # noqa: F401

from ovrs_segmentation.preprocessing.global_loveda import main


if __name__ == "__main__":
    main()
