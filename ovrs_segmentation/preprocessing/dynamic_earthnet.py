"""Prepare one labeled DynamicEarthNet date from the authors' range-readable ZIPs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image


BASE = 'https://dataserv.ub.tum.de/public.php/dav/files/m1650201/'
CLASSES = ('impervious surface', 'agriculture', 'forest and other vegetation',
           'wetland', 'soil', 'water', 'snow and ice')


def fetch_range(url, start, end):
    for attempt in range(4):
        try:
            # Some intermediary caches ignore Range unless the URL distinguishes each byte span.
            range_url = url + ('&' if '?' in url else '?') + f'ovrs_range={start}-{end}'
            request = urllib.request.Request(range_url, headers={
                'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError('Server ignored Range; refusing a full archive download')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if not match or int(match[1]) != start or int(match[2]) != end:
                    raise ValueError('Unexpected HTTP Content-Range')
                data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError('Truncated range response')
                return data, int(match[3])
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        _, self.size = fetch_range(url, 0, 1)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0 or whence not in (0, 1, 2):
            raise ValueError('Invalid ZIP seek')
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b''
        if count > 64 * 1024 * 1024:
            raise ValueError('Refusing unexpectedly large ZIP member read')
        data, total = fetch_range(self.url, self.position, self.position + count - 1)
        if total != self.size:
            raise ValueError('Archive size changed while downloading')
        self.position += len(data)
        return data


def selected_members(archive, date):
    result = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if not (path.name == date + '.tif' or path.name.endswith('-' + date + '.tif')):
            continue
        if path.is_absolute() or '..' in path.parts or len(path.parts) < 2:
            raise ValueError('Unsafe ZIP member path')
        if 'PF-QA' in path.parts:
            continue
        cubes = [p.replace('-', '_') for p in path.parts[:-1]
                 if re.fullmatch(r'\d+_\d+_\d+[-_]\d+[NS]', p)]
        if not cubes and 'PF-SR' in path.parts:
            ids = [p for p in path.parts if re.fullmatch(r'\d+_\d+_\d+', p)]
            zones = [p for p in path.parts if re.fullmatch(r'\d+[NS]', p)]
            if len(ids) == len(zones) == 1:
                cubes = [ids[0] + '_' + zones[0]]
        if len(cubes) != 1:
            raise ValueError(f'Unexpected AOI path: {path}')
        cube = cubes[0]
        if cube in result or info.file_size > 40 * 1024 * 1024:
            raise ValueError('Duplicate AOI or unexpected TIFF size')
        result[cube] = info.filename
    return result


def decode_onehot(label):
    if label.ndim != 3 or label.shape[0] != 7 or not np.isin(label, (0, 255)).all():
        raise ValueError('Expected seven binary 0/255 label bands')
    positive = label == 255
    valid = positive.sum(axis=0) == 1
    return np.where(valid, positive.argmax(axis=0), 255).astype(np.uint8)


def rgb_uint8(bands):
    if bands.ndim != 3 or bands.shape[0] != 4 or not np.isfinite(bands).all():
        raise ValueError('Expected finite B,G,R,NIR Planet reflectance bands')
    # A fixed, GT-independent 0--0.3 reflectance display range, identical for every method.
    return np.round(np.clip(bands[[2, 1, 0]].astype(np.float32) / 3000, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def tiles(rgb, mask, size=518):
    if rgb.shape[:2] != mask.shape:
        raise ValueError('Image and label grids differ')
    for top in range(0, mask.shape[0], size):
        for left in range(0, mask.shape[1], size):
            h, w = min(size, mask.shape[0] - top), min(size, mask.shape[1] - left)
            image = np.full((size, size, 3), 128, dtype=np.uint8)
            target = np.full((size, size), 255, dtype=np.uint8)
            image[:h, :w] = rgb[top:top+h, left:left+w]
            target[:h, :w] = mask[top:top+h, left:left+w]
            if np.any(target != 255):
                yield top, left, image, target


def prepare(output, date='2018-07-01', max_samples=2000, workers=4, inventory=False):
    if not re.fullmatch(r'201[89]-(0[1-9]|1[0-2])-01', date) or not 4 <= max_samples <= 2000 or not 1 <= workers <= 8:
        raise ValueError('Use a monthly annotation date, 4--2000 samples and 1--8 workers')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    with HTTPRangeFile(BASE + 'labels.zip') as stream, zipfile.ZipFile(stream) as archive:
        labels = selected_members(archive, date)
    if not labels:
        raise ValueError('No labels for selected date')
    cubes = sorted(labels)[:max_samples // 4]
    zones = sorted({c.rsplit('_', 1)[-1] for c in cubes})
    plan = dict(date=date, labeled_aois=len(labels), selected_aois=len(cubes), zones=zones,
                maximum_prepared_samples=len(cubes)*4, source_url=BASE)
    print(json.dumps(plan), flush=True)
    if inventory:
        return plan

    from rasterio.io import MemoryFile
    from ovrs_segmentation.datasets import SemanticSegmentationDataset, write_class_catalog
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.dynamic-earthnet-') as temporary:
        stage = Path(temporary) / output.name
        for folder in ('images', 'masks', 'splits'):
            (stage / folder).mkdir(parents=True)

        def zone_records(zone):
            records = []
            with HTTPRangeFile(BASE + f'planet.{zone}.zip') as images_file, zipfile.ZipFile(images_file) as images, \
                    HTTPRangeFile(BASE + 'labels.zip') as labels_file, zipfile.ZipFile(labels_file) as label_zip:
                available = selected_members(images, date)
                for cube in [c for c in cubes if c.endswith('_' + zone)]:
                    if cube not in available:
                        raise ValueError(f'Missing same-date image for {cube}')
                    image_bytes, label_bytes = images.read(available[cube]), label_zip.read(labels[cube])
                    with MemoryFile(image_bytes) as memory, memory.open() as image, \
                            MemoryFile(label_bytes) as label_memory, label_memory.open() as label:
                        if (image.width, image.height) != (1024, 1024) or image.shape != label.shape:
                            raise ValueError('Unexpected source grid')
                        if image.crs != label.crs or not image.transform.almost_equals(label.transform):
                            raise ValueError('Image and GT georeferencing differ')
                        if not np.allclose((abs(image.res[0]), abs(image.res[1])), (3, 3), atol=.01):
                            raise ValueError('Expected native 3 m pixels')
                        rgb, mask = rgb_uint8(image.read()), decode_onehot(label.read())
                        mask[image.dataset_mask() == 0] = 255
                    tile_ids = []
                    counts = np.zeros(7, dtype=np.int64)
                    for top, left, tile, target in tiles(rgb, mask):
                        sample_id = f'{cube}_{date.replace("-", "")}_y{top}_x{left}'
                        Image.fromarray(tile).save(stage / 'images' / (sample_id+'.png'))
                        Image.fromarray(target).save(stage / 'masks' / (sample_id+'.png'))
                        counts += np.bincount(target[target != 255], minlength=7)
                        tile_ids.append(sample_id)
                    records.append(dict(cube=cube, date=date, samples=tile_ids, class_pixel_counts=counts.tolist(),
                        image_member=available[cube], label_member=labels[cube],
                        image_sha256=hashlib.sha256(image_bytes).hexdigest(), label_sha256=hashlib.sha256(label_bytes).hexdigest()))
                    print(f'[DynamicEarthNet] {cube}: {len(tile_ids)} valid tiles', flush=True)
            return records

        with ThreadPoolExecutor(workers) as pool:
            records = [row for group in pool.map(zone_records, zones) for row in group]
        records.sort(key=lambda row: row['cube'])
        ids = [i for row in records for i in row['samples']]
        if not ids or len(ids) > max_samples or len(ids) != len(set(ids)):
            raise ValueError('Invalid prepared sample count')
        write_class_catalog(stage/'classes.json', 'DynamicEarthNet', dict(enumerate(CLASSES)),
                            default_semantic_level='lulc', background_id=None, ignore_id=255)
        (stage/'splits/test.txt').write_text('\n'.join(ids)+'\n')
        metadata = dict(format_version='ovrs-semantic-v1', dataset_name='DynamicEarthNet', dataset_id='DynamicEarthNet',
            image_dir='images', mask_dir='masks', classes_file='classes.json', background_id=None, ignore_id=255,
            semantic_level='lulc', supervision_level='pixel', usage='out_of_domain_test_only', splits={'test':len(ids)},
            source_url='https://mediatum.ub.tum.de/1650201', source_download_base=BASE, date=date,
            source_aois=len(records), native_gsd_m=3, original_size=[1024,1024], prepared_size=[518,518],
            original_split='Publicly available author label archive; held out entirely from project adaptation',
            rgb_bands_one_based=[3,2,1], rgb_uint8_policy='clip reflectance DN to [0,3000], divide by 3000, round times 255',
            label_policy='Seven native one-hot bands; zero/multiple positive labels and padded pixels ignored',
            tile_policy='518 non-overlapping, right/bottom padded with RGB128 and ignore255; no resampling',
            class_pixel_counts={str(i):int(v) for i,v in enumerate(np.sum([x['class_pixel_counts'] for x in records],axis=0))})
        (stage/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (stage/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
        dataset = SemanticSegmentationDataset(stage, 'test', validate_masks=True)
        for sample in dataset:
            if sample['image'].shape != (518,518,3) or sample['class_id_mask'].shape != (518,518):
                raise ValueError('Prepared sample shape mismatch')
        stage.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--date', default='2018-07-01')
    parser.add_argument('--max-samples', default=2000, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.date, args.max_samples, args.workers, args.inventory),indent=2),flush=True)


if __name__ == '__main__':
    main()
