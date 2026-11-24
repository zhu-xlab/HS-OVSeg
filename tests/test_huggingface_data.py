import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import yaml

from ovrs_segmentation.datasets import open_text_prompt_dataset, text_prompt_collate
from ovrs_segmentation.datasets.class_prompts import build_class_catalog
from ovrs_segmentation.datasets.semantic_dataset import SemanticSegmentationDataset
from ovrs_segmentation.preprocessing.huggingface_data import (
    dataset_entries, download_datasets, extract_archive, validate_datasets,
)
from ovrs_segmentation.preprocessing.huggingface_export import export_dataset, archive_dataset
from ovrs_segmentation.evaluation.segmentation_test import evaluate_segmentation
from types import SimpleNamespace
import torch


def prepared(root, splits):
    for directory in ("images", "masks", "splits"):
        (root / directory).mkdir(parents=True)
    Image.new("RGB", (3, 2), "green").save(root / "images/a.png")
    Image.fromarray(np.ones((2, 3), np.uint8)).save(root / "masks/a.png")
    for split in splits:
        (root / "splits" / f"{split}.txt").write_text("a\n")
    (root / "dataset.json").write_text(json.dumps(dict(
        format_version="ovrs-semantic-v1", dataset_name="Toy", semantic_level="lulc",
        supervision_level="pixel", background_id=0, ignore_id=255)))
    (root / "classes.json").write_text(json.dumps(build_class_catalog(
        "Toy", {0: "background", 1: "tree"}, default_semantic_level="lulc", background_id=0)))


class HuggingFaceDataTest(unittest.TestCase):
    def test_sharded_download_restores_training_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); remote = root / "hub"; local = root / "download"
            prepared(root / "source", ("train", "val"))
            shards = archive_dataset(root / "source", "Toy", remote / "archives", shard_bytes=1024)
            self.assertGreater(len(shards), 1)
            suite = root / "train.yaml"
            suite.write_text(yaml.safe_dump({"datasets": [dict(name="Toy", source_task="lulc",
                root="{data_root}/Toy")]}))
            def download(**kwargs):
                self.assertIn("archives/Toy/*.tar", kwargs["allow_patterns"])
                shutil.copytree(remote, kwargs["local_dir"])
            with patch("huggingface_hub.snapshot_download", side_effect=download):
                report = download_datasets("owner/data", local, mode="train", extract=True,
                    train_suite=suite)
            self.assertTrue(report["validated"])
            self.assertEqual(len(open_text_prompt_dataset(local / "Toy", "train")), 1)

    def test_publication_uploads_only_selected_tar_shards_and_skips_verified_uploads(self):
        import hashlib
        from ovrs_segmentation.preprocessing.huggingface_publish import publish
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"; work = root / "work"
            prepared(source / "Toy", ("train", "val"))
            entries = [dict(name="Toy", root=source / "Toy", split=split, reader="training",
                            source_task="lulc") for split in ("train", "val")]
            uploaded, calls = {}, []
            class API:
                def repo_info(self, *args, **kwargs):
                    return SimpleNamespace(private=True)
                def get_paths_info(self, repo, paths, **kwargs):
                    return [SimpleNamespace(lfs=SimpleNamespace(sha256=uploaded[paths[0]]))] if paths[0] in uploaded else []
                def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
                    uploaded[path_in_repo] = hashlib.sha256(Path(path_or_fileobj).read_bytes()).hexdigest()
                    calls.append(path_in_repo)
            with patch("huggingface_hub.HfApi", API), patch(
                    "ovrs_segmentation.preprocessing.huggingface_publish.dataset_entries", return_value=entries):
                publish("owner/data", source, work)
                self.assertEqual(calls, ["archives/Toy/part-00000.tar", "data-manifest.json"])
                calls.clear()
                publish("owner/data", source, work)
                self.assertEqual(calls, ["data-manifest.json"])
            self.assertEqual(json.loads((work / "publication-status.json").read_text())["phase"], "complete")
    def test_export_keeps_only_selected_samples_once_and_unpacks_lmdb(self):
        from ovrs_segmentation.preprocessing.lmdb_dataset import pack_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            prepared(source, ("train", "val", "test"))
            # Unselected images, source archives, and backups must never be exported.
            Image.new("RGB", (3, 2), "red").save(source / "images/unused.png")
            Image.new("L", (3, 2), 1).save(source / "masks/unused.png")
            (source / "original.zip").write_bytes(b"unused")
            (source / "classes.backup.json").write_text("{}")
            pack_dataset(source, remove_source_files=True)
            metadata = json.loads((source / "dataset.json").read_text())
            metadata.update(source_path="/private/server/data", optional_modalities={})
            (source / "dataset.json").write_text(json.dumps(metadata))
            destination = root / "published"
            result = export_dataset(source, destination, ["train", "val"],
                                    reader="training", level="lulc")
            self.assertEqual(result["unique_samples"], 1)
            files = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
            self.assertEqual(files, {"dataset.json", "classes.json", "splits/train.txt",
                "splits/val.txt", "images/a.png", "masks/a.png"})
            self.assertNotIn("storage", json.loads((destination / "dataset.json").read_text()))
            self.assertNotIn("/private", (destination / "dataset.json").read_text())
            sample = open_text_prompt_dataset(destination, "train")[0]
            np.testing.assert_array_equal(sample["image"], np.full((2, 3, 3), [0, 128, 0]))

    def test_scene_export_filters_annotation_rows_and_keeps_portable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"; source.mkdir()
            (source / "images").mkdir(); (source / "splits").mkdir()
            for name in ("a", "unused"):
                Image.new("RGB", (3, 2), "green").save(source / "images" / f"{name}.png")
            (source / "splits/train.txt").write_text("a\n")
            (source / "dataset.json").write_text(json.dumps(dict(dataset_name="ToyScene",
                semantic_level="scene", supervision_level="image", annotations_file="annotations.jsonl")))
            (source / "classes.json").write_text(json.dumps(build_class_catalog(
                "ToyScene", {1: "forest"}, default_semantic_level="scene")))
            rows = [dict(sample_id=name, image=f"images/{name}.png", annotations=[dict(
                dataset_class_id=1, semantic_level="scene", supervision_level="image", present=True)])
                for name in ("a", "unused")]
            (source / "annotations.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            destination = root / "published"
            export_dataset(source, destination, ["train"], reader="training", level="scene")
            self.assertEqual(len((destination / "annotations.jsonl").read_text().splitlines()), 1)
            self.assertFalse((destination / "images/unused.png").exists())
            self.assertEqual(len(open_text_prompt_dataset(destination, "train")), 1)

    def test_object_export_keeps_boxes_but_omits_unused_pixel_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"
            prepared(source, ("train", "val"))
            metadata = json.loads((source / "dataset.json").read_text())
            metadata.update(semantic_level="object", supervision_level="box", annotations_dir="boxes")
            (source / "dataset.json").write_text(json.dumps(metadata))
            (source / "boxes").mkdir()
            (source / "boxes/a.json").write_text(json.dumps(dict(sample_id="a",
                annotations=[dict(dataset_class_id=1, bbox_xyxy=[0, 0, 2, 1])])))
            destination = root / "published"
            export_dataset(source, destination, ["train", "val"], reader="training", level="object")
            self.assertFalse((destination / "masks").exists())
            self.assertTrue((destination / "boxes/a.json").is_file())
            sample = open_text_prompt_dataset(destination, "train", level="object")[0]
            self.assertEqual(sample["annotations"][0]["boxes_xyxy"], [[0., 0., 2., 1.]])

    def test_downloaded_tree_feeds_training_collation_and_semantic_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            remote, local = base / "remote", base / "download"
            prepared(remote / "Toy", ("train", "val"))
            prepared(remote / "segmentation_test/Toy", ("test",))
            train, test = base / "train.yaml", base / "test.yaml"
            train.write_text(yaml.safe_dump({"datasets": [dict(name="Toy", source_task="lulc",
                root="{data_root}/Toy")]}))
            test.write_text(yaml.safe_dump({"datasets": [dict(name="Toy", source_task="lulc",
                root="{test_root}/Toy", split="test")]}))

            def download(**kwargs):
                self.assertEqual(kwargs["repo_type"], "dataset")
                self.assertEqual(kwargs["revision"], "v1")
                self.assertEqual(set(kwargs["allow_patterns"]), {"Toy/**", "segmentation_test/Toy/**"})
                self.assertNotIn("token", kwargs)
                shutil.copytree(remote, kwargs["local_dir"])
                return kwargs["local_dir"]

            with patch("huggingface_hub.snapshot_download", side_effect=download):
                report = download_datasets("owner/data", local, revision="v1",
                    train_suite=train, test_suite=test)
            self.assertEqual([row["split"] for row in report["splits"]], ["train", "val", "test"])
            dataset = open_text_prompt_dataset(local / "Toy", split="train")
            batch = text_prompt_collate([dataset[0]], spatial_size=4)
            self.assertEqual(tuple(batch["images"].shape), (1, 3, 4, 4))
            dataset = SemanticSegmentationDataset(local / "segmentation_test/Toy", split="test")
            with patch("ovrs_segmentation.evaluation.segmentation_test._forward_prompt_chunks",
                       return_value={"unified_pixel_logits": torch.ones(1, 2, 3)}):
                result, _ = evaluate_segmentation(SimpleNamespace(sam3_input_size=(4, 4)),
                    [dataset[0]], dataset, torch.device("cpu"), "fp32")
            self.assertEqual(result["mean_iou"], 1.)

    def test_modes_keep_manual_test_data_out_of_training(self):
        train = dataset_entries("data", "train")
        test = dataset_entries("data", "test", scope="training_source")
        self.assertEqual(len(train), 40)
        self.assertEqual(len(test), 20)
        self.assertTrue(all(row["split"] in ("train", "val") for row in train))
        self.assertTrue(all("segmentation_test" not in row["root"].parts for row in train))
        self.assertTrue(all("segmentation_test" in row["root"].parts for row in test))
        self.assertEqual(len(dataset_entries("data", "all")), 60)

    def test_incomplete_download_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Toy"
            prepared(root, ("train",))
            (root / "images/a.png").unlink()
            with self.assertRaisesRegex(ValueError, "not ready"):
                validate_datasets([dict(name="Toy", source_task="lulc", root=root,
                    split="train", reader="training")])

    def test_zip_and_tar_preserve_expected_dataset_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "Toy.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("Toy/splits/train.txt", "a\n")
            extract_archive(archive, root / "zip")
            self.assertEqual((root / "zip/Toy/splits/train.txt").read_text(), "a\n")
            archive = root / "Toy.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                data = b"a\n"
                member = tarfile.TarInfo("segmentation_test/Toy/splits/test.txt")
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
            extract_archive(archive, root / "tar")
            self.assertEqual((root / "tar/segmentation_test/Toy/splits/test.txt").read_text(), "a\n")

    def test_archives_reject_escape_paths_and_links_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../escape", "/absolute", "C:/absolute", "..\\escape"):
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr(name, "bad")
                with self.assertRaises(ValueError):
                    extract_archive(archive, root / "output")
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as stream:
                member = tarfile.TarInfo("link"); member.type = tarfile.SYMTYPE
                member.linkname = "../outside"; stream.addfile(member)
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
