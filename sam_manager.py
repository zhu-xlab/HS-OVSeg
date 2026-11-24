import base64
from contextlib import nullcontext
import json
import os
import traceback
import urllib.error
import urllib.request
import zlib

import numpy as np
from PIL import Image


class SAMManager:
    def __init__(self, remote_url=None):
        self.sam_predictor = None
        self.sam3_processor = None
        self.sam3_model = None
        self.sam3_state = None
        self.sam3_backend = None
        self.sam3_concept = None
        self.sam3_probabilities = None
        self.sam3_scores = None
        self.last_probability_range = None
        self._model_loaded = False
        self.current_image = None
        self.current_image_shape = None
        self.current_image_key = None
        self.image_cache = {}
        self.last_error = None
        # ViT-B is the practical default for CPU-only annotation machines.
        self.model_type = "vit_b"
        self.torch = None
        self.device = "cpu"
        self.preferred_device = os.environ.get("SAM_DEVICE", "auto").lower()
        self.strategy = "sam"
        self.remote_url = (
            os.environ.get("SAM_REMOTE_URL", "").rstrip("/")
            if remote_url is None
            else remote_url.rstrip("/")
        )

    def is_initialized(self):
        if self.remote_url:
            return self._model_loaded
        predictor = self.sam_predictor if self.strategy == "sam" else self.sam3_processor
        return self._model_loaded and predictor is not None

    def configure_backend(self, remote_url="", device="auto"):
        remote_url, device = remote_url.rstrip("/"), (device or "auto").lower()
        if remote_url == self.remote_url and device == self.preferred_device:
            return
        self.unload_model()
        self.remote_url, self.preferred_device = remote_url, device

    def initialize(self, strategy="sam", checkpoint_path=None, device=None):
        strategy = strategy.lower()
        if strategy not in {"sam", "sam3"}:
            self.last_error = f"Unsupported prompt strategy: {strategy}"
            return False
        requested_device = (device or self.preferred_device or "auto").lower()
        if requested_device != self.preferred_device:
            self.unload_model()
            self.preferred_device = requested_device
        if self.remote_url:
            already_loaded = self._model_loaded and self.strategy == strategy
            try:
                result = self._remote_request(
                    "/initialize", {"strategy": strategy, "device": requested_device}
                )
                self.strategy = strategy
                self.device = result.get("device", "remote")
                self._model_loaded = True
                if self.current_image is not None and not already_loaded:
                    self._send_remote_image()
                return True
            except Exception as exc:
                self.last_error = f"Remote initialization failed: {exc}"
                self._model_loaded = False
                return False
        if self.strategy != strategy:
            self.unload_model()
            self.strategy = strategy
        if strategy == "sam3":
            return self._init_sam3_model(checkpoint_path)
        return self._init_sam_model(checkpoint_path)

    def _select_device(self, torch):
        requested = self.preferred_device
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            requested = "cuda:0"
        if requested.startswith("cuda:") and requested[5:].isdigit():
            index = int(requested[5:])
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU {index} is unavailable; detected {torch.cuda.device_count()} CUDA device(s)"
                )
            torch.cuda.set_device(index)
            return requested
        raise ValueError("Device must be auto, cpu, cuda, or cuda:N")

    def _init_sam_model(self, checkpoint_path=None):
        if self._model_loaded:
            return True

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            self.torch = torch
            self.device = self._select_device(torch)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_names = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth",
            }
            default_checkpoint = os.path.join(
                base_dir, "checkpoints", checkpoint_names[self.model_type]
            )
            checkpoint_path = checkpoint_path or default_checkpoint

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
            if self.model_type not in sam_model_registry:
                raise ValueError(f"Unsupported SAM model type: {self.model_type}")

            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam.to(device=torch.device(self.device))
            self.sam_predictor = SamPredictor(sam)
            self._model_loaded = True

            if self.current_image is not None:
                self.sam_predictor.set_image(self.current_image, image_format="RGB")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.sam_predictor = None
            self._model_loaded = False
            traceback.print_exc()
            return False

    def _init_sam3_model(self, checkpoint_path=None):
        if self._model_loaded and self.sam3_processor is not None:
            return True

        try:
            import torch

            self.torch = torch
            self.device = self._select_device(torch)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_checkpoint = checkpoint_path or os.path.join(base_dir, "checkpoints", "sam3.pt")
            if os.path.isfile(local_checkpoint):
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model

                self.sam3_model = build_sam3_image_model(
                    checkpoint_path=local_checkpoint,
                    device=self.device,
                    load_from_HF=False,
                )
                # The upstream builder treats only the literal "cuda" as a full-device move.
                # Explicit cuda:N selection can otherwise leave submodules on CPU.
                self.sam3_model.to(self.device).eval()
                self.sam3_processor = Sam3Processor(self.sam3_model, device=self.device)
                self.sam3_backend = "meta"
            else:
                from transformers import Sam3Model, Sam3Processor

                model_id = checkpoint_path or os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
                load_options = {"low_cpu_mem_usage": True}
                if self.device.startswith("cuda"):
                    load_options.update({"device_map": {"": self.device}, "dtype": torch.float16})
                self.sam3_processor = Sam3Processor.from_pretrained(model_id)
                self.sam3_model = Sam3Model.from_pretrained(model_id, **load_options)
                if self.device == "cpu":
                    self.sam3_model.to("cpu")
                self.sam3_model.eval()
                self.sam3_backend = "transformers"
            self._model_loaded = True
            if self.current_image is not None:
                self._set_sam3_image()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.sam3_processor = None
            self.sam3_model = None
            self.sam3_state = None
            self.sam3_backend = None
            self._model_loaded = False
            traceback.print_exc()
            return False

    def set_image(self, image, cache_key=None):
        self.last_error = None
        self.sam3_concept = None
        self.sam3_probabilities = None
        self.sam3_scores = None
        self.last_probability_range = None
        if image is None:
            self.current_image = None
            self.current_image_shape = None
            self.current_image_key = None
            self.image_cache.clear()
            if self.is_initialized() and self.strategy == "sam":
                self.sam_predictor.reset_image()
            self.sam3_state = None
            return

        try:
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Invalid RGB image")
            self.current_image = image.copy()
            self.current_image_shape = image.shape[:2]
            self.current_image_key = str(cache_key) if cache_key else None
            if self.remote_url and self.is_initialized():
                self._send_remote_image(self.current_image_key)
                if self.current_image_key:
                    self.image_cache[self.current_image_key] = {
                        "shape": tuple(self.current_image_shape), "state": None
                    }
                return
            if self.is_initialized() and self.strategy == "sam":
                self.sam_predictor.set_image(self.current_image, image_format="RGB")
            elif self.is_initialized():
                self._set_sam3_image()
            if self.current_image_key and self.is_initialized():
                self._cache_current_image()
        except Exception as exc:
            self.last_error = f"Image processing error: {exc}"
            traceback.print_exc()

    def _remote_request(self, path, payload):
        request = urllib.request.Request(
            self.remote_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(detail or str(exc)) from exc
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Unknown remote inference error"))
        return result

    def _send_remote_image(self, cache_key=None):
        import cv2

        success, encoded = cv2.imencode(
            ".png", cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise ValueError("Could not encode image for remote inference")
        payload = {"image": base64.b64encode(encoded).decode("ascii")}
        if cache_key:
            payload["image_key"] = cache_key
        self._remote_request("/image", payload)

    def _remote_prediction(self, payload):
        if self.current_image_key:
            payload = {**payload, "image_key": self.current_image_key}
        result = self._remote_request("/predict", payload)
        self.last_probability_range = result.get("probability_range")
        shape = tuple(result["shape"])
        raw = zlib.decompress(base64.b64decode(result["masks"]))
        count = int(np.prod(shape))
        masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=count).reshape(shape)
        return masks.astype(np.bool_), np.asarray(result["scores"], dtype=np.float32)

    def _move_tensors(self, value, device):
        if self.torch is not None and self.torch.is_tensor(value):
            return value.detach().to(device)
        if isinstance(value, dict):
            return {key: self._move_tensors(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_tensors(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_tensors(item, device) for item in value)
        return value

    def _cache_current_image(self):
        shape = tuple(self.current_image_shape)
        if self.strategy == "sam":
            state = {
                "features": self.sam_predictor.features.detach().cpu(),
                "input_size": tuple(self.sam_predictor.input_size),
                "original_size": tuple(self.sam_predictor.original_size),
            }
        elif self.sam3_backend == "meta":
            state = self._move_tensors(self.sam3_state, "cpu")
        else:
            # ponytail: transformers SAM3 has no exposed reusable image state;
            # retain the patch image until that backend gains a feature API.
            state = Image.fromarray(self.current_image.astype(np.uint8), "RGB")
        self.image_cache[self.current_image_key] = {"shape": shape, "state": state}

    def activate_image(self, cache_key):
        if not cache_key:
            return
        cache_key = str(cache_key)
        if cache_key == self.current_image_key:
            return
        self.current_image_key = cache_key
        if self.remote_url:
            cached = self.image_cache.get(cache_key)
            if cached is None:
                raise ValueError(f"Remote image feature cache not found: {cache_key}")
            self.current_image_shape = cached["shape"]
            return
        cached = self.image_cache.get(cache_key)
        if cached is None:
            raise ValueError(f"Image feature cache not found: {cache_key}")
        self.current_image_shape = cached["shape"]
        self.sam3_concept = None
        self.sam3_probabilities = None
        self.sam3_scores = None
        self.last_probability_range = None
        if self.strategy == "sam":
            self.sam_predictor.features = cached["state"]["features"].to(self.device)
            self.sam_predictor.input_size = cached["state"]["input_size"]
            self.sam_predictor.original_size = cached["state"]["original_size"]
            self.sam_predictor.is_image_set = True
        elif self.sam3_backend == "meta":
            self.sam3_state = self._move_tensors(cached["state"], self.device)
        else:
            self.sam3_state = cached["state"]

    def _set_sam3_image(self):
        image = Image.fromarray(self.current_image.astype(np.uint8), "RGB")
        if self.sam3_backend == "meta":
            precision = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if self.device.startswith("cuda")
                else nullcontext()
            )
            with precision:
                self.sam3_state = self.sam3_processor.set_image(image)
        else:
            self.sam3_state = image

    def predict_mask(self, points, labels, multimask_output=True):
        self.last_error = None
        if self.strategy != "sam" or not self.is_initialized():
            self.last_error = "SAM predictor is not initialized"
            return None, None
        if self.remote_url:
            try:
                return self._remote_prediction(
                    {"kind": "point", "points": points, "labels": labels}
                )
            except Exception as exc:
                self.last_error = f"Remote point prediction failed: {exc}"
                return None, None
        if len(points) != len(labels):
            self.last_error = f"Points ({len(points)}) and labels ({len(labels)}) length mismatch"
            return None, None
        if not points:
            self.last_error = "No prompt points provided"
            return None, None
        if self.current_image_shape is None:
            self.last_error = "No image is loaded for SAM"
            return None, None

        try:
            points_array = np.asarray(points, dtype=np.float32)
            labels_array = np.asarray(labels, dtype=np.int32)
            img_h, img_w = self.current_image_shape
            out_of_bounds = (
                (points_array[:, 0] < 0).any()
                or (points_array[:, 0] >= img_w).any()
                or (points_array[:, 1] < 0).any()
                or (points_array[:, 1] >= img_h).any()
            )
            if out_of_bounds:
                self.last_error = f"Prompt point outside image bounds ({img_w}x{img_h})"
                return None, None

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=multimask_output,
            )
            masks = masks.astype(np.bool_)
            if masks.shape[1:] != (img_h, img_w):
                self.last_error = f"Mask size mismatch: expected {img_h}x{img_w}, got {masks.shape[1:]}"
                return None, None
            return masks, scores
        except RuntimeError as exc:
            self.last_error = f"Runtime error: {exc}"
            if "CUDA out of memory" in str(exc):
                self.last_error += "\nTry a smaller image or a smaller SAM checkpoint."
            return None, None
        except Exception as exc:
            self.last_error = f"Prediction failed: {exc}"
            traceback.print_exc()
            return None, None

    def predict_box(self, box, multimask_output=True):
        """Predict SAM masks from an XYXY image-coordinate box."""
        self.last_error = None
        if self.strategy != "sam" or not self.is_initialized():
            self.last_error = "SAM predictor is not initialized"
            return None, None
        if self.current_image_shape is None:
            self.last_error = "No image is loaded for SAM"
            return None, None
        if self.remote_url:
            try:
                return self._remote_prediction({"kind": "box", "box": list(box)})
            except Exception as exc:
                self.last_error = f"Remote box prediction failed: {exc}"
                return None, None

        try:
            box_array = np.asarray(box, dtype=np.float32)
            if box_array.shape != (4,):
                raise ValueError("Box must contain x1, y1, x2, y2")
            img_h, img_w = self.current_image_shape
            box_array[[0, 2]] = np.clip(box_array[[0, 2]], 0, img_w - 1)
            box_array[[1, 3]] = np.clip(box_array[[1, 3]], 0, img_h - 1)
            if box_array[2] <= box_array[0] or box_array[3] <= box_array[1]:
                raise ValueError("Bounding box is empty")
            masks, scores, _ = self.sam_predictor.predict(
                box=box_array,
                multimask_output=multimask_output,
            )
            return masks.astype(np.bool_), scores
        except Exception as exc:
            self.last_error = f"Box prediction failed: {exc}"
            traceback.print_exc()
            return None, None

    def predict_boxes(self, boxes):
        """Predict one best SAM mask for every XYXY box, preserving box order."""
        self.last_error = None
        if self.strategy != "sam" or not self.is_initialized():
            self.last_error = "SAM predictor is not initialized"
            return None, None
        if self.current_image_shape is None:
            self.last_error = "No image is loaded for SAM"
            return None, None
        if not boxes:
            shape = (0, *self.current_image_shape)
            return np.empty(shape, dtype=np.bool_), np.empty(0, dtype=np.float32)
        if self.remote_url:
            try:
                return self._remote_prediction({"kind": "boxes", "boxes": boxes})
            except Exception as exc:
                self.last_error = f"Remote box prediction failed: {exc}"
                return None, None

        masks, scores = [], []
        for box in boxes:
            box_masks, box_scores = self.predict_box(box, multimask_output=False)
            if box_masks is None:
                return None, None
            masks.append(box_masks[0])
            scores.append(box_scores[0])
        return np.asarray(masks, dtype=np.bool_), np.asarray(scores, dtype=np.float32)

    def predict_concept(self, concept, probability_min=None, probability_max=None):
        """Return all SAM3 masks matching a text concept."""
        self.last_error = None
        concept = (concept or "").strip()
        if self.strategy != "sam3" or not self.is_initialized():
            self.last_error = "SAM3 processor is not initialized"
            return None, None
        if not concept:
            self.last_error = "Concept prompt cannot be empty"
            return None, None
        if (probability_min is None) != (probability_max is None):
            self.last_error = "Probability min and max must be supplied together"
            return None, None
        if probability_min is not None:
            probability_min, probability_max = float(probability_min), float(probability_max)
            if not 0 <= probability_min < probability_max <= 1:
                self.last_error = "Probability range must satisfy 0 <= min < max <= 1"
                return None, None
        if self.sam3_state is None:
            if not self.remote_url or self.current_image_shape is None:
                self.last_error = "No image is loaded for SAM3"
                return None, None

        if self.remote_url:
            try:
                payload = {"kind": "concept", "concept": concept}
                if probability_min is not None:
                    payload.update({"probability_min": probability_min, "probability_max": probability_max})
                return self._remote_prediction(payload)
            except Exception as exc:
                self.last_error = f"Remote concept prediction failed: {exc}"
                return None, None

        try:
            if self.sam3_backend == "meta":
                if self.sam3_concept != concept or self.sam3_probabilities is None:
                    precision = (
                        self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                        if self.device.startswith("cuda")
                        else nullcontext()
                    )
                    with precision:
                        result = self.sam3_processor.set_text_prompt(concept, self.sam3_state)
                    self.sam3_state = result
                    self.sam3_concept = concept
                    self.sam3_probabilities = result["masks_logits"]
                    self.sam3_scores = result["scores"]
                probabilities, scores = self.sam3_probabilities, self.sam3_scores
                if probabilities.numel():
                    observed_min = float(probabilities.min().item())
                    observed_max = float(probabilities.max().item())
                    window_min = observed_min if probability_min is None else probability_min
                    window_max = observed_max if probability_max is None else probability_max
                    if window_max <= window_min:
                        window_min, window_max = max(0.0, window_min - 1e-6), min(1.0, window_max + 1e-6)
                    if window_max <= window_min:
                        window_min, window_max = 0.0, 1.0
                    masks = ((probabilities - window_min) / (window_max - window_min)).clamp(0, 1) > 0.5
                else:
                    observed_min, observed_max = 0.0, 1.0
                    window_min = observed_min if probability_min is None else probability_min
                    window_max = observed_max if probability_max is None else probability_max
                    masks = probabilities.bool()
                self.last_probability_range = {
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "window_min": window_min,
                    "window_max": window_max,
                }
                masks = masks.detach().bool().cpu().numpy()
                if hasattr(scores, "detach"):
                    scores = scores.detach().float().cpu().numpy()
                masks = np.asarray(masks, dtype=np.bool_)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                return masks, np.asarray(scores)

            inputs = self.sam3_processor(
                images=self.sam3_state,
                text=concept,
                return_tensors="pt",
            ).to(self.sam3_model.device)
            with self.torch.inference_mode():
                output = self.sam3_model(**inputs)
            result = self.sam3_processor.post_process_instance_segmentation(
                output,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            masks = result["masks"]
            scores = result.get("scores", np.ones(len(masks), dtype=np.float32))
            if hasattr(masks, "detach"):
                masks = masks.detach().bool().cpu().numpy()
            if hasattr(scores, "detach"):
                scores = scores.detach().float().cpu().numpy()
            masks = np.asarray(masks, dtype=np.bool_)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            return masks, np.asarray(scores)
        except Exception as exc:
            self.last_error = f"Concept prediction failed: {exc}"
            traceback.print_exc()
            return None, None

    def get_last_error(self):
        return self.last_error

    def clear_image_cache(self):
        if self.remote_url and self._model_loaded:
            try:
                self._remote_request("/cache/clear", {})
            except RuntimeError as exc:
                if "Not found" in str(exc):
                    raise RuntimeError(
                        "Remote GPU service is outdated and does not support patch feature caching; "
                        "update and restart sam_remote_server.py on the GPU host"
                    ) from exc
                raise
        self.image_cache.clear()
        self.current_image_key = None
        self.current_image_shape = None
        self.sam3_state = None
        if self.sam_predictor is not None:
            self.sam_predictor.reset_image()

    def unload_model(self):
        self.image_cache.clear()
        self.current_image_key = None
        if self.remote_url:
            try:
                self._remote_request("/unload", {})
            except Exception:
                pass
            self._model_loaded = False
            return
        if self.sam_predictor is not None:
            del self.sam_predictor
            self.sam_predictor = None
        if self.sam3_processor is not None:
            del self.sam3_processor
            self.sam3_processor = None
        if self.sam3_model is not None:
            del self.sam3_model
            self.sam3_model = None
        self.sam3_state = None
        self.sam3_backend = None
        self.sam3_concept = None
        self.sam3_probabilities = None
        self.sam3_scores = None
        self.last_probability_range = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self._model_loaded = False
