"""SAM/SAM3 inference service intended for an SSH tunnel."""

import argparse
import base64
import json
import subprocess
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from sam_manager import SAMManager


manager = SAMManager(remote_url="")


def encode_prediction(masks, scores, probability_range=None):
    masks = np.asarray(masks, dtype=np.bool_)
    packed = np.packbits(masks.reshape(-1)).tobytes()
    result = {
        "ok": True,
        "shape": list(masks.shape),
        "masks": base64.b64encode(zlib.compress(packed)).decode("ascii"),
        "scores": np.asarray(scores, dtype=np.float32).tolist(),
    }
    if probability_range is not None:
        result["probability_range"] = probability_range
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def _reply(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply(
                {
                    "ok": True,
                    "initialized": manager.is_initialized(),
                    "strategy": manager.strategy,
                    "device": manager.device,
                    "protocolVersion": 2,
                    "patchCache": True,
                }
            )
        elif self.path == "/devices":
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            devices = []
            for line in output.splitlines():
                index, name, total, free, utilization = [part.strip() for part in line.split(",", 4)]
                devices.append(
                    {
                        "index": int(index), "name": name,
                        "totalMemory": int(total), "freeMemory": int(free),
                        "utilization": int(utilization),
                    }
                )
            self._reply({"ok": True, "devices": devices})
        else:
            self._reply({"ok": False, "error": "Not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100 * 1024 * 1024:
                raise ValueError("Request is larger than 100 MB")
            payload = json.loads(self.rfile.read(length) or b"{}")

            if self.path == "/initialize":
                if not manager.initialize(
                    payload.get("strategy", "sam"), device=payload.get("device", "auto")
                ):
                    raise RuntimeError(manager.get_last_error())
                result = {"ok": True, "device": manager.device}
            elif self.path == "/image":
                data = np.frombuffer(base64.b64decode(payload["image"]), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid encoded image")
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if payload.get("image_key"):
                    manager.set_image(rgb, payload["image_key"])
                else:
                    manager.set_image(rgb)
                if manager.get_last_error():
                    raise RuntimeError(manager.get_last_error())
                result = {"ok": True}
            elif self.path == "/predict":
                kind = payload.get("kind")
                if payload.get("image_key"):
                    manager.activate_image(payload["image_key"])
                if kind == "point":
                    masks, scores = manager.predict_mask(payload["points"], payload["labels"])
                elif kind == "box":
                    masks, scores = manager.predict_box(payload["box"])
                elif kind == "boxes":
                    masks, scores = manager.predict_boxes(payload["boxes"])
                elif kind == "concept":
                    masks, scores = manager.predict_concept(
                        payload["concept"],
                        payload.get("probability_min"),
                        payload.get("probability_max"),
                    )
                else:
                    raise ValueError(f"Unsupported prediction kind: {kind}")
                if masks is None:
                    raise RuntimeError(manager.get_last_error())
                result = encode_prediction(masks, scores, manager.last_probability_range if kind == "concept" else None)
            elif self.path == "/unload":
                manager.unload_model()
                result = {"ok": True}
            elif self.path == "/cache/clear":
                manager.clear_image_cache()
                result = {"ok": True}
            else:
                self._reply({"ok": False, "error": "Not found"}, 404)
                return
            self._reply(result)
        except Exception as exc:
            traceback.print_exc()
            self._reply({"ok": False, "error": str(exc)}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--strategy", choices=("sam", "sam3"))
    args = parser.parse_args()

    if args.strategy and not manager.initialize(args.strategy):
        raise SystemExit(manager.get_last_error())
    print(f"SAM inference server listening on {args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
