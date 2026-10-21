#!/usr/bin/env python3
"""
OAK4 vision service — DepthAI v3 (Peripheral mode over Ethernet/PoE).

OAK4 is not a USB webcam. Connect it on the network (oakctl connect / PoE+),
then run this script on the Jetson/middleware host:

    pip install -r requirements.txt
    python main.py --list-devices
    python main.py --mode preview --show
    python main.py --mode detect --middleware-port 9091

oakctl connect crashes in some terminals (indicatif "motion of zero" panic).
Fix: export COLUMNS=120 LINES=40  (or use macOS Terminal / iTerm).
Peripheral Python apps do not require oakctl connect — use main.py --list-devices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from typing import Any, Optional

import cv2
import depthai as dai

STEREO_FPS = 20
STREAM_SIZE = (640, 400)
MODEL = dai.NNModelDescription("yolov6-nano")


def list_devices() -> list[dai.DeviceInfo]:
    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        print("No DepthAI devices found. Check Ethernet/PoE and run: oakctl connect")
        return []
    for info in infos:
        state = str(info.state).split("X_LINK_")[-1]
        print(f"  {info.name}  id={info.deviceId}  state={state}")
    return infos


def pick_device(device_id: Optional[str]) -> dai.DeviceInfo:
    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        raise RuntimeError(
            "No DepthAI device on the network. Use --list-devices or: oakctl connect"
        )
    if device_id is None:
        return infos[0]
    for info in infos:
        if info.deviceId == device_id or info.name == device_id:
            return info
    raise RuntimeError(f"Device not found: {device_id}")


def mm_to_m(value: float) -> float:
    return round(value / 1000.0, 4)


def detections_to_payload(
    detections: dai.SpatialImgDetections,
    camera_id: str,
    min_confidence: float,
) -> Optional[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for det in detections.detections:
        if det.confidence < min_confidence:
            continue
        coords = det.spatialCoordinates
        objects.append(
            {
                "label": det.labelName,
                "confidence": round(det.confidence, 4),
                # 3D position in camera frame (metres): x=right, y=down, z=forward
                "position": {
                    "x": mm_to_m(coords.x),
                    "y": mm_to_m(coords.y),
                    "z": mm_to_m(coords.z),
                },
                # 2D bounding box in image (normalised 0..1)
                "bbox": {
                    "xmin": round(det.xmin, 4),
                    "ymin": round(det.ymin, 4),
                    "xmax": round(det.xmax, 4),
                    "ymax": round(det.ymax, 4),
                },
            }
        )
    if not objects:
        return None
    return {
        "type": "vision.detection",
        "camera": camera_id,
        "ts": int(time.time() * 1000),
        "objects": objects,
    }


def format_detection_line(obj: dict[str, Any]) -> str:
    pos = obj["position"]
    bb = obj["bbox"]
    return (
        f"{obj['label']} conf={obj['confidence']:.2f} "
        f"xyz=({pos['x']:.2f},{pos['y']:.2f},{pos['z']:.2f})m "
        f"bbox=({bb['xmin']:.2f},{bb['ymin']:.2f})-({bb['xmax']:.2f},{bb['ymax']:.2f})"
    )


def draw_detections(
    frame,
    detections: dai.SpatialImgDetections,
    min_confidence: float,
) -> None:
    h, w = frame.shape[:2]
    for det in detections.detections:
        if det.confidence < min_confidence:
            continue
        x1, y1 = int(det.xmin * w), int(det.ymin * h)
        x2, y2 = int(det.xmax * w), int(det.ymax * h)
        z_m = mm_to_m(det.spatialCoordinates.z)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{det.labelName} {z_m:.2f}m",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


class MiddlewarePublisher:
    """Thread-safe JSON line broadcaster for connected middleware clients."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._clients: list[asyncio.StreamWriter] = []
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        del reader
        addr = writer.get_extra_info("peername")
        async with self._lock:
            self._clients.append(writer)
        print(f">> middleware connected {addr}")
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
        finally:
            async with self._lock:
                if writer in self._clients:
                    self._clients.remove(writer)
            writer.close()
            print(f">> middleware disconnected {addr}")

    async def _run_server(self) -> None:
        server = await asyncio.start_server(
            self._handle_client, self._host, self._port
        )
        print(f"Middleware publisher listening on {self._host}:{self._port}")
        async with server:
            await server.serve_forever()

    def start(self) -> None:
        def runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_server())

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        while self._loop is None:
            time.sleep(0.01)

    def publish(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return
        line = (json.dumps(payload) + "\n").encode()

        async def _broadcast() -> None:
            async with self._lock:
                dead: list[asyncio.StreamWriter] = []
                for writer in self._clients:
                    try:
                        writer.write(line)
                        await writer.drain()
                    except (ConnectionError, OSError):
                        dead.append(writer)
                for writer in dead:
                    self._clients.remove(writer)

        asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)


def run_preview(device_info: dai.DeviceInfo, show: bool) -> None:
    with dai.Device(device_info) as device:
        print(f"Connected: {device.getDeviceName()} ({device.getPlatformAsString()})")
        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            queue = cam.requestOutput(STREAM_SIZE).createOutputQueue()
            pipeline.start()
            print("RGB preview running (q to quit)")
            while pipeline.isRunning():
                frame = queue.get()
                if not isinstance(frame, dai.ImgFrame):
                    continue
                if show:
                    cv2.imshow("oak_rgb", frame.getCvFrame())
                    if cv2.waitKey(1) == ord("q"):
                        break
                else:
                    print(f"frame seq={frame.getSequenceNum()}")


def run_detect(
    device_info: dai.DeviceInfo,
    camera_id: str,
    publisher: MiddlewarePublisher,
    show: bool,
    min_confidence: float,
    verbose: bool,
) -> None:
    with dai.Device(device_info) as device:
        print(f"Connected: {device.getDeviceName()} ({device.getPlatformAsString()})")
        with dai.Pipeline(device) as pipeline:
            cam_rgb = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A, sensorFps=STEREO_FPS
            )
            cam_left = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B, sensorFps=STEREO_FPS
            )
            cam_right = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C, sensorFps=STEREO_FPS
            )

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setExtendedDisparity(True)
            cam_left.requestOutput(STREAM_SIZE).link(stereo.left)
            cam_right.requestOutput(STREAM_SIZE).link(stereo.right)

            spatial_nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
                cam_rgb, stereo, MODEL
            )
            spatial_nn.setDepthLowerThreshold(100)
            spatial_nn.setDepthUpperThreshold(5000)

            det_queue = spatial_nn.out.createOutputQueue(maxSize=4, blocking=False)
            preview_q = (
                spatial_nn.passthrough.createOutputQueue(maxSize=4, blocking=False)
                if show
                else None
            )

            print("Spatial detection running (q to quit preview window)")
            pipeline.start()
            last_detections: Optional[dai.SpatialImgDetections] = None
            while pipeline.isRunning():
                detections = det_queue.tryGet()
                if isinstance(detections, dai.SpatialImgDetections):
                    last_detections = detections
                    payload = detections_to_payload(
                        detections, camera_id, min_confidence
                    )
                    if payload is not None:
                        publisher.publish(payload)
                        if verbose:
                            print(json.dumps(payload, ensure_ascii=False))
                        else:
                            for obj in payload["objects"]:
                                print(f"  {format_detection_line(obj)}")

                if preview_q is not None:
                    frame = preview_q.tryGet()
                    if isinstance(frame, dai.ImgFrame):
                        img = frame.getCvFrame()
                        if last_detections is not None:
                            draw_detections(img, last_detections, min_confidence)
                        cv2.imshow("oak_rgb", img)
                        if cv2.waitKey(1) == ord("q"):
                            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OAK4 vision service (DepthAI v3, Ethernet/PoE)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print network devices and exit",
    )
    parser.add_argument(
        "--device-id",
        help="Device name or ID (default: first available)",
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "detect"),
        default="preview",
        help="preview=RGB only, detect=spatial YOLO + middleware events",
    )
    parser.add_argument(
        "--camera-id",
        default="oak4_front",
        help="Camera name in vision.* JSON messages",
    )
    parser.add_argument(
        "--middleware-host",
        default="0.0.0.0",
        help="TCP bind host for JSON line publisher",
    )
    parser.add_argument(
        "--middleware-port",
        type=int,
        default=9091,
        help="TCP port for middleware clients",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open local OpenCV preview window",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON payload per detection frame",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0

    device_info = pick_device(args.device_id)
    publisher: Optional[MiddlewarePublisher] = None

    if args.mode == "detect":
        publisher = MiddlewarePublisher(args.middleware_host, args.middleware_port)
        publisher.start()
        run_detect(
            device_info,
            args.camera_id,
            publisher,
            args.show,
            args.min_confidence,
            args.verbose,
        )
    else:
        if args.mode != "preview":
            print(f"Unknown mode: {args.mode}", file=sys.stderr)
            return 1
        run_preview(device_info, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
