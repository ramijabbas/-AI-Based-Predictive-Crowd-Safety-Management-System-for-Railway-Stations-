import argparse
import csv
import json
import os
import sys
import time
import threading
from collections import deque, OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
PERSON_ID = CLASSES.index("person")

BOX_COLOR = (0, 255, 0)
TRACK_COLOR = (0, 165, 255)      # orange for tracked
INTRUSION_COLOR = (0, 0, 255)    # red for intrusion
TEXT_COLOR = (255, 255, 255)
HUD_BG = (30, 30, 30)
ZONE_COLOR = (0, 0, 255)
ZONE_FILL = (0, 0, 255)
ALERT_BG = (0, 0, 255)

DEFAULT_ZONE_NORM = [[0.0, 0.60], [1.0, 0.60], [1.0, 1.0], [0.0, 1.0]]  # bottom 40% as track

# ---------------------------------------------------------------------------
# Centroid Tracker  (lightweight SORT-like without Kalman)
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Simple centroid tracker with disappearance handling and greedy matching.

    Tracks bounding boxes across frames and assigns persistent IDs.
    Reference: pyimagesearch centroid tracker.
    """

    def __init__(self, max_disappeared=15, max_distance=60):
        self.next_id = 0
        self.objects = OrderedDict()      # id -> centroid (x,y)
        self.bboxes = {}                  # id -> (x1,y1,x2,y2)
        self.disappeared = OrderedDict()  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """rects: list of (x1,y1,x2,y2)"""
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.bboxes

        # compute centroids for new rects
        input_centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # distance matrix: existing x new
            D = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)

            # greedy matching: smallest distances first
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                oid = object_ids[row]
                self.objects[oid] = input_centroids[col]
                self.bboxes[oid] = rects[col]
                self.disappeared[oid] = 0
                used_rows.add(row)
                used_cols.add(col)

            # handle unmatched existing objects -> disappeared
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            for row in unused_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            # handle new objects
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.objects, self.bboxes


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------
def parse_zone_str(s):
    """Parse 'x1,y1 x2,y2 ...' where x,y are 0-1 floats."""
    pts = []
    for token in s.strip().split():
        if "," not in token:
            continue
        x_s, y_s = token.split(",", 1)
        pts.append([float(x_s), float(y_s)])
    return pts if len(pts) >= 3 else None


def load_zone(zone_file, zone_str, frame_w, frame_h):
    """Return pixel polygon (np.int32) and normalized polygon."""
    norm = None
    if zone_str:
        norm = parse_zone_str(zone_str)
        if norm is None:
            print(f"[WARN] Invalid --zone '{zone_str}', falling back.")
    if norm is None and zone_file and os.path.exists(zone_file):
        try:
            with open(zone_file, "r") as f:
                data = json.load(f)
            # support {"polygon": [[x,y],...]} or [[x,y],...]
            if isinstance(data, dict) and "polygon" in data:
                norm = data["polygon"]
            elif isinstance(data, list):
                norm = data
            print(f"[INFO] Loaded zone from {zone_file}: {norm}")
        except Exception as e:
            print(f"[WARN] Failed to load zone file {zone_file}: {e}")
    if norm is None:
        norm = DEFAULT_ZONE_NORM
        print(f"[INFO] Using default track zone (bottom 40%): {norm}")
        print(f"       Tip: calibrate via --zone-file track_zone.json or interactive 'z' key.")

    # clamp 0-1
    norm = [[max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))] for x, y in norm]
    pts_px = np.array([[int(x * frame_w), int(y * frame_h)] for x, y in norm], dtype=np.int32)
    return pts_px, norm


def save_zone(zone_file, norm_polygon):
    try:
        with open(zone_file, "w") as f:
            json.dump({"polygon": norm_polygon}, f, indent=2)
        print(f"[INFO] Zone saved to {zone_file}")
    except Exception as e:
        print(f"[ERROR] Could not save zone to {zone_file}: {e}")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_people(frame, net, confidence, nms_threshold=0.4):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        if class_id == PERSON_ID and conf > confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confidences.append(float(conf))

    # NMS
    kept = []
    if len(boxes) > 0:
        # NMSBoxes expects [x,y,w,h]
        bboxes_for_nms = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(bboxes_for_nms, confidences, confidence, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
            for i in indices:
                x1, y1, x2, y2 = boxes[i]
                kept.append(((x1, y1, x2, y2), confidences[i]))
    return kept


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------
class AlertManager:
    def __init__(self, alert_dir="alerts", csv_path="alerts/intrusions.csv",
                 webhook_url=None, telegram_token=None, telegram_chat_id=None,
                 cooldown=3.0, no_sound=False):
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(csv_path)
        # ensure csv header
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "event", "num_intruders", "ids", "confidences", "snapshot"])
        self.webhook_url = webhook_url
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.cooldown = cooldown
        self.no_sound = no_sound
        self._last_alert_time = 0
        self._lock = threading.Lock()
        self.total_alerts = 0

    def _play_sound(self):
        if self.no_sound:
            return
        try:
            import winsound
            # need to be non-blocking; winsound.Beep blocks briefly
            winsound.Beep(1200, 400)
            winsound.Beep(800, 400)
        except Exception:
            # fallback: bell
            print("\a", end="", flush=True)

    def _send_webhook(self, payload, snapshot_path):
        if not self.webhook_url:
            return
        def _do():
            try:
                import urllib.request
                import urllib.error
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(self.webhook_url, data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    print(f"[ALERT] Webhook sent, status {resp.status}")
            except Exception as e:
                print(f"[WARN] Webhook failed: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def _send_telegram(self, text, snapshot_path):
        if not (self.telegram_token and self.telegram_chat_id):
            return
        def _do():
            try:
                import urllib.request
                import urllib.parse
                # sendMessage
                msg_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                data = urllib.parse.urlencode({
                    "chat_id": self.telegram_chat_id,
                    "text": text,
                    "parse_mode": "Markdown"
                }).encode()
                req = urllib.request.Request(msg_url, data=data)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    print(f"[ALERT] Telegram message sent {resp.status}")
                # try sendPhoto if snapshot exists
                if snapshot_path and os.path.exists(snapshot_path):
                    import requests  # optional
                    try:
                        with open(snapshot_path, "rb") as f:
                            import urllib.request as ur
                            import http.client
                            # fallback without requests: use multipart manually is complex, try requests if available
                            # if requests missing, skip photo
                            r = requests.post(
                                f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto",
                                data={"chat_id": self.telegram_chat_id, "caption": text},
                                files={"photo": f},
                                timeout=10
                            )
                            print(f"[ALERT] Telegram photo status {r.status_code}")
                    except ImportError:
                        pass
                    except Exception as e:
                        print(f"[WARN] Telegram photo failed: {e}")
            except Exception as e:
                print(f"[WARN] Telegram failed: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def trigger(self, frame, intruding_ids, confidences_map, zone_name="TRACK"):
        now = time.time()
        with self._lock:
            if now - self._last_alert_time < self.cooldown:
                # still log but don't spam snapshot/webhook every frame
                return False, None
            self._last_alert_time = now

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        snapshot_name = f"intrusion_{fname_ts}_{zone_name}.jpg"
        snapshot_path = self.alert_dir / snapshot_name

        # annotate snapshot with red border before save
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, h), INTRUSION_COLOR, 6)
        cv2.putText(annotated, "TRACK INTRUSION ALERT!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, INTRUSION_COLOR, 3)
        cv2.putText(annotated, timestamp, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, INTRUSION_COLOR, 2)
        try:
            cv2.imwrite(str(snapshot_path), annotated)
            print(f"[ALERT] Snapshot saved {snapshot_path}")
        except Exception as e:
            print(f"[WARN] Could not save snapshot: {e}")
            snapshot_path = None

        # CSV log
        try:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                confs = [f"{confidences_map.get(i, 0):.2f}" for i in intruding_ids]
                w.writerow([timestamp, "TRACK_INTRUSION", len(intruding_ids),
                            ";".join(map(str, intruding_ids)), ";".join(confs),
                            str(snapshot_path) if snapshot_path else ""])
        except Exception as e:
            print(f"[WARN] CSV log failed: {e}")

        # console alert
        print(f"\n[ALERT {timestamp}] {len(intruding_ids)} person(s) ENTERED RAILWAY TRACK! IDs={list(intruding_ids)} Snapshot={snapshot_path}\n")

        # sound
        threading.Thread(target=self._play_sound, daemon=True).start()

        # webhook / telegram
        payload = {
            "event": "track_intrusion",
            "timestamp": timestamp,
            "num_intruders": len(intruding_ids),
            "ids": list(intruding_ids),
            "confidences": {str(k): float(v) for k, v in confidences_map.items() if k in intruding_ids},
            "zone": zone_name,
            "snapshot": str(snapshot_path) if snapshot_path else None
        }
        self._send_webhook(payload, str(snapshot_path) if snapshot_path else None)
        text = f"*TRACK INTRUSION ALERT*\nTime: {timestamp}\nIntruders: {len(intruding_ids)} (IDs {list(intruding_ids)})\nZone: {zone_name}"
        self._send_telegram(text, str(snapshot_path) if snapshot_path else None)

        with self._lock:
            self.total_alerts += 1
        return True, str(snapshot_path) if snapshot_path else None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_zone(frame, pts_px, is_alert=False):
    overlay = frame.copy()
    color = INTRUSION_COLOR if is_alert else ZONE_COLOR
    # fill semi-transparent
    if len(pts_px) >= 3:
        cv2.fillPoly(overlay, [pts_px], color)
        alpha = 0.30 if is_alert else 0.18
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        thickness = 3 if is_alert else 2
        cv2.polylines(frame, [pts_px], True, color, thickness)
        # label
        M = cv2.moments(pts_px)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = pts_px[0]
        label = "RAILWAY TRACK - DANGER ZONE" if not is_alert else "!!! DANGER ZONE !!!"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (cx - tw // 2 - 6, cy - th - 10), (cx + tw // 2 + 6, cy + 6), color, -1)
        cv2.putText(frame, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)


def draw_tracks(frame, bboxes_dict, objects_dict, intruding_ids, confidences_map, zone_pts_px):
    """Draw bboxes with ID and intrusion highlight."""
    for oid, bbox in bboxes_dict.items():
        x1, y1, x2, y2 = bbox
        conf = confidences_map.get(oid, 0)
        is_intrusion = oid in intruding_ids
        color = INTRUSION_COLOR if is_intrusion else BOX_COLOR
        thickness = 3 if is_intrusion else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # foot point
        cx = (x1 + x2) // 2
        foot_y = y2
        cv2.circle(frame, (cx, foot_y), 4, color, -1)
        cv2.circle(frame, objects_dict[oid], 4, TRACK_COLOR, -1)
        # trace line from centroid to foot
        cv2.line(frame, objects_dict[oid], (cx, foot_y), color, 1)

        label = f"ID {oid} {conf*100:.0f}%"
        if is_intrusion:
            label += " INTRUSION!"
        ty = y1 - 8 if y1 - 8 > 14 else y1 + 18
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)


def draw_hud(frame, total_count, intruding_count, stable_count, fps, alert_manager, is_alert):
    h, w = frame.shape[:2]
    # top bar
    cv2.rectangle(frame, (0, 0), (w, 48), HUD_BG, -1)
    cv2.putText(frame, f"People: {total_count} (stable {stable_count})", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2)
    col = INTRUSION_COLOR if intruding_count > 0 else TEXT_COLOR
    cv2.putText(frame, f"In TRACK zone: {intruding_count}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # right side FPS + alerts
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, fps_text, (w - tw - 12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)
    alert_text = f"Alerts: {alert_manager.total_alerts}"
    (tw2, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, alert_text, (w - tw2 - 12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, INTRUSION_COLOR if alert_manager.total_alerts else TEXT_COLOR, 2)

    # bottom help
    help_text = "q/Esc:quit  z:calibrate zone  s:save zone  r:reset alerts  h:help"
    cv2.putText(frame, help_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
    if is_alert:
        # flashing banner
        cv2.rectangle(frame, (0, h // 2 - 30), (w, h // 2 + 30), ALERT_BG, -1)
        cv2.putText(frame, "!!! TRACK INTRUSION !!!", (w // 2 - 210, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 3)


# ---------------------------------------------------------------------------
# Args & source
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Railway Track Intrusion Detection - detects people entering railway tracks and sends alerts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python people_counter.py --source 0
  python people_counter.py --source rtsp://user:pass@ip/stream --zone-file track_zone.json
  python people_counter.py --source video.mp4 --zone "0.0,0.65 1.0,0.65 1.0,1.0 0.0,1.0" --alert-dir alerts
  python people_counter.py --source 0 --webhook-url https://example.com/hook --telegram-token 123:ABC --telegram-chat-id -100123

Calibrate zone interactively: press 'z' then click polygon points on video, press ENTER to confirm, 's' to save.
        """
    )
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or RTSP/HTTP URL or video file path")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Minimum detection confidence (default: 0.3)")
    parser.add_argument("--nms-threshold", type=float, default=0.4,
                        help="NMS IoU threshold (default: 0.4)")
    parser.add_argument("--zone", type=str, default=None,
                        help="Danger zone polygon as 'x1,y1 x2,y2 ...' normalized 0-1 (overrides --zone-file)")
    parser.add_argument("--zone-file", type=str, default="track_zone.json",
                        help="JSON file with polygon normalized 0-1, e.g. {\"polygon\": [[0,0.6],...]} (default: track_zone.json)")
    parser.add_argument("--save-zone", type=str, default=None,
                        help="Path to save calibrated zone JSON (default: --zone-file)")
    parser.add_argument("--max-disappeared", type=int, default=15,
                        help="Frames to keep disappeared track before deregister (default: 15)")
    parser.add_argument("--max-distance", type=int, default=70,
                        help="Max centroid distance for tracking association (default: 70)")
    parser.add_argument("--alert-cooldown", type=float, default=3.0,
                        help="Seconds between consecutive alerts (default: 3.0)")
    parser.add_argument("--alert-dir", type=str, default="alerts",
                        help="Directory for alert snapshots (default: alerts)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV log path (default: <alert-dir>/intrusions.csv)")
    parser.add_argument("--webhook-url", type=str, default=None,
                        help="HTTP webhook URL to POST intrusion JSON")
    parser.add_argument("--telegram-token", type=str, default=None,
                        help="Telegram bot token for alerts")
    parser.add_argument("--telegram-chat-id", type=str, default=None,
                        help="Telegram chat ID to send alerts to")
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable audible beep on alert")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output video to file (e.g. output.mp4)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display window (for servers / CCTV)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Start in zone calibration mode")
    parser.add_argument("--prototxt", type=str, default="models/MobileNetSSD_deploy.prototxt",
                        help="Path to Caffe prototxt")
    parser.add_argument("--caffemodel", type=str, default="models/mobilenet.caffemodel",
                        help="Path to Caffe caffemodel (MobileNetSSD)")
    # fallback alternative model path mentioned in folder
    parser.add_argument("--alt-caffemodel", type=str, default="models/MobileNetSSD_deploy.caffemodel",
                        help="Alternative caffemodel path if primary missing")
    return parser.parse_args()


def open_source(source):
    # handle integer string
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        # check if file exists -> open as file
        if os.path.isfile(source):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def resolve_model(prototxt, caffemodel, alt_caffemodel):
    if not os.path.exists(prototxt):
        raise SystemExit(f"Prototxt not found: {prototxt}")
    # try primary, then alt, validating that OpenCV can actually load it
    for candidate in [caffemodel, alt_caffemodel]:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            # quick validation — try loading
            net = cv2.dnn.readNetFromCaffe(prototxt, candidate)
            del net
            if candidate != caffemodel:
                print(f"[INFO] Using alternative model {candidate} (primary {caffemodel} failed or not usable)")
            return prototxt, candidate
        except Exception as e:
            print(f"[WARN] Model {candidate} failed to load: {e}")
            continue
    raise SystemExit(f"No usable caffemodel found. Tried: {caffemodel}, {alt_caffemodel}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    prototxt, caffemodel = resolve_model(args.prototxt, args.caffemodel, args.alt_caffemodel)

    # resolve csv
    csv_path = args.csv if args.csv else os.path.join(args.alert_dir, "intrusions.csv")
    save_zone_path = args.save_zone if args.save_zone else args.zone_file

    print(f"[INFO] Loading model {caffemodel} ...")
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    cap = open_source(args.source)
    is_stream = not (os.path.isfile(args.source) or args.source.isdigit())
    # try get first frame for zone setup
    ok, first_frame = cap.read()
    if not ok:
        # reconnect logic for stream will handle; for file we error
        print("[WARN] Could not read first frame, trying again...")
        time.sleep(0.5)
        ok, first_frame = cap.read()
        if not ok:
            raise SystemExit("Could not read frame from source.")
    h0, w0 = first_frame.shape[:2]
    # keep first frame as queued for all sources to avoid dropping it;
    # this also handles single-image VideoCapture where rewind does not work.
    queued_frame = first_frame
    first_frame = None
    # init zone
    zone_pts_px, zone_norm = load_zone(args.zone_file, args.zone, w0, h0)

    tracker = CentroidTracker(max_disappeared=args.max_disappeared,
                              max_distance=args.max_distance)

    alert_mgr = AlertManager(
        alert_dir=args.alert_dir,
        csv_path=csv_path,
        webhook_url=args.webhook_url,
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat_id,
        cooldown=args.alert_cooldown,
        no_sound=args.no_sound
    )

    failed_reads = 0
    prev_time = time.time()
    fps = 0.0
    count_history = deque(maxlen=15)
    intruding_history = deque(maxlen=30)

    # calibration state
    calibrate_mode = args.calibrate
    calib_points_norm = []  # normalized
    calib_points_px = []
    # for mouse
    calib_frame_w = w0
    calib_frame_h = h0

    video_writer = None
    # queued frame handling
    has_queued = 'queued_frame' in locals()

    # Intrusion state per ID (to detect entry event vs continuous)
    prev_intruding_ids = set()
    total_intrusion_events = 0  # counts entry events

    def mouse_cb(event, x, y, flags, param):
        nonlocal calib_points_norm, calib_points_px
        if not calibrate_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # add point normalized
            calib_points_norm.append([x / calib_frame_w, y / calib_frame_h])
            calib_points_px.append([x, y])
            print(f"[CALIB] Point {len(calib_points_px)}: ({x},{y}) -> norm {[x/calib_frame_w, y/calib_frame_h]}")

    if not args.headless:
        cv2.namedWindow("People Detection - Railway Track", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("People Detection - Railway Track", mouse_cb)
        print("""\n[CONTROLS]
  q / Esc  : quit
  z        : toggle zone calibration mode (click to add points)
  Enter    : confirm calibration (needs >=3 points)
  c        : clear calibration points
  s        : save current zone to file
  r        : reset alert counter
  h        : print help
        """)

    frame_idx = 0
    consecutive_intrusion_frames = 0

    while True:
        if has_queued:
            frame = queued_frame
            ok = True
            has_queued = False
        else:
            ok, frame = cap.read()
        if not ok:
            if os.path.isfile(args.source):
                print("Video file ended.")
                break
            if not is_stream or failed_reads >= 5:
                print("Stream ended or connection lost.")
                break
            failed_reads += 1
            print(f"Frame read failed ({failed_reads}/5), reconnecting...")
            cap.release()
            time.sleep(1)
            try:
                cap = open_source(args.source)
            except SystemExit:
                pass
            continue
        failed_reads = 0
        frame_idx += 1
        h, w = frame.shape[:2]
        calib_frame_w, calib_frame_h = w, h
        # if frame size changed, recompute zone_px
        if w != w0 or h != h0:
            # update zone pts to new size using norm
            zone_pts_px = np.array([[int(x * w), int(y * h)] for x, y in zone_norm], dtype=np.int32)
            w0, h0 = w, h

        # video writer init lazy
        if args.output and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_guess = cap.get(cv2.CAP_PROP_FPS)
            if fps_guess <= 1 or np.isnan(fps_guess):
                fps_guess = 20.0
            video_writer = cv2.VideoWriter(args.output, fourcc, fps_guess, (w, h))
            print(f"[INFO] Recording output to {args.output} @ {fps_guess:.1f} FPS")

        # detection
        detections = detect_people(frame, net, args.confidence, args.nms_threshold)
        rects = [b for b, _ in detections]
        confs_list = [c for _, c in detections]

        # tracking
        objects, bboxes = tracker.update(rects)

        # Need to associate confidences to IDs: for each ID, find closest detection bbox
        # simple: for each tracked ID, find min distance between its bbox center and detection centers
        confidences_map = {}
        if len(detections) > 0:
            det_centers = [(((b[0]+b[2])//2), ((b[1]+b[3])//2)) for b, _ in detections]
            for oid, tbbox in bboxes.items():
                tcx = (tbbox[0] + tbbox[2]) // 2
                tcy = (tbbox[1] + tbbox[3]) // 2
                best_i = -1
                best_dist = 1e9
                for i, (dcx, dcy) in enumerate(det_centers):
                    d = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_i = i
                if best_i >= 0 and best_dist < (args.max_distance * 2) ** 2:
                    confidences_map[oid] = confs_list[best_i]
                else:
                    confidences_map[oid] = 0.5  # fallback

        # intrusion check: foot point inside polygon
        intruding_ids = set()
        if len(zone_pts_px) >= 3:
            for oid, bbox in bboxes.items():
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                foot = (cx, y2)  # bottom center
                # also try centroid if foot fails? we use foot which is more accurate for ground contact
                inside = cv2.pointPolygonTest(zone_pts_px, foot, False) >= 0
                # fallback: if bbox heavily overlaps zone (>30% area inside), also consider intrusion
                if not inside:
                    # quick overlap check: sample 9 points of bbox
                    # but keep simple: check centroid inside
                    centroid = objects[oid]
                    if cv2.pointPolygonTest(zone_pts_px, tuple(map(int, centroid)), False) >= 0:
                        inside = True
                if inside:
                    intruding_ids.add(oid)

        # detect *new* entries (was not intruding before, now is)
        new_entries = intruding_ids - prev_intruding_ids
        if len(new_entries) > 0:
            total_intrusion_events += len(new_entries)
            print(f"[EVENT] New track entry: IDs {new_entries} (total events {total_intrusion_events})")

        # also consider continuous intrusion as alert-worthy but with cooldown via AlertManager
        is_intrusion_active = len(intruding_ids) > 0
        if is_intrusion_active:
            consecutive_intrusion_frames += 1
        else:
            consecutive_intrusion_frames = 0

        # stabilize counts
        count_history.append(len(objects))
        stable_count = max(count_history) if len(count_history) else 0
        intruding_history.append(len(intruding_ids))

        # Alert trigger: on new entry OR if intrusion persists for >5 frames and cooldown allows
        triggered = False
        if is_intrusion_active:
            # trigger on new entry immediately
            should_trigger = len(new_entries) > 0 or consecutive_intrusion_frames == 5 or consecutive_intrusion_frames % 30 == 0
            if should_trigger:
                # pass confidences_map filtered
                ok_trig, _ = alert_mgr.trigger(frame, intruding_ids, confidences_map)
                triggered = ok_trig

        prev_intruding_ids = set(intruding_ids)

        # FPS
        now_time = time.time()
        instant_fps = 1.0 / max(now_time - prev_time, 1e-6)
        fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps
        prev_time = now_time

        # Drawing
        # zone
        draw_zone(frame, zone_pts_px, is_alert=is_intrusion_active)
        # tracks
        draw_tracks(frame, bboxes, objects, intruding_ids, confidences_map, zone_pts_px)
        # HUD including alert banner
        draw_hud(frame, len(objects), len(intruding_ids), stable_count, fps, alert_mgr, is_alert=is_intrusion_active and (consecutive_intrusion_frames % 10 < 5))

        # calibration overlay
        if calibrate_mode:
            # dim
            cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)
            cv2.putText(frame, "CALIBRATION MODE: Click to add points, ENTER=confirm, c=clear, z=exit", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Points: {len(calib_points_px)} (need >=3)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            if calib_points_px:
                pts = np.array(calib_points_px, dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
                for i, (px, py) in enumerate(calib_points_px):
                    cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
                    cv2.putText(frame, str(i+1), (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, "Press 's' to save zone after confirming", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
            cv2.putText(frame, f"Current zone: {zone_norm}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

        # extra status if triggered flash
        if triggered:
            # add timestamp on frame already handled by alert manager snapshot
            pass

        # Show
        if not args.headless:
            cv2.imshow("People Detection - Railway Track", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                print("[INFO] Quit requested.")
                break
            elif key == ord("z"):
                calibrate_mode = not calibrate_mode
                if calibrate_mode:
                    calib_points_norm = []
                    calib_points_px = []
                    print("[CALIB] Entered calibration mode. Click points on image.")
                else:
                    print("[CALIB] Exited calibration mode.")
            elif key == ord("c") and calibrate_mode:
                calib_points_norm = []
                calib_points_px = []
                print("[CALIB] Cleared points.")
            elif key == 13 and calibrate_mode:  # Enter
                if len(calib_points_norm) >= 3:
                    zone_norm = [list(map(float, p)) for p in calib_points_norm]
                    zone_pts_px = np.array([[int(x * w), int(y * h)] for x, y in zone_norm], dtype=np.int32)
                    print(f"[CALIB] Zone updated: {zone_norm}")
                    calibrate_mode = False
                    calib_points_norm = []
                    calib_points_px = []
                else:
                    print("[CALIB] Need at least 3 points.")
            elif key == ord("s"):
                # save current zone
                if calibrate_mode and len(calib_points_norm) >= 3:
                    zone_norm = [list(map(float, p)) for p in calib_points_norm]
                    zone_pts_px = np.array([[int(x * w), int(y * h)] for x, y in zone_norm], dtype=np.int32)
                save_zone(save_zone_path, zone_norm)
            elif key == ord("r"):
                alert_mgr.total_alerts = 0
                total_intrusion_events = 0
                print("[INFO] Alert counters reset.")
            elif key == ord("h"):
                print("""
[HELP]
  q/Esc : quit
  z     : toggle calibration (click polygon points)
  Enter : confirm polygon (>=3 points)
  c     : clear points (in calibration)
  s     : save zone to file
  r     : reset alert counts
  h     : show this help
                """)
            elif key == ord("p"):
                # pause
                print("[INFO] Paused - press any key to continue")
                cv2.waitKey(0)
        else:
            # headless: small sleep to not spin too fast if no display wait
            pass

        # save video frame
        if video_writer is not None:
            video_writer.write(frame)

        # handle case where source is image? cap.read will fail next loop and we exit
        # for headless, allow Ctrl+C
        if frame_idx % 200 == 0:
            print(f"[STATS] Frame {frame_idx} | People {len(objects)} | InTrack {len(intruding_ids)} | FPS {fps:.1f} | Alerts {alert_mgr.total_alerts} | Events {total_intrusion_events}")

    # cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] Output video saved to {args.output}")
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"\n[SUMMARY] Total frames {frame_idx} | Total alerts {alert_mgr.total_alerts} | Total intrusion events {total_intrusion_events}")
    print(f"[SUMMARY] Snapshots in {alert_mgr.alert_dir} | Log {csv_path}")
    if zone_norm:
        print(f"[SUMMARY] Final zone (norm): {zone_norm} -> file {save_zone_path}")


if __name__ == "__main__":
    main()
