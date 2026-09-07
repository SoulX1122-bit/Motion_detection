"""Webcam pose tracking with movement and observability states by body region.

Press Q or Esc to close the webcam window.
"""

from collections import deque
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time

import cv2
import mediapipe as mp


# Calibration settings. Values are normalized image-coordinate distances.
# Recommended calibration protocol (press R to record the full sequence):
# 1. Sit/stand still for 10 seconds.
# 2. Move right arm for 10 seconds.
# 3. Stay still for 10 seconds.
# 4. Move left arm for 10 seconds.
# 5. Stay still for 10 seconds.
# 6. Move head for 10 seconds.
# 7. Stay still for 10 seconds.
# 8. Move right leg for 10 seconds.
# 9. Stay still for 10 seconds.
# 10. Move left leg for 10 seconds.
# 11. Stay still for 10 seconds.
# 12. Move whole body for 10 seconds.

CAMERA_INDEX = 0
CALIBRATION_MODE = True
SMOOTHING_WINDOW = 8

MOVING_THRESHOLD = 0.012
STATIONARY_THRESHOLD = 0.006
MIN_MOVING_FRAMES = 3
MIN_STATIONARY_FRAMES = 8
MIN_VISIBILITY = 0.50


# MediaPipe Pose Landmarker indices.
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

REGION_LANDMARKS = {
    "HEAD": [
        NOSE,
        LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER, LEFT_EAR,
        RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER, RIGHT_EAR,
    ],
    "LEFT ARM": [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST],
    "RIGHT ARM": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
    "TORSO": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
    "LEFT LEG": [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE],
    "RIGHT LEG": [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE],
}

REGION_CONNECTIONS = {
    "HEAD": [
        (NOSE, LEFT_EYE_INNER),
        (NOSE, RIGHT_EYE_INNER),
        (LEFT_EYE_INNER, LEFT_EYE),
        (LEFT_EYE, LEFT_EYE_OUTER),
        (LEFT_EYE_OUTER, LEFT_EAR),
        (RIGHT_EYE_INNER, RIGHT_EYE),
        (RIGHT_EYE, RIGHT_EYE_OUTER),
        (RIGHT_EYE_OUTER, RIGHT_EAR),
    ],
    "LEFT ARM": [
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
    ],
    "RIGHT ARM": [
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
    ],
    "TORSO": [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
    ],
    "LEFT LEG": [
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
    ],
    "RIGHT LEG": [
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ],
}

# OpenCV uses BGR color order.
MOVEMENT_COLORS = {
    "MOVING": (0, 0, 255),       # Red
    "STATIONARY": (0, 255, 0),   # Green
    "UNCERTAIN": (0, 165, 255),  # Orange
}

PARTIAL_COLOR = (255, 0, 255)     # Purple


# Trial labels are calibration metadata only. They never affect tracking.
TRIAL_LABELS = {
    0: "NO TRIAL",
    1: "STILL",
    2: "RIGHT_ARM",
    3: "LEFT_ARM",
    4: "HEAD",
    5: "RIGHT_LEG",
    6: "LEFT_LEG",
    7: "WHOLE_BODY",
    8: "SITTING_STILL",
    9: "LYING_STILL",
    10: "LYING_SIDEWAYS_STILL",
}

TRIAL_KEY_BINDINGS = {
    ord("0"): 0,
    ord("1"): 1,
    ord("2"): 2,
    ord("3"): 3,
    ord("4"): 4,
    ord("5"): 5,
    ord("6"): 6,
    ord("7"): 7,
    ord("8"): 8,
    ord("9"): 9,
    ord("s"): 10,
    ord("S"): 10,
}


class MovementCsvLogger:
    """Writes one calibration row per processed frame while recording is on."""

    def __init__(self, project_directory: Path) -> None:
        data_directory = project_directory / "data"
        data_directory.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = data_directory / f"movement_data_{timestamp}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self._fieldnames())
        self.writer.writeheader()
        self.file.flush()
        self.recording = False

    @staticmethod
    def _fieldnames() -> list[str]:
        fieldnames = ["timestamp_ms", "frame_number", "trial"]

        for region_name in REGION_LANDMARKS:
            csv_name = region_name.replace(" ", "_")
            fieldnames.append(f"{csv_name}_motion")

        for region_name in REGION_LANDMARKS:
            csv_name = region_name.replace(" ", "_")
            fieldnames.append(f"{csv_name}_observability")

        for region_name in REGION_LANDMARKS:
            csv_name = region_name.replace(" ", "_")
            fieldnames.append(f"{csv_name}_state")

        fieldnames.append("OVERALL_state")
        return fieldnames

    def write_frame(
        self,
        timestamp_ms: int,
        frame_number: int,
        trial: str,
        region_trackers: dict[str, "RegionTracker"],
        overall_state: str,
    ) -> None:
        """Write smoothed motion; use a blank field for unobservable regions."""
        row = {
            "timestamp_ms": timestamp_ms,
            "frame_number": frame_number,
            "trial": trial,
            "OVERALL_state": overall_state,
        }

        for region_name, tracker in region_trackers.items():
            csv_name = region_name.replace(" ", "_")
            row[f"{csv_name}_motion"] = (
                f"{tracker.smoothed_motion:.8f}"
                if tracker.is_observable
                else ""
            )
            row[f"{csv_name}_observability"] = tracker.observability
            row[f"{csv_name}_state"] = tracker.state

        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        """Close the CSV file before the application exits."""
        self.file.close()


def is_landmark_reliable(landmark) -> bool:
    """Check visibility and reject landmarks projected outside the image."""
    return (
        landmark.visibility >= MIN_VISIBILITY
        and 0.0 <= landmark.x <= 1.0
        and 0.0 <= landmark.y <= 1.0
    )


def get_observability(region_name: str, visible_indices: set[int]) -> str:
    """Classify whether a region has enough meaningful geometry for movement.

    RELIABLE means every region point is available. PARTIAL means at least one
    actual skeleton connection has both endpoints visible. UNRELIABLE means no
    connected segment can be measured, so movement must not be calculated.
    """
    all_indices = set(REGION_LANDMARKS[region_name])

    if visible_indices == all_indices:
        return "RELIABLE"

    for first_index, second_index in REGION_CONNECTIONS[region_name]:
        if first_index in visible_indices and second_index in visible_indices:
            return "PARTIAL"

    return "UNRELIABLE"


@dataclass
class RegionTracker:
    """Stores movement history, observability, and stable state for one region."""

    name: str
    previous_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    motion_history: deque = field(
        default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW)
    )
    state: str = "UNCERTAIN"
    observability: str = "UNRELIABLE"
    is_observable: bool = False
    moving_frames: int = 0
    stationary_frames: int = 0
    smoothed_motion: float = 0.0

    def mark_unreliable(self) -> None:
        """Clear history when movement can no longer be safely measured."""
        self.previous_positions.clear()
        self.motion_history.clear()
        self.smoothed_motion = 0.0
        self.moving_frames = 0
        self.stationary_frames = 0
        self.state = "UNCERTAIN"
        self.observability = "UNRELIABLE"
        self.is_observable = False

    def update(self, landmarks, landmark_indices: list[int]) -> str:
        """Update one region using only currently reliable landmark positions."""
        current_positions = {}

        for index in landmark_indices:
            landmark = landmarks[index]
            if is_landmark_reliable(landmark):
                current_positions[index] = (landmark.x, landmark.y)

        visible_indices = set(current_positions)
        self.observability = get_observability(self.name, visible_indices)

        if self.observability == "UNRELIABLE":
            self.mark_unreliable()
            return self.state

        # The first observable frame establishes a fresh baseline. This avoids
        # a false large displacement when a hidden body part reappears.
        if not self.is_observable:
            self.previous_positions = current_positions
            self.is_observable = True
            self.state = "STATIONARY"
            return self.state

        displacements = []

        for index, position in current_positions.items():
            if index not in self.previous_positions:
                continue

            old_x, old_y = self.previous_positions[index]
            new_x, new_y = position
            displacement = ((new_x - old_x) ** 2 + (new_y - old_y) ** 2) ** 0.5
            displacements.append(displacement)

        # A PARTIAL region uses only its visible, connected landmarks. New
        # points have no old position and are deliberately ignored this frame.
        if displacements:
            average_displacement = sum(displacements) / len(displacements)
            self.motion_history.append(average_displacement)
            self.smoothed_motion = sum(self.motion_history) / len(self.motion_history)

        self.previous_positions = current_positions
        self._update_state()
        return self.state

    def _update_state(self) -> None:
        """Apply the existing hysteresis and persistence rules."""
        if self.state == "STATIONARY":
            if self.smoothed_motion >= MOVING_THRESHOLD:
                self.moving_frames += 1
            else:
                self.moving_frames = 0

            if self.moving_frames >= MIN_MOVING_FRAMES:
                self.state = "MOVING"
                self.moving_frames = 0
                self.stationary_frames = 0

        elif self.state == "MOVING":
            if self.smoothed_motion <= STATIONARY_THRESHOLD:
                self.stationary_frames += 1
            else:
                self.stationary_frames = 0

            if self.stationary_frames >= MIN_STATIONARY_FRAMES:
                self.state = "STATIONARY"
                self.stationary_frames = 0
                self.moving_frames = 0


def get_region_color(tracker: RegionTracker) -> tuple[int, int, int]:
    """Give PARTIAL stationary regions a visibly different skeleton color."""
    if tracker.observability == "PARTIAL" and tracker.state == "STATIONARY":
        return PARTIAL_COLOR

    return MOVEMENT_COLORS[tracker.state]


def landmark_display_state(index: int, region_trackers: dict[str, RegionTracker]) -> str:
    """Choose one dot state when a landmark belongs to multiple regions."""
    related_trackers = [
        tracker
        for name, tracker in region_trackers.items()
        if index in REGION_LANDMARKS[name]
    ]

    if any(tracker.state == "MOVING" for tracker in related_trackers):
        return "MOVING"
    if any(tracker.state == "STATIONARY" for tracker in related_trackers):
        return "STATIONARY"
    return "UNCERTAIN"


def draw_skeleton(frame, landmarks, region_trackers: dict[str, RegionTracker]) -> None:
    """Draw observable region connections and landmark dots."""
    height, width = frame.shape[:2]

    for region_name, connections in REGION_CONNECTIONS.items():
        tracker = region_trackers[region_name]

        # Unreliable regions are intentionally omitted from the skeleton.
        if tracker.observability == "UNRELIABLE":
            continue

        color = get_region_color(tracker)
        thickness = 2 if tracker.observability == "PARTIAL" else 3

        for first_index, second_index in connections:
            first = landmarks[first_index]
            second = landmarks[second_index]

            if not (
                is_landmark_reliable(first)
                and is_landmark_reliable(second)
            ):
                continue

            first_point = (int(first.x * width), int(first.y * height))
            second_point = (int(second.x * width), int(second.y * height))

            cv2.line(
                frame,
                first_point,
                second_point,
                color,
                thickness,
                cv2.LINE_AA,
            )

    all_indices = set()
    for indices in REGION_LANDMARKS.values():
        all_indices.update(indices)

    for index in all_indices:
        landmark = landmarks[index]
        if not is_landmark_reliable(landmark):
            continue

        state = landmark_display_state(index, region_trackers)
        color = MOVEMENT_COLORS[state]
        radius = 6 if state == "MOVING" else 4
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, radius, color, -1)


def get_overall_state(region_trackers: dict[str, RegionTracker]) -> str:
    """Use only RELIABLE/PARTIAL regions that support movement analysis."""
    observable_trackers = [
        tracker
        for tracker in region_trackers.values()
        if tracker.is_observable
    ]

    if not observable_trackers:
        return "UNCERTAIN"

    if any(tracker.state == "MOVING" for tracker in observable_trackers):
        return "MOVING"

    return "STATIONARY"


def draw_status_panel(
    frame,
    region_trackers: dict[str, RegionTracker],
    csv_logger: MovementCsvLogger | None,
    current_trial: str,
) -> None:
    """Draw observability, motion, and state for every body region."""
    panel_width = 590 if CALIBRATION_MODE else 390
    panel_height = 465 if CALIBRATION_MODE else 305

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        "BODY MOVEMENT",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y = 68
    if CALIBRATION_MODE:
        threshold_text = (
            f"Thresholds: moving >= {MOVING_THRESHOLD:.4f} | "
            f"stationary <= {STATIONARY_THRESHOLD:.4f}"
        )
        cv2.putText(frame, threshold_text, (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 0), 1, cv2.LINE_AA)
        y += 27

        settings_text = (
            f"Smoothing: {SMOOTHING_WINDOW} frames | "
            f"Confirm: {MIN_MOVING_FRAMES} moving, {MIN_STATIONARY_FRAMES} still"
        )
        cv2.putText(frame, settings_text, (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (180, 180, 180), 1, cv2.LINE_AA)
        y += 29

    for name, tracker in region_trackers.items():
        if tracker.is_observable:
            motion_text = f"{tracker.smoothed_motion:.5f}"
        else:
            motion_text = "---"

        if tracker.observability == "PARTIAL" and tracker.state == "STATIONARY":
            color = PARTIAL_COLOR
        else:
            color = MOVEMENT_COLORS[tracker.state]

        if CALIBRATION_MODE:
            text = (
                f"{name:<10} Obs: {tracker.observability:<10} "
                f"Motion: {motion_text:<7} State: {tracker.state}"
            )
        else:
            text = f"{name}: {tracker.state} ({tracker.observability})"

        cv2.putText(frame, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48 if CALIBRATION_MODE else 0.50, color,
                    1 if CALIBRATION_MODE else 2, cv2.LINE_AA)
        y += 31 if CALIBRATION_MODE else 28

    overall_state = get_overall_state(region_trackers)
    cv2.putText(frame, f"OVERALL: {overall_state}", (25, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, MOVEMENT_COLORS[overall_state],
                2, cv2.LINE_AA)

    # Eye detection is intentionally deferred; Pose Landmarker lacks eyelids.
    cv2.putText(frame, "LEFT EYE: UNCERTAIN", (25, y + 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, MOVEMENT_COLORS["UNCERTAIN"],
                1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT EYE: UNCERTAIN", (25, y + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, MOVEMENT_COLORS["UNCERTAIN"],
                1, cv2.LINE_AA)

    if csv_logger is not None:
        recording_text = "RECORDING: ON" if csv_logger.recording else "RECORDING: OFF"
        recording_color = (0, 0, 255) if csv_logger.recording else (180, 180, 180)
        cv2.putText(frame, recording_text, (25, y + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, recording_color,
                    2, cv2.LINE_AA)
        cv2.putText(frame, f"CSV: {csv_logger.path.name}", (25, y + 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                    1, cv2.LINE_AA)

        cv2.putText(frame, f"TRIAL: {current_trial}", (25, y + 146),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0),
                    2, cv2.LINE_AA)


def main() -> None:
    project_directory = Path(__file__).resolve().parent
    model_path = project_directory / "models" / "pose_landmarker_full.task"
    if not model_path.exists():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    region_trackers = {name: RegionTracker(name) for name in REGION_LANDMARKS}
    csv_logger = MovementCsvLogger(project_directory) if CALIBRATION_MODE else None

    if csv_logger is not None:
        print(f"Calibration CSV created: {csv_logger.path}")
        print("Controls: R = start/stop recording")
        print("Trials: 0=NO TRIAL, 1=STILL, 2=RIGHT_ARM, 3=LEFT_ARM, 4=HEAD")
        print("        5=RIGHT_LEG, 6=LEFT_LEG, 7=WHOLE_BODY, 8=SITTING_STILL")
        print("        9=LYING_STILL, S=LYING_SIDEWAYS_STILL")

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}.")

    print("Camera started. Press Q or Esc to close the window.")
    start_time = time.perf_counter()
    frame_number = 0
    current_trial_number = 0
    current_trial = TRIAL_LABELS[current_trial_number]

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame_number += 1

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    for name, indices in REGION_LANDMARKS.items():
                        region_trackers[name].update(landmarks, indices)

                    draw_skeleton(frame, landmarks, region_trackers)
                else:
                    for tracker in region_trackers.values():
                        tracker.mark_unreliable()

                overall_state = get_overall_state(region_trackers)
                if csv_logger is not None and csv_logger.recording:
                    csv_logger.write_frame(
                        timestamp_ms,
                        frame_number,
                        current_trial,
                        region_trackers,
                        overall_state,
                    )

                draw_status_panel(
                    frame,
                    region_trackers,
                    csv_logger,
                    current_trial,
                )
                cv2.imshow("Body Movement Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("r"), ord("R")) and csv_logger is not None:
                    csv_logger.recording = not csv_logger.recording
                    status = "ON" if csv_logger.recording else "OFF"
                    print(f"Recording: {status}")
                elif key in TRIAL_KEY_BINDINGS:
                    current_trial_number = TRIAL_KEY_BINDINGS[key]
                    current_trial = TRIAL_LABELS[current_trial_number]
                    print(f"Trial selected: {current_trial}")
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if csv_logger is not None:
            csv_logger.close()


if __name__ == "__main__":
    main()
