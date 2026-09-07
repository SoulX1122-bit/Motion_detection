"""Simple, independent movement detection for named pose regions."""

from collections import deque
import math

from landmark_mapper import BODY_REGIONS


class RegionMovementDetector:
    """Measure average 2-D landmark displacement over a recent frame window.

    A result must stay above ``movement_threshold`` for ``persistence_frames``
    frames before its region is marked MOVING.  This is intentionally a small
    first version: later stages can add smoothing and separate enter/exit
    thresholds without changing the camera or pose-detection code.
    """

    def __init__(
        self,
        history_size=15,
        movement_threshold=0.012,
        persistence_frames=3,
    ):
        self.history_size = history_size
        self.movement_threshold = movement_threshold
        self.persistence_frames = persistence_frames
        self.position_history = {
            region_name: deque(maxlen=history_size)
            for region_name in BODY_REGIONS
        }
        self.moving_frame_counts = {
            region_name: 0 for region_name in BODY_REGIONS
        }

    def update(self, landmarks):
        """Update every region and return its numeric score and state.

        ``landmarks`` is MediaPipe's list of normalized pose landmarks.  Scores
        are normalized image-coordinate units, not pixels, so they work at
        different camera resolutions.  The threshold must still be calibrated
        for the subject, camera placement, and desired sensitivity.
        """
        results = {}

        for region_name, landmark_indices in BODY_REGIONS.items():
            current_position = self._region_center(landmarks, landmark_indices)
            history = self.position_history[region_name]
            history.append(current_position)

            score = self._movement_score(history)
            is_above_threshold = score >= self.movement_threshold

            if is_above_threshold:
                self.moving_frame_counts[region_name] += 1
            else:
                self.moving_frame_counts[region_name] = 0

            is_moving = (
                self.moving_frame_counts[region_name]
                >= self.persistence_frames
            )
            results[region_name] = {
                "score": score,
                "is_moving": is_moving,
            }

        return results

    @staticmethod
    def _region_center(landmarks, landmark_indices):
        """Return the mean normalized x/y position for one body region."""
        x_total = sum(landmarks[index].x for index in landmark_indices)
        y_total = sum(landmarks[index].y for index in landmark_indices)
        landmark_count = len(landmark_indices)
        return x_total / landmark_count, y_total / landmark_count

    @staticmethod
    def _movement_score(history):
        """Return mean displacement between consecutive saved positions."""
        if len(history) < 2:
            return 0.0

        distances = []
        for previous, current in zip(history, list(history)[1:]):
            delta_x = current[0] - previous[0]
            delta_y = current[1] - previous[1]
            distances.append(math.hypot(delta_x, delta_y))

        return sum(distances) / len(distances)
