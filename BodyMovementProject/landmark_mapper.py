"""Named MediaPipe Pose Landmarker regions.

The index values follow the 33-landmark Pose Landmarker definition.  Keeping
this mapping in one small module makes it easy to review or refine later.
"""

from enum import IntEnum


class PoseLandmark(IntEnum):
    """Only the landmark indices used by this first movement detector."""

    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
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


BODY_REGIONS = {
    "HEAD": (
        PoseLandmark.NOSE,
        PoseLandmark.LEFT_EYE,
        PoseLandmark.RIGHT_EYE,
        PoseLandmark.LEFT_EAR,
        PoseLandmark.RIGHT_EAR,
    ),
    "LEFT ARM": (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_WRIST,
    ),
    "RIGHT ARM": (
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.RIGHT_WRIST,
    ),
    "TORSO": (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.RIGHT_HIP,
    ),
    "LEFT LEG": (
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_KNEE,
        PoseLandmark.LEFT_ANKLE,
    ),
    "RIGHT LEG": (
        PoseLandmark.RIGHT_HIP,
        PoseLandmark.RIGHT_KNEE,
        PoseLandmark.RIGHT_ANKLE,
    ),
}
