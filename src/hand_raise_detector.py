from __future__ import annotations

from src.config import (
    HAND_RAISE_SECONDS_THRESHOLD,
    WRIST_SHOULDER_VERTICAL_THRESHOLD,
)
from src.definitions import LEFT_SHOULDER, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_WRIST


class HandRaiseDetector:
    """
    Detects hand-raising gestures from pose estimation keypoints.
    """

    def __init__(self, fps: float):
        """
        Initializes the detector with necessary thresholds from the config.
        Args:
            fps (float): The frames per second of the video.
        """
        self.fps = fps
        self.min_raise_frames = int(HAND_RAISE_SECONDS_THRESHOLD * self.fps)
        self.vertical_threshold = WRIST_SHOULDER_VERTICAL_THRESHOLD

        # State tracking for each person ID
        self.track_states: dict[int, dict] = {}

    def process_frame(self, frame_number: int, tracks: list[dict]) -> list[dict]:
        """
        Processes a single frame to detect hand raises for all tracked persons.
        Args:
            frame_number (int): The current frame number.
            tracks (list[dict]): A list of track data from the pose estimator.
                                 Each track should have "id" and "keypoints".
        Returns:
            list[dict]: A list of completed hand-raise events.
        """
        completed_events = []

        current_track_ids = {track["id"] for track in tracks}

        for track in tracks:
            track_id = track["id"]
            keypoints = track["keypoints"]

            if track_id not in self.track_states:
                self.track_states[track_id] = self._get_initial_state()

            person_events = self._check_person_hand_raise(frame_number, track_id, keypoints)
            completed_events.extend(person_events)

        # Clean up states for tracks that are no longer present
        for track_id in list(self.track_states.keys()):
            if track_id not in current_track_ids:
                # Finalize any ongoing events before deleting the state
                final_events = self._finalize_events(frame_number - 1, track_id)
                completed_events.extend(final_events)
                del self.track_states[track_id]

        return completed_events

    def _get_initial_state(self) -> dict:
        """Returns the initial state dictionary for a new track."""
        return {
            "left": {"is_raising": False, "start_frame": None},
            "right": {"is_raising": False, "start_frame": None},
        }

    def _check_person_hand_raise(self, frame_number: int, track_id: int, keypoints: list) -> list[dict]:
        """
        Checks and updates the hand-raising state for a single person.
        Returns a list of completed events for this person.
        """
        completed_events = []
        state = self.track_states[track_id]

        for side in ["left", "right"]:
            side_state = state[side]
            is_raised_now = self._is_hand_raised(keypoints, side)

            if is_raised_now and not side_state["is_raising"]:
                # Start of a potential hand raise
                side_state["is_raising"] = True
                side_state["start_frame"] = frame_number
            elif not is_raised_now and side_state["is_raising"]:
                # End of a hand raise
                duration = frame_number - side_state["start_frame"]
                if duration >= self.min_raise_frames:
                    completed_events.append(
                        {
                            "track_id": track_id,
                            "side": side,
                            "start_frame": side_state["start_frame"],
                            "end_frame": frame_number - 1,
                            "duration_frames": duration,
                        }
                    )
                side_state["is_raising"] = False
                side_state["start_frame"] = None

        return completed_events

    def _is_hand_raised(self, keypoints: list, side: str) -> bool:
        """
        Determines if a hand is raised based on keypoint positions.
        A hand is considered raised if the wrist is vertically above the shoulder.
        """
        if side == "left":
            wrist_idx, shoulder_idx = LEFT_WRIST, LEFT_SHOULDER
        else:
            wrist_idx, shoulder_idx = RIGHT_WRIST, RIGHT_SHOULDER

        # Ensure keypoints are valid before accessing them
        if (
            len(keypoints) > wrist_idx
            and len(keypoints) > shoulder_idx
            and keypoints[wrist_idx] is not None
            and keypoints[shoulder_idx] is not None
        ):
            wrist_y = keypoints[wrist_idx][1]
            shoulder_y = keypoints[shoulder_idx][1]

            # y-coordinate is 0 at the top.
            # A smaller y-value means a higher position.
            # self.vertical_threshold can be negative to allow some tolerance.
            return wrist_y < shoulder_y + self.vertical_threshold
        return False

    def _finalize_events(self, last_frame_number: int, track_id: int) -> list[dict]:
        """
        Finalizes any ongoing hand-raise events for a track that is ending.
        """
        completed_events = []
        if track_id not in self.track_states:
            return completed_events

        state = self.track_states[track_id]
        for side in ["left", "right"]:
            side_state = state[side]
            if side_state["is_raising"]:
                duration = last_frame_number - side_state["start_frame"] + 1
                if duration >= self.min_raise_frames:
                    completed_events.append(
                        {
                            "track_id": track_id,
                            "side": side,
                            "start_frame": side_state["start_frame"],
                            "end_frame": last_frame_number,
                            "duration_frames": duration,
                        }
                    )
        return completed_events
