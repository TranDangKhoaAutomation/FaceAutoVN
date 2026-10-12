"""Choose the first signup action from scored screen evidence.

The detector may emit unrelated page labels on the login screen. A strong,
well-placed button is more useful than requiring a brittle set of class names.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    state: str
    target: str
    x: float
    y: float
    score: float
    reason: str


class StartupPolicy:
    """Confirm each opening-screen action across fresh frames and tap it once."""

    def __init__(self, min_button_confidence=0.65, required_frames=2):
        self.min_button_confidence = min_button_confidence
        self.required_frames = required_frames
        self.last_sequence = -1
        self.stable_frames = 0
        self.last_candidate = None
        self.completed_target = None
        self.missing_frames = 0

    def observe(self, predictions, screen_size, sequence):
        if sequence == self.last_sequence:
            return None
        self.last_sequence = sequence

        width, height = screen_size
        if width <= 0 or height <= 0:
            return None

        candidates = []
        create = predictions.get("button_tao_tai_khoan_moi")
        if create:
            confidence = create["confidence"]
            x, y, box_width, _ = create["box"]
            if (confidence >= self.min_button_confidence
                    and y / height >= 0.67 and box_width / width >= 0.25):
                blockers = (
                    "input_ten", "input_ho", "input_ngay_thang_nam_sinh",
                    "input_email", "input_mat_khau", "input_ma_xac_nhan", "dong_y",
                )
                blocker_strength = max(
                    (predictions[label]["confidence"] for label in blockers if label in predictions),
                    default=0.0,
                )
                score = (
                    2.4 * confidence + 0.4
                    + 0.45 * predictions.get("page_dang_nhap", {}).get("confidence", 0.0)
                    + 0.2 * predictions.get("facebook", {}).get("confidence", 0.0)
                    + 0.2 * predictions.get("tham_gia_Facebook", {}).get("confidence", 0.0)
                    - 1.5 * blocker_strength
                )
                if score >= 2.0 and blocker_strength < 0.65:
                    candidates.append(("signup_start", "button_tao_tai_khoan_moi", x, y, score, confidence))

        start = predictions.get("button_start")
        if start:
            confidence = start["confidence"]
            x, y, box_width, _ = start["box"]
            page_conf = predictions.get("page_dang_nhap", {}).get("confidence", 0.0)
            facebook_conf = predictions.get("facebook", {}).get("confidence", 0.0)
            score = 2.0 * confidence + page_conf + 0.4 * facebook_conf + 0.4
            if (confidence >= 0.20 and page_conf >= 0.30 and facebook_conf >= 0.45
                    and y / height >= 0.75 and box_width / width >= 0.70 and score >= 1.3):
                candidates.append(("signup_welcome", "button_start", x, y, score, confidence))

        if not candidates:
            self.last_candidate = None
            self.stable_frames = 0
            self.missing_frames += 1
            if self.missing_frames >= 2:
                self.completed_target = None
            return None

        self.missing_frames = 0
        state, target, x, y, score, confidence = max(candidates, key=lambda item: item[4])
        if target == self.last_candidate:
            self.stable_frames += 1
        else:
            self.last_candidate = target
            self.stable_frames = 1
        if self.stable_frames < self.required_frames or target == self.completed_target:
            return None
        return Decision(
            state=state,
            target=target,
            x=x,
            y=y,
            score=score,
            reason=f"button={confidence:.2f}, frames={self.stable_frames}",
        )

    def mark_action_done(self, target):
        self.completed_target = target
