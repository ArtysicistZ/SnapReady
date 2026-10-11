"""Module 2 - Guide & capture: rules -> ONE instruction -> automatic capture.

Each rule limits one measurement from check.py with "min", "max" or "max_abs",
e.g. {"max_abs": 11} on head_tilt. The limits live in config.json (fitted by benchmark.py).
Background and cropping are not rules: finish.py fixes those in software.
"""

from collections import deque

import numpy as np

# Rules in the order the user should fix them.
RULE_ORDER = [
    "distance_cm", "face_height", "offset_x", "offset_y", "head_turn", "head_nod",
    "head_tilt", "eye_closure", "brightness", "shadow_ratio", "sharpness",
]
FIXED_INSTRUCTIONS = {
    "face_height": "Move a little closer",
    "head_tilt": "Straighten your head",
    "eye_closure": "Open your eyes",
    "shadow_ratio": "Uneven light - turn toward the light",
    "sharpness": "Hold still - the image is blurry",
}


def passes(value, rule):
    """True if value is within the rule's limits. NaN (could not measure) fails. Works on arrays."""
    value = np.asarray(value, dtype=float)
    if "max_abs" in rule:
        return np.abs(value) <= rule["max_abs"]
    return (value >= rule.get("min", -np.inf)) & (value <= rule.get("max", np.inf))


def instruction(name, value, rule):
    """What to tell the user when rule `name` fails. The camera faces the user,
    so the image's right is the user's left."""
    if name == "distance_cm":
        return f"Step back - {value:.0f} cm is too close"
    if name == "offset_x":
        return "Move slightly to your right" if value > 0 else "Move slightly to your left"
    if name == "offset_y":
        return "Move down a little" if value < rule["min"] else "Move up a little"
    if name == "head_turn":
        return "Turn your head slightly right" if value > 0 else "Turn your head slightly left"
    if name == "head_nod":
        return "Raise your chin a little" if value > rule["max"] else "Lower your chin a little"
    if name == "brightness":
        if value < rule["min"]:
            return "Too dark - add light in front of you"
        return "Too bright - move away from the light"
    return FIXED_INSTRUCTIONS[name]


def evaluate(measurements, rules):
    """List of (rule name, passed, instruction), in fixing order."""
    if measurements is None:
        return [("face", False, "Move into the frame")]
    results = []
    for name in RULE_ORDER:
        value = measurements[name]
        if value is None:          # not measured, e.g. distance when the camera is unknown
            continue
        passed = bool(passes(value, rules[name]))
        results.append((name, passed, instruction(name, value, rules[name])))
    return results


class Coach:
    """Smooths pass/fail over recent frames and decides when to take the photo."""

    SMOOTH = 5    # a rule passes if it passed in most of the last SMOOTH frames (no flicker)
    HOLD = 20     # capture after HOLD frames in a row where every rule passes

    def __init__(self, rules):
        self.rules = rules
        self.reset()

    def reset(self):
        self.history = {}      # rule name -> recent pass/fail values
        self.burst = []        # (frame, measurements) of the current passing streak
        self.streak = 0

    def update(self, frame, measurements):
        """Returns (checks, instruction or None if all pass, (frame, measurements) to capture or None)."""
        if measurements is None:
            self.history.clear()

        checks = []
        for name, passed, text in evaluate(measurements, self.rules):
            recent = self.history.setdefault(name, deque(maxlen=self.SMOOTH))
            recent.append(passed)
            checks.append((name, sum(recent) > len(recent) / 2, text))

        failing = [text for name, passed, text in checks if not passed]
        if failing:
            self.streak = 0
            self.burst = []
            return checks, failing[0], None

        self.streak += 1
        self.burst.append((frame, measurements))
        if self.streak < self.HOLD:
            return checks, None, None
        sharpest = max(self.burst, key=lambda item: item[1]["sharpness"])
        return checks, None, sharpest
