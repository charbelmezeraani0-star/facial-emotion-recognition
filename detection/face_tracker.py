"""
Multi-face tracker with temporal emotion smoothing.

Tracks each face across frames using IoU matching.
Smooths emotion predictions over a sliding window so the
label doesn't flicker every frame.
"""
import numpy as np
import collections


def iou(box_a, box_b):
    """Intersection-over-Union between two (x,y,w,h) boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = box_a[2]*box_a[3] + box_b[2]*box_b[3] - inter
    return inter / (union + 1e-6)


class FaceTrack:
    _next_id = 1

    def __init__(self, box, smoothing=8):
        self.id          = FaceTrack._next_id
        FaceTrack._next_id += 1
        self.box         = box
        self.missed      = 0
        # Sliding window of probability vectors for temporal smoothing
        self._probs_hist = collections.deque(maxlen=smoothing)

    def update(self, box, probs):
        self.box    = box
        self.missed = 0
        self._probs_hist.append(probs)

    @property
    def smooth_probs(self):
        if not self._probs_hist:
            return np.ones(7) / 7
        # Exponential weighting — recent frames count more
        weights = np.exp(np.linspace(-1, 0, len(self._probs_hist)))
        weights /= weights.sum()
        return np.average(list(self._probs_hist), axis=0, weights=weights)

    @property
    def emotion(self):
        from detection.face_detector import EMOTIONS
        return EMOTIONS[self.smooth_probs.argmax()]

    @property
    def confidence(self):
        return float(self.smooth_probs.max())


class MultiTracker:
    def __init__(self, iou_threshold=0.4, max_missed=10, smoothing=8):
        self.tracks        = []
        self.iou_thresh    = iou_threshold
        self.max_missed    = max_missed
        self.smoothing     = smoothing

    def update(self, detections, probs_list):
        """
        detections : list of (x,y,w,h)
        probs_list : list of np.array(7,) softmax probabilities
        Returns    : list of FaceTrack (one per detection)
        """
        matched_track_ids = set()
        matched_det_ids   = set()
        result_tracks     = []

        # Match detections to existing tracks by IoU
        for di, (box, probs) in enumerate(zip(detections, probs_list)):
            best_iou, best_track = 0, None
            for track in self.tracks:
                score = iou(box, track.box)
                if score > best_iou:
                    best_iou, best_track = score, track

            if best_iou > self.iou_thresh and best_track is not None:
                best_track.update(box, probs)
                matched_track_ids.add(id(best_track))
                matched_det_ids.add(di)
                result_tracks.append(best_track)

        # Create new tracks for unmatched detections
        for di, (box, probs) in enumerate(zip(detections, probs_list)):
            if di not in matched_det_ids:
                track = FaceTrack(box, smoothing=self.smoothing)
                track.update(box, probs)
                self.tracks.append(track)
                result_tracks.append(track)

        # Age unmatched tracks and remove stale ones
        for track in self.tracks:
            if id(track) not in matched_track_ids:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return result_tracks
