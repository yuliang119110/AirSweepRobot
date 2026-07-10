# Stub for decord (no aarch64 wheel available).
# LocateAnything only uses decord for video processing; image inference works without it.

class VideoReader:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("decord VideoReader is stubbed out on aarch64. Video input is not supported.")

    def get_avg_fps(self):
        return 0

    def __len__(self):
        return 0

    def get_batch(self, *args, **kwargs):
        import numpy as np
        return np.array([])
