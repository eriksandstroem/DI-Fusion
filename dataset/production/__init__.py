import numpy as np

# Association function for CoRBS
def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation
    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


# read colmap binary depth images
def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)

    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


class FrameIntrinsic:
    def __init__(self, fx, fy, cx, cy, dscale):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.dscale = dscale

    def to_K(self):
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


class FrameData:
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.gt_pose = None
        self.calib = None


class RGBDSequence:
    def __init__(self):
        self.frame_id = 0

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError
