import cv2
import os
import torch
from dataset.production import *
from pyquaternion import Quaternion
import itertools
import re
from pathlib import Path
from utils import motion_util

cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))

# various rotation matrices for debugging
rot_180_around_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).astype(np.float32)
rot_90_around_y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).astype(np.float32)
rot_180_around_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(np.float32)
rot_90_around_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype(np.float32)
rot_270_around_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).astype(np.float32)
rot_90_around_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(np.float32)
rot_180_around_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float32)
rotation = np.matmul(rot_270_around_z, rot_180_around_y)


class CORBSSequence(RGBDSequence):
    def __init__(
        self,
        path: str,
        start_frame: int = 0,
        end_frame: int = -1,
        first_tq: list = None,
        load_gt: bool = False,
    ):
        super().__init__()
        self.path = Path(path)

        trajectory_file = self.path / "data" / "H1_Trajectory" / "groundtruth.txt"
        rgb_file = self.path / "data" / "H1_pre_registereddata" / "rgb.txt"
        depth_file = self.path / "data" / "H1_pre_registereddata" / "depth.txt"
        self.stereo_path = self.path / "colmap" / "dense" / "stereo" / "depth_maps"
        self.tof_path = self.path / "data" / "H1_pre_registereddata"
        self.rgb_path = self.path / "data" / "H1_pre_registereddata"

        # read all files for pose, rgb, and depth
        self.poses = {}
        with open(trajectory_file, "r") as file:
            for line in file:
                # skip comment lines
                if line[0] == "#":
                    continue
                elems = line.rstrip().split(" ")
                timestamp = float(elems[0])
                pose = [float(e) for e in elems[1:]]
                self.poses[timestamp] = pose

        self.rgb_frames = {}
        with open(rgb_file, "r") as file:
            for line in file:
                # skip comment lines
                if line[0] == "#":
                    continue
                timestamp, file_path = line.rstrip().split(" ")
                timestamp = float(timestamp)
                self.rgb_frames[timestamp] = file_path

        self.depth_frames = {}
        with open(depth_file, "r") as file:
            for line in file:
                # skip comment lines
                if line[0] == "#":
                    continue
                timestamp, file_path = line.rstrip().split(" ")
                timestamp = float(timestamp)
                self.depth_frames[timestamp] = file_path

        # match pose to rgb timestamp
        rgb_matches = associate(
            self.poses, self.rgb_frames, offset=0.0, max_difference=0.02
        )
        # build mapping databases to get matches from pose timestamp to frame timestamp
        self.pose_to_rgb = {t_p: t_r for (t_p, t_r) in rgb_matches}

        # match poses that are matched with rgb to a corresponding depth timestamp
        depth_matches = associate(
            self.pose_to_rgb, self.depth_frames, offset=0.0, max_difference=0.02
        )
        # build mapping databases to get matches from pose timestamp to frame timestamp
        self.pose_to_depth = {t_p: t_d for (t_p, t_d) in depth_matches}
        self.poses_matched = {t_p: self.poses[t_p] for (t_p, t_r) in rgb_matches}

        if end_frame == -1:
            end_frame = len(self.poses_matched.keys())
        else:
            cleaned_poses_matched = dict()
            for i, key in enumerate(self.poses_matched.keys()):
                if i < end_frame:
                    cleaned_poses_matched[key] = self.poses_matched[key]
                else:
                    self.poses_matched = cleaned_poses_matched
                    break

    def __len__(self):
        return len(self.poses_matched.keys())

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        timestamp_pose = list(self.poses_matched.keys())[self.frame_id]
        timestamp_rgb = self.pose_to_rgb[timestamp_pose]
        timestamp_depth = self.pose_to_depth[timestamp_pose]

        # specify intrinsics
        if True:  # str(depth_img_path).endswith(".png"):  # ToF
            calib = [
                525.0 * 256 / 640,
                525.0 * 256 / 480,
                319.5 * 256 / 640,
                239.5 * 256 / 480,
                5000.0,
            ]
        else:  # MVS
            calib = [
                525.0 * 256 / 640,
                525.0 * 256 / 480,
                319.5 * 256 / 640,
                239.5 * 256 / 480,
                1.0,
            ]

        # read RGB frame
        rgb_file = os.path.join(
            self.rgb_path, self.rgb_frames[timestamp_rgb].replace("\\", "/")
        )
        rgb_data = cv2.imread(rgb_file).astype(np.float32)

        # downsample to 256x256
        step_x = rgb_data.shape[0] / 256
        step_y = rgb_data.shape[1] / 256

        index_y = [int(step_y * i) for i in range(0, int(rgb_data.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in range(0, int(rgb_data.shape[0] / step_x))]

        rgb_data = rgb_data[:, index_y]
        rgb_data = rgb_data[index_x, :]

        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.0

        # read kinect depth file
        depth_file = os.path.join(
            self.tof_path, self.depth_frames[timestamp_depth].replace("\\", "/")
        )
        # read colmap stereo depth file
        # depth_file = os.path.join(
        #     self.stereo_path,
        #     self.rgb_frames[timestamp_rgb].replace("rgb\\", "") + ".geometric.bin",
        # )

        # Convert depth image into point cloud.
        if True:  # str(depth_img_path).endswith(".png"):  # ToF
            depth_data = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        else:  # MVS
            depth_data = read_array(str(depth_file))

        # downsample to 256x256
        step_x = depth_data.shape[0] / 256
        step_y = depth_data.shape[1] / 256

        index_y = [int(step_y * i) for i in range(0, int(depth_data.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in range(0, int(depth_data.shape[0] / step_x))]

        depth_data = depth_data[:, index_y]
        depth_data = depth_data[index_x, :]

        depth_data[0:10, :] = 0
        depth_data[-10:-1, :] = 0
        depth_data[:, 0:10] = 0
        depth_data[:, -10:-1] = 0

        depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / calib[4]

        frame_data = FrameData()

        # load extrinsics
        rotation = self.poses_matched[timestamp_pose][3:]
        rotation = Quaternion(rotation[-1], rotation[0], rotation[1], rotation[2])
        translation = self.poses_matched[timestamp_pose][:3]

        frame_data.gt_pose = motion_util.Isometry(
            q=rotation,
            t=translation,
        )

        frame_data.calib = FrameIntrinsic(
            calib[0], calib[1], calib[2], calib[3], calib[4]
        )
        frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        self.frame_id += 1
        return frame_data
