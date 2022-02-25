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


class SCENE3DSequence(RGBDSequence):
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

        # self.color_names = sorted(
        #     [f"images/{t}" for t in os.listdir(self.path / "images")],
        #     key=lambda t: int(t.split("/")[-1].split(".")[0]),
        # )

        color_names1 = [f"images/{t}" for t in os.listdir(self.path / "images")]

        color_names2 = [f"images/{t}" for t in os.listdir(self.path / "images")]
        self.color_names = sorted(
            color_names1 + color_names2,
            key=lambda t: int(t.split("/")[-1].split(".")[0]),
        )

        # self.depth_names = sorted(
        #     [
        #         f"copyroom_png/depth/{t}"
        #         for t in os.listdir(self.path / "copyroom_png/depth")
        #     ],
        #     key=lambda t: int(t.split("/")[-1].split(".")[0]),
        # )
        # self.depth_names = sorted(
        #     [
        #         f"dense/stereo/depth_maps/{t}"
        #         for t in os.listdir(self.path / "dense/stereo/depth_maps")
        #     ],
        #     key=lambda t: int(t.split("/")[-1].split(".")[0]),
        # )

        depth_names1 = [
            f"dense/stereo/depth_maps/{t}"
            for t in os.listdir(self.path / "dense/stereo/depth_maps/")
        ]

        depth_names2 = [
            f"copyroom_png/depth/{t}"
            for t in os.listdir(self.path / "copyroom_png/depth")
        ]

        self.depth_names = sorted(
            depth_names1 + depth_names2,
            key=lambda t: int(t.split("/")[-1].split(".")[0]),
        )

        if first_tq is not None:
            self.first_iso = motion_util.Isometry(
                q=Quaternion(array=first_tq[3:]), t=np.array(first_tq[:3])
            )
        else:
            self.first_iso = motion_util.Isometry(
                q=Quaternion(array=[0.0, -1.0, 0.0, 0.0])
            )

        if end_frame == -1:
            end_frame = len(self.color_names)

        self.color_names = self.color_names[start_frame:end_frame]

        self.depth_names = self.depth_names[start_frame:end_frame]

        if load_gt:
            gt_traj_path = self.path / "copyroom_trajectory.log"
            self.gt_trajectory = self._parse_traj_file(gt_traj_path)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            # change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
            # self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            assert len(self.gt_trajectory) == len(self.color_names)
        else:
            self.gt_trajectory = None

    def _parse_traj_file(self, traj_path):
        def grouper_it(n, iterable):
            it = iter(iterable)
            while True:
                chunk_it = itertools.islice(it, n)
                try:
                    first_el = next(chunk_it)
                except StopIteration:
                    return
                yield itertools.chain((first_el,), chunk_it)

        cameras = dict()

        with open(traj_path, "r") as traj_file:
            chunk_iterable = grouper_it(5, traj_file)
            for i, frame in enumerate(chunk_iterable):
                frame_id = next(frame)[:-1]
                frame_id = re.split(r"\t+", frame_id.rstrip("\t"))[-1]
                first = np.fromstring(next(frame), count=4, sep=" ", dtype=float)
                second = np.fromstring(next(frame), count=4, sep=" ", dtype=float)
                third = np.fromstring(next(frame), count=4, sep=" ", dtype=float)
                fourth = np.fromstring(next(frame), count=4, sep=" ", dtype=float)

                extrinsics = np.zeros((4, 4))
                extrinsics[0, :] = first
                extrinsics[1, :] = second
                extrinsics[2, :] = third
                extrinsics[3, :] = fourth

                # code to change the camera pose convention. Should not be necessary
                # and is thus commented
                # extrinsics = np.linalg.inv(extrinsics)
                # extrinsics = np.matmul(rot_180_around_x, extrinsics[0:3, 0:4])
                # extrinsics = np.linalg.inv(
                #   np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
                # )

                pose_data = extrinsics
                # rot = pose_data[:3, :3]
                cur_t = pose_data[:3, -1]

                # cur_q = rot
                # cur_q[1] = -cur_q[1]
                # cur_q[:, 1] = -cur_q[:, 1]
                # cur_t[1] = -cur_t[1] # commented out this one!
                # cur_iso = motion_util.Isometry(
                #     q=Quaternion(matrix=cur_q, atol=1e-03, rtol=1e-03),
                #     t=cur_t,
                # )
                # cameras[i] = cano_quat.dot(cur_iso)
                cameras[i] = motion_util.Isometry(
                    q=Quaternion(matrix=pose_data[0:3, 0:3], atol=1e-03, rtol=1e-03),
                    t=cur_t,
                )

        # downsample camera dict (as camera dict contains the original dataset while I have
        # downsampled the dataset 10 times in # of images).
        new_cameras = dict()
        a = 0
        for i, key in enumerate(cameras.keys()):
            if i % 10 == 0:
                new_cameras[i // 10 + a] = cameras[i]
                new_cameras[i // 10 + 1 + a] = cameras[i]
                a += 1

        return [new_cameras[t] for t in range(len(new_cameras))]

    def __len__(self):
        return len(self.color_names)

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        depth_img_path = self.path / self.depth_names[self.frame_id]
        rgb_img_path = self.path / self.color_names[self.frame_id]

        if str(depth_img_path).endswith(".png"):  # ToF
            calib = [
                525.0 * 256 / 640,
                525.0 * 256 / 480,
                319.5 * 256 / 640,
                239.5 * 256 / 480,
                1000.0,
            ]
        else:  # MVS
            calib = [
                525.0 * 256 / 640,
                525.0 * 256 / 480,
                319.5 * 256 / 640,
                239.5 * 256 / 480,
                1.0,
            ]
        # print(depth_img_path)
        # print(rgb_img_path)
        # print(self.frame_id)
        # print(self.frame_id * 10)
        # print(len(self.gt_trajectory))

        # Convert depth image into point cloud.

        if str(depth_img_path).endswith(".png"):  # ToF
            depth_data = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        else:  # MVS
            depth_data = read_array(str(depth_img_path))
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

        depth_data = depth_data
        rgb_data = cv2.imread(str(rgb_img_path))
        # downsample to 256x256
        step_x = rgb_data.shape[0] / 256
        step_y = rgb_data.shape[1] / 256

        index_y = [int(step_y * i) for i in range(0, int(rgb_data.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in range(0, int(rgb_data.shape[0] / step_x))]

        rgb_data = rgb_data[:, index_y]
        rgb_data = rgb_data[index_x, :]

        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.0

        frame_data = FrameData()
        frame_data.gt_pose = (
            self.gt_trajectory[self.frame_id]
            if self.gt_trajectory is not None
            else None
        )
        frame_data.calib = FrameIntrinsic(
            calib[0], calib[1], calib[2], calib[3], calib[4]
        )
        frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        self.frame_id += 1
        return frame_data
