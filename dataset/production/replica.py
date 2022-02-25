import cv2
import os
import torch
from dataset.production import *
from pyquaternion import Quaternion
from pathlib import Path
from utils import motion_util

# the fusion code expects that the camera coordinate system is such that z is in the
# camera viewing direction, y is down and x is to the right. This is achieved by a serie of rotations
rot_180_around_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).astype(np.float32)
rot_90_around_y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).astype(np.float32)
rot_180_around_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(np.float32)
rot_90_around_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype(np.float32)
rot_270_around_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).astype(np.float32)
rot_90_around_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(np.float32)
rot_180_around_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float32)
rotation = np.matmul(rot_180_around_z, rot_180_around_y)
cano_quat = motion_util.Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))


class REPLICASequence(RGBDSequence):
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

        color_names1 = [f"left_rgb/{t}" for t in os.listdir(self.path / "left_rgb")]

        color_names2 = [f"left_rgb/{t}" for t in os.listdir(self.path / "left_rgb")]
        self.color_names = sorted(
            color_names1 + color_names2,
            key=lambda t: int(t.split("/")[-1].split(".")[0]),
        )
        # self.color_names = self.color_names[::8]

        depth_names1 = [
            f"left_routing_refined_psmnet/{t}"
            for t in os.listdir(self.path / "left_routing_refined_psmnet")
        ]

        depth_names2 = [
            f"left_routing_refined_sgm/{t}"
            for t in os.listdir(self.path / "left_routing_refined_sgm")
        ]

        self.depth_names = sorted(
            depth_names1 + depth_names2,
            key=lambda t: int(t.split("/")[-1].split(".")[0]),
        )
        # self.depth_names = self.depth_names[::8]

        self.calib = [128.0, 128.0, 128.0, 128.0, 1000.0]
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
            gt_traj_path = self.path / "left_camera_matrix"
            self.gt_trajectory = self._parse_traj_file(gt_traj_path)
            self.gt_trajectory = self.gt_trajectory[start_frame:end_frame]
            # change_iso = self.first_iso.dot(self.gt_trajectory[0].inv())
            # self.gt_trajectory = [change_iso.dot(t) for t in self.gt_trajectory]
            assert len(self.gt_trajectory) == len(self.color_names)
        else:
            self.gt_trajectory = None

    def _parse_traj_file(self, traj_path):
        camera_ext = {}
        traj_list = sorted(
            [t for t in os.listdir(traj_path)],
            key=lambda t: int(t.split(".")[0]),
        )

        for i, pose in enumerate(traj_list):
            i = i * 2
            # print(traj_path / pose)
            # print(i)
            pose_data = np.loadtxt(traj_path / pose).astype(np.float32)
            rotation = pose_data[0:3, 0:3]
            translation = pose_data[0:3, -1]
            iso = motion_util.Isometry(
                q=Quaternion(matrix=rotation, atol=1e-07, rtol=1e-07),
                t=translation,
            )
            rot = motion_util.Isometry(
                q=Quaternion(matrix=rot_180_around_x, atol=1e-07, rtol=1e-07),
            )
            iso = iso.dot(rot)
            pose_data = iso.matrix

            # pose_data = np.linalg.inv(pose_data)
            # pose_data = np.matmul(np.eye(3), pose_data[0:3, 0:4])
            # pose_data = np.linalg.inv(
            #     np.concatenate((pose_data, np.array([[0, 0, 0, 1]])), axis=0)
            # )
            # pose_data = np.matmul(pose_data[])
            # pose_data = np.matmul(
            #     rot_90_around_x, pose_data[0:3, 0:4]
            # )  # does not matter. Only in global frame so entire reconstruction rotates.
            # rot = pose_data[:3, :3]
            # cur_t = pose_data[:3, -1]

            # cur_q = rot
            # cur_q[1] = -cur_q[1]
            # cur_q[:, 1] = -cur_q[:, 1]
            # cur_t[1] = -cur_t[1]
            # cur_iso = motion_util.Isometry(
            #     q=Quaternion(matrix=cur_q, atol=1e-07, rtol=1e-07),
            #     t=cur_t,
            # )
            # camera_ext[i] = cano_quat.dot(cur_iso)
            camera_ext[i] = motion_util.Isometry(
                q=Quaternion(matrix=pose_data[:3, :3], atol=1e-03, rtol=1e-03),
                t=pose_data[:3, -1],
            )
            camera_ext[i + 1] = motion_util.Isometry(
                q=Quaternion(matrix=pose_data[:3, :3], atol=1e-03, rtol=1e-03),
                t=pose_data[:3, -1],
            )

        cam_list = [camera_ext[t] for t in range(len(camera_ext))]
        # cam_list = cam_list[::8]

        return cam_list

    def __len__(self):
        return len(self.color_names)

    def __next__(self):
        if self.frame_id >= len(self):
            raise StopIteration

        depth_img_path = self.path / self.depth_names[self.frame_id]
        rgb_img_path = self.path / self.color_names[self.frame_id]
        # print(depth_img_path)
        # print(rgb_img_path)
        # Convert depth image into point cloud.
        depth_data = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        # downsample to 256x256
        depth_data = depth_data[::2, ::2]
        depth_data = (
            torch.from_numpy(depth_data.astype(np.float32)).cuda() / self.calib[4]
        )
        depth_data[0:10, :] = 0
        depth_data[-10:-1, :] = 0
        depth_data[:, 0:10] = 0
        depth_data[:, -10:-1] = 0

        depth_data = depth_data
        rgb_data = cv2.imread(str(rgb_img_path))
        # downsample to 256x256
        rgb_data = rgb_data[::2, ::2]
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.0

        frame_data = FrameData()
        frame_data.gt_pose = (
            self.gt_trajectory[self.frame_id]
            if self.gt_trajectory is not None
            else None
        )
        frame_data.calib = FrameIntrinsic(
            self.calib[0], self.calib[1], self.calib[2], self.calib[3], self.calib[4]
        )
        frame_data.depth = depth_data
        frame_data.rgb = rgb_data

        self.frame_id += 1
        return frame_data
