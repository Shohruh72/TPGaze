import os

import cv2
import torch
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train
        self.gaze_idx, self.head_idx = 5, 5

        self.samples = self.load_label(self.args.data_dir, self.is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        face_img = cv2.imread(os.path.join(self.args.data_dir, 'Image', sample[0]))
        face_img = face_img / 255.0
        face_img = face_img.transpose(2, 0, 1)

        face_tensor = torch.from_numpy(face_img).float()
        head_pose_tensor = torch.tensor(sample[self.head_idx], dtype=torch.float32)

        img = {"face": face_tensor,
               "head_pose": head_pose_tensor,
               "name": sample[0]}

        label = torch.tensor(sample[self.gaze_idx], dtype=torch.float32)

        return img, label

    @staticmethod
    def load_label(data_dir, is_train):
        samples = []
        gaze_idx, head_idx = 5, 5
        label_file = 'train.label' if is_train else 'test.label'
        label_path = os.path.join(data_dir, 'Label', label_file)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        with open(label_path, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            for line in lines:
                line = line.split(' ')
                for i in range(gaze_idx, head_idx + 1):
                    line[i] = list(map(float, line[i].split(',')))
                samples.append(line)
        return samples
