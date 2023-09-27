import json
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import parse_panels, bbox_to_xyxy


class AVRStage1Dataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            split: str = 'train',
            transform = None,
            target_transform = None):
        assert split in ['train', 'val', 'test']

        self.split = split
        self.configurations = [
            'center_single',
            'distribute_four',
            'distribute_nine',
            'in_center_single_out_center_single',
            'in_distribute_four_out_center_single',
            'left_center_single_right_center_single',
            'up_center_single_down_center_single'
        ]
        self.dataset_path = Path(dataset_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.all_file_stems = list(fn.stem for fn in (self.dataset_path / Path(self.configurations[0])).glob(f'*_{self.split}.npz'))
        self.all_file_paths = [Path(self.dataset_path, config, base_fn) for config, base_fn in product(self.configurations, self.all_file_stems)]

        self.id2type = ['none', 'triangle', 'square', 'pentagon', 'hexagon', 'circle']
        self.id2size = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.id2color = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]

    def get_panel_by_id(self, idx: int):
        file_idx, panel_idx = int(idx / 16), int(idx % 16)
        return self.all_file_paths[file_idx], panel_idx

    def __len__(self):
        return len(self.all_file_paths) * 16

    def __getitem__(self, idx):
        file_path, panel_idx = self.get_panel_by_id(idx)

        npz = np.load(file_path.with_suffix('.npz'))
        image = torch.as_tensor(npz['image'][panel_idx, :, :], dtype=torch.float) / 255
        image = torch.unsqueeze(image, dim=0)

        targets = {}
        boxes = []
        types = []
        sizes = []
        colors = []
        xml = ET.parse(file_path.with_suffix('.xml'))
        xml_root = xml.getroot()
        panel_info_list = parse_panels(xml_root)
        panel_info = panel_info_list[panel_idx]
        all_entities = []
        for component in panel_info['components']:
            all_entities += component['entities']
        for entity in all_entities:
            boxes.append(bbox_to_xyxy(json.loads(entity['real_bbox'])))
            types.append(int(entity['Type']))
            sizes.append(int(entity['Size']))
            colors.append(int(entity['Color']))

        targets['boxes'] = torch.as_tensor(boxes, dtype=torch.float)
        targets['types'] = torch.as_tensor(types, dtype=torch.int64)
        targets['labels'] = torch.as_tensor(types, dtype=torch.int64)
        targets['sizes'] = torch.as_tensor(sizes, dtype=torch.int64)
        targets['colors'] = torch.as_tensor(colors, dtype=torch.int64)
        targets['image_id'] = torch.tensor(idx)
        # targets['image'] = image

        return image, targets


def collate_fn(batch):
    return tuple(zip(*batch))
