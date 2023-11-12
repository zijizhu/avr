import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

from utils import parse_panels, bbox_to_xyxy, parse_rules

# Constants

configurations = [
    'center_single',
    'distribute_four',
    'distribute_nine',
    'in_center_single_out_center_single',
    'in_distribute_four_out_center_single',
    'left_center_single_right_center_single',
    'up_center_single_down_center_single'
]
id2type = ['none', 'triangle', 'square', 'pentagon', 'hexagon', 'circle']
id2size = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
id2color = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
id2slot_all = [(0.5, 0.5, 0.33, 0.33), (0.42, 0.42, 0.15, 0.15), (0.25, 0.75, 0.5, 0.5), (0.5, 0.75, 0.5, 0.5),
           (0.5, 0.5, 1, 1), (0.75, 0.5, 0.5, 0.5), (0.58, 0.58, 0.15, 0.15), (0.83, 0.83, 0.33, 0.33),
           (0.83, 0.16, 0.33, 0.33), (0.42, 0.58, 0.15, 0.15), (0.83, 0.5, 0.33, 0.33), (0.16, 0.5, 0.33, 0.33),
           (0.75, 0.25, 0.5, 0.5), (0.5, 0.83, 0.33, 0.33), (0.58, 0.42, 0.15, 0.15), (0.5, 0.25, 0.5, 0.5),
           (0.16, 0.83, 0.33, 0.33), (0.16, 0.16, 0.33, 0.33), (0.5, 0.16, 0.33, 0.33), (0.25, 0.5, 0.5, 0.5),
           (0.75, 0.75, 0.5, 0.5), (0.25, 0.25, 0.5, 0.5)]
slot2id_all = {slot: idx for idx, slot in enumerate(id2slot_all)}

id2slot_center_single = [(0.5, 0.5, 1, 1)]
slot2id_center_single = {slot: idx for idx, slot in enumerate(id2slot_center_single)}
id2slot_distribute4 = [(0.25, 0.25, 0.5, 0.5), (0.25, 0.75, 0.5, 0.5), (0.75, 0.25, 0.5, 0.5), (0.75, 0.75, 0.5, 0.5)]
slot2id_distribute4 = {slot: idx for idx, slot in enumerate(id2slot_distribute4)}
id2slot_distribute9 = [(0.16, 0.16, 0.33, 0.33), (0.16, 0.5, 0.33, 0.33), (0.16, 0.83, 0.33, 0.33),
                       (0.5, 0.16, 0.33, 0.33), (0.5, 0.5, 0.33, 0.33), (0.5, 0.83, 0.33, 0.33),
                       (0.83, 0.16, 0.33, 0.33), (0.83, 0.5, 0.33, 0.33), (0.83, 0.83, 0.33, 0.33)]
slot2id_distribute9 = {slot: idx for idx, slot in enumerate(id2slot_distribute9)}


def panel_dict_to_df(indices: list | range, panel_dict: dict, file_path: str, slot2id: dict = slot2id_all):
    full_panel_data = []
    for panel_idx, panel in zip(indices, panel_dict):
        for component in panel['components']:
            component_idx = component['component']['id']
            comp_slots_data = {}
            for entity in component['entities']:
                entity_size = eval(entity['Size'])
                entity_type = eval(entity['Type'])
                entity_color = eval(entity['Color'])
                entity_slot_id = slot2id[tuple(eval(entity['bbox']))]
                comp_slots_data.update(
                    {entity_slot_id: {'size': entity_size, 'type': entity_type, 'color': entity_color}})

            data = {'file': str(file_path), 'panel': int(panel_idx), 'component': int(component_idx)}
            data_pd = data.copy()

            for slot_idx in range(0, len(slot2id)):
                if slot_idx in comp_slots_data:
                    data_pd.update({f'slot{slot_idx}_color': comp_slots_data[slot_idx]['color'],
                                    f'slot{slot_idx}_size': comp_slots_data[slot_idx]['size'],
                                    f'slot{slot_idx}_type': comp_slots_data[slot_idx]['type']})
                else:
                    data_pd.update(
                        {f'slot{slot_idx}_color': -1, f'slot{slot_idx}_size': -1, f'slot{slot_idx}_type': -1})
            full_panel_data.append(data_pd)
    return pd.DataFrame(full_panel_data)


def extract_stage3_ground_truth(dataset_dir: str, split: str, configs, all_panels=True):
    if configs == 'all':
        configs = configurations
    dataset_path = Path(dataset_dir)
    all_file_stems = list(fn.stem for fn in (dataset_path / Path(configs[0])).glob(f'*_{split}.npz'))
    all_file_paths = [Path(dataset_path, config, base_fn)
                      for config, base_fn in
                      product(configs, all_file_stems)]

    all_panel_df = []
    full_rule_data = []
    full_target_data = []

    for file_path in all_file_paths:
        xml = ET.parse(file_path.with_suffix('.xml'))
        npz = np.load(file_path.with_suffix('.npz'))
        xml_root = xml.getroot()
        panel_info_list = parse_panels(xml_root)
        component_rules = parse_rules(xml_root)
        panels = panel_info_list if all_panels else panel_info_list[6:]
        
        full_target_data.append({'file': str(file_path), 'target': npz['target'].item()})

        # Get rules (labels)
        rule_data = {'file': str(file_path)}
        for component in component_rules:
            cid = int(component['component_id'])
            for rule in component['rules']:
                if (rule['attr'] == 'Number/Position') or (rule['attr'] == 'Number') or (rule['attr'] == 'Position'):
                    rule_data[f'component{cid}_number'] = rule['name']
                    rule_data[f'component{cid}_position'] = rule['name']
                elif rule['attr'] == 'Type':
                    rule_data[f'component{cid}_type'] = rule['name']
                elif rule['attr'] == 'Size':
                    rule_data[f'component{cid}_size'] = rule['name']
                elif rule['attr'] == 'Color':
                    rule_data[f'component{cid}_color'] = rule['name']
        full_rule_data.append(rule_data)

        # Get discrete panel representations (features)
        panel_idx_range = range(16) if all_panels else range(6, 16)
        slot_list = []
        if 'center_single' in configs:
            slot_list += id2slot_center_single
        if 'distribute_four' in configs:
            slot_list += id2slot_distribute4
        if 'distribute_nine' in configs:
            slot_list += id2slot_distribute9
        slot2id = {slot: idx for idx, slot in enumerate(slot_list)}
        panel_df = panel_dict_to_df(panel_idx_range, panels, str(file_path), slot2id=slot2id)
        all_panel_df.append(panel_df)

    return (pd.concat(all_panel_df).reset_index(drop=True),
            pd.DataFrame(full_rule_data),
            pd.DataFrame(full_target_data))