# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

# Keys that we have already warned about in concat_batch, so each missing key
# only produces a single warning per process (avoid log spam in the replay /
# demo batch pipeline).
_CONCAT_BATCH_WARNED_KEYS: set[str] = set()


def update_nested_cfg(base_cfg, override_cfg):
    for key, value in override_cfg.items():
        if (
            key in base_cfg
            and isinstance(base_cfg[key], dict)
            and isinstance(value, dict)
        ):
            update_nested_cfg(base_cfg[key], value)
        else:
            base_cfg[key] = value
    return base_cfg


def copy_dict_tensor(next_extracted_obs: dict):
    """
    Recursively clones all torch tensors in a dict.
    """
    ret = {}
    for key, value in next_extracted_obs.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.clone()
        elif isinstance(value, dict):
            ret[key] = copy_dict_tensor(value)
        else:
            ret[key] = value
    return ret


def put_tensor_device(data_dict, device):
    if data_dict is None:
        return None

    if isinstance(data_dict, torch.Tensor):
        return data_dict.to(device=device).contiguous()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_device(value, device)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device=device).contiguous()
    return data_dict


def split_dict_to_chunk(data: dict, split_size, dim=0):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = [
                chunk.contiguous() for chunk in torch.chunk(value, split_size, dim=dim)
            ]
        elif value is None:
            split_vs = [None for _ in range(split_size)]
        elif isinstance(value, dict):
            split_vs = split_dict_to_chunk(value, split_size, dim)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = (
                split_vs[split_id].contiguous()
                if isinstance(split_vs[split_id], torch.Tensor)
                else split_vs[split_id]
            )
    return splited_list


def concat_batch(data1, data2):
    batch = {}
    for key, value in data1.items():
        if isinstance(value, torch.Tensor):
            if key not in data2:
                # NOTE: NO WARNING FOR THE CASE THAT DATA2 DOES NOT CONTAIN SOME KEYS IN DATA1
                continue
            batch[key] = torch.cat([data1[key], data2[key]], dim=0)
        elif isinstance(value, dict):
            # NOTE: added this for dealing with different keys in demo data.
            if key not in data2:
                if key not in _CONCAT_BATCH_WARNED_KEYS:
                    _CONCAT_BATCH_WARNED_KEYS.add(key)
                    # Lazy import to avoid pulling rlinf.scheduler.worker (and
                    # its heavy deps) at module import time. This only runs
                    # once per missing key, inside a worker where that import
                    # is essentially free.
                    from rlinf.utils.logging import get_logger

                    get_logger().warning(
                        "concat_batch: key '%s' not found in data2 (value type: %s), "
                        "skipping. This warning is only emitted once per key.",
                        key,
                        type(value).__name__,
                    )
                continue
            batch[key] = concat_batch(data1[key], data2[key])
    return batch


def _pad_tensors_to_max_shape(tensors: list) -> list:
    # Pad tensors on the right of each dim to a common max shape so torch.stack
    # works across rollout epochs whose per-step features (e.g. language tokens)
    # have variable length. Pads with 0 (False for bool, 0 / 0.0 for int / float
    # — all match the conventional pad-token / pad-mask semantics).
    if len(tensors) <= 1:
        return tensors
    shapes = [tuple(t.shape) for t in tensors]
    if all(s == shapes[0] for s in shapes):
        return tensors
    rank = len(shapes[0])
    max_shape = [max(s[d] for s in shapes) for d in range(rank)]
    padded = []
    for t in tensors:
        if list(t.shape) == max_shape:
            padded.append(t)
            continue
        pad_spec = []
        for d in range(rank - 1, -1, -1):
            pad_spec.extend([0, max_shape[d] - t.shape[d]])
        padded.append(torch.nn.functional.pad(t, pad_spec, mode="constant", value=0))
    return padded


def stack_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            v_list = _pad_tensors_to_max_shape(v_list)
            ret[key] = torch.stack(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = stack_list_of_dict_tensor(v_list)
        elif _v0 is None:
            pass
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret


def cat_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = cat_list_of_dict_tensor(v_list)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret


def split_dict(
    batch: dict[str, Any],
    split_sizes: list[int],
) -> list[dict[str, Any]]:
    """Split one batch dict into size-specified sub-batches along dim-0.

    Tensor values are chunked on dim-0; list values are sliced proportionally;
    nested dict values are split recursively.

    Args:
        batch: Dict.
        split_sizes: Batch sizes for each destination rank.

    Returns:
        A list of splited batches, one item per destination rank.
    """
    count = len(split_sizes)
    total_size = sum(split_sizes)
    splitted_batches = [{} for _ in range(count)]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == total_size, (
                f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
            )
            splitted_values = torch.split(value, split_sizes, dim=0)
            for i in range(count):
                splitted_batches[i][key] = splitted_values[i].contiguous()
        elif isinstance(value, list):
            length = len(value)
            assert length == total_size, (
                f"List field '{key}' expected length {total_size}, got {length}."
            )
            begin = 0
            for i, size in enumerate(split_sizes):
                splitted_batches[i][key] = value[begin : begin + size]
                begin += size
        elif isinstance(value, dict):
            splitted_sub_batches = split_dict(value, split_sizes)
            for i in range(count):
                splitted_batches[i][key] = splitted_sub_batches[i]
        else:
            for i in range(count):
                splitted_batches[i][key] = value

    return splitted_batches
