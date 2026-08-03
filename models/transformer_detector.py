"""Lightweight DETR-style transformer detector for Waymo camera frames.

This deliberately small implementation is intended for local experiments and
architecture comparisons.  It has a convolutional image encoder, learned object
queries, a Transformer encoder/decoder, and set-based matching loss.  Unlike
YOLOv8 it has no anchors or NMS-dependent training targets.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_detector
from models.base_detector import BaseDetector


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized centre boxes to corner boxes."""
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack((cx - width / 2, cy - height / 2,
                        cx + width / 2, cy + height / 2), dim=-1)


def _generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise generalized IoU for corner-format boxes."""
    if not len(boxes1) or not len(boxes2):
        return boxes1.new_zeros((len(boxes1), len(boxes2)))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    enc_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_area = (enc_rb - enc_lt).clamp(min=0).prod(dim=-1)
    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


def _greedy_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic one-to-one cost matching without an optional SciPy dependency.

    A production DETR baseline should replace this with exact Hungarian matching.
    Greedy matching is deliberately explicit here so this experimental backend
    remains runnable in a minimal local environment.
    """
    if cost.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    remaining_queries = torch.ones(cost.shape[0], dtype=torch.bool, device=cost.device)
    remaining_targets = torch.ones(cost.shape[1], dtype=torch.bool, device=cost.device)
    queries: list[int] = []
    targets: list[int] = []
    for _ in range(min(cost.shape)):
        masked = cost.masked_fill(~remaining_queries[:, None], float('inf'))
        masked = masked.masked_fill(~remaining_targets[None, :], float('inf'))
        flat = int(masked.argmin())
        q, t = divmod(flat, cost.shape[1])
        if not torch.isfinite(masked[q, t]):
            break
        queries.append(q); targets.append(t)
        remaining_queries[q] = False; remaining_targets[t] = False
    return (torch.tensor(queries, dtype=torch.long, device=cost.device),
            torch.tensor(targets, dtype=torch.long, device=cost.device))


def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Small dependency-free NMS implementation for a single class."""
    keep: list[int] = []
    order = scores.argsort(descending=True)
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        lt = torch.maximum(boxes[current, :2], boxes[rest, :2])
        rb = torch.minimum(boxes[current, 2:], boxes[rest, 2:])
        inter = (rb - lt).clamp(min=0).prod(dim=1)
        area_current = (boxes[current, 2:] - boxes[current, :2]).clamp(min=0).prod()
        area_rest = (boxes[rest, 2:] - boxes[rest, :2]).clamp(min=0).prod(dim=1)
        iou = inter / (area_current + area_rest - inter).clamp(min=1e-6)
        order = rest[iou <= threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class _ConvEncoder(nn.Module):
    """Compact 1/16-resolution CNN feature extractor."""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        channels = [in_channels, hidden_dim // 2, hidden_dim // 2, hidden_dim, hidden_dim]
        layers: list[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers.extend((nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def _sine_position_encoding(height: int, width: int, dim: int,
                            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return (H*W, dim) 2-D sine/cosine positional encodings."""
    if dim % 4:
        raise ValueError('hidden_dim must be divisible by 4 for 2-D positional encoding.')
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing='ij')
    scale = 2 * math.pi
    y = y / max(height - 1, 1) * scale
    x = x / max(width - 1, 1) * scale
    dim_t = torch.arange(dim // 4, device=device, dtype=dtype)
    dim_t = 10000 ** (2 * torch.floor(dim_t / 2) / (dim // 2))
    pos_x = torch.stack((torch.sin(x[..., None] / dim_t), torch.cos(x[..., None] / dim_t)), dim=-1).flatten(-2)
    pos_y = torch.stack((torch.sin(y[..., None] / dim_t), torch.cos(y[..., None] / dim_t)), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1).reshape(height * width, dim)


class QueryTransformerDetector(nn.Module):
    """CNN + Transformer object-query detector shared by camera backends."""

    def __init__(self, num_classes: int, in_channels: int = 3, hidden_dim: int = 128,
                 num_queries: int = 100, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3, nhead: int = 8):
        super().__init__()
        self.encoder = _ConvEncoder(in_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, hidden_dim * 4,
                                                   dropout=0.1, batch_first=True,
                                                   activation='gelu', norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nhead, hidden_dim * 4,
                                                   dropout=0.1, batch_first=True,
                                                   activation='gelu', norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # final label is no-object
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, 4))
        self.num_classes = num_classes

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        batch, channels, height, width = features.shape
        source = features.flatten(2).transpose(1, 2)
        position = _sine_position_encoding(height, width, channels, image.device, image.dtype)
        memory = self.transformer_encoder(source + position.unsqueeze(0))
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.transformer_decoder(queries, memory + position.unsqueeze(0))
        return {'logits': self.class_head(decoded), 'boxes': self.box_head(decoded).sigmoid()}


@register_detector('TransformerDetector')
class TransformerDetector(BaseDetector):
    """A configurable DETR-style detector for the existing camera data flow."""

    input_channels = 3

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build_model(self, cfg: dict, device: torch.device) -> nn.Module:
        model_cfg = cfg['model']
        model = QueryTransformerDetector(
            num_classes=model_cfg.get('num_classes', 4),
            in_channels=self.input_channels,
            hidden_dim=model_cfg.get('hidden_dim', 128),
            num_queries=model_cfg.get('num_queries', 100),
            num_encoder_layers=model_cfg.get('num_encoder_layers', 3),
            num_decoder_layers=model_cfg.get('num_decoder_layers', 3),
            nhead=model_cfg.get('nhead', 8),
        )
        weights = cfg.get('resume_weights')
        if weights:
            checkpoint = torch.load(weights, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model'), nn.Module):
                model = checkpoint['model']
                print(f'Resumed from segment {checkpoint.get("seg", "?")}')
            else:
                model.load_state_dict(checkpoint)
        return model.to(device)

    def loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        outputs = model(batch['img'])
        logits, boxes = outputs['logits'], outputs['boxes']
        batch_size, query_count, _ = logits.shape
        no_object = model.num_classes
        class_weight = logits.new_ones(no_object + 1)
        class_weight[no_object] = float(self.cfg['model'].get('no_object_weight', 0.1))
        class_loss = logits.new_zeros(())
        bbox_loss = logits.new_zeros(())
        giou_loss = logits.new_zeros(())
        target_count = 0

        for batch_index in range(batch_size):
            target = batch['targets'][batch_index]
            labels = target['labels'].to(logits.device, dtype=torch.long)
            target_boxes = target['boxes'].to(logits.device)
            target_classes = torch.full((query_count,), no_object, device=logits.device, dtype=torch.long)
            if len(labels):
                probability = logits[batch_index].softmax(-1)
                class_cost = -probability[:, labels]
                l1_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
                giou_cost = -_generalized_iou(_cxcywh_to_xyxy(boxes[batch_index]),
                                               _cxcywh_to_xyxy(target_boxes))
                query_idx, target_idx = _greedy_assignment(class_cost + 5.0 * l1_cost + 2.0 * giou_cost)
                target_classes[query_idx] = labels[target_idx]
                matched_boxes = boxes[batch_index, query_idx]
                assigned_boxes = target_boxes[target_idx]
                bbox_loss = bbox_loss + F.l1_loss(matched_boxes, assigned_boxes, reduction='sum')
                giou = _generalized_iou(_cxcywh_to_xyxy(matched_boxes), _cxcywh_to_xyxy(assigned_boxes))
                giou_loss = giou_loss + (1 - giou.diag()).sum()
                target_count += len(target_idx)
            class_loss = class_loss + F.cross_entropy(logits[batch_index], target_classes, weight=class_weight)

        normalizer = max(target_count, 1)
        return (class_loss / batch_size + 5.0 * bbox_loss / normalizer + 2.0 * giou_loss / normalizer)

    def _image_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_size = int(self.cfg['data'].get('img_size', 640))
        image = cv2.resize(image_bgr, (image_size, image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

    def predict(self, model: nn.Module, img_bgr: np.ndarray, cfg: dict, **_: object) -> list[dict]:
        """Run one image and return pixel-coordinate boxes after class-wise NMS."""
        device = next(model.parameters()).device
        image = self._image_tensor(img_bgr).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
        probabilities = output['logits'][0].softmax(-1)[:, :-1]
        scores, labels = probabilities.max(dim=-1)
        threshold = float(cfg.get('eval', {}).get('conf', 0.25))
        valid = scores >= threshold
        boxes = _cxcywh_to_xyxy(output['boxes'][0][valid]).clamp(0, 1)
        scores, labels = scores[valid], labels[valid]
        if not len(boxes):
            return []
        height, width = img_bgr.shape[:2]
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        iou_threshold = float(cfg.get('eval', {}).get('iou', 0.45))
        kept: list[int] = []
        for label in labels.unique():
            indices = torch.where(labels == label)[0]
            kept.extend(indices[_nms(boxes[indices], scores[indices], iou_threshold)].tolist())
        return [{'box': boxes[index].detach().cpu().tolist(), 'label': int(labels[index]),
                 'score': float(scores[index])} for index in kept]

    def collate_fn(self, batch: list) -> dict:
        images, labels = zip(*batch)
        return {
            'img': torch.stack(images),
            'targets': [{'labels': item[:, 0].long(), 'boxes': item[:, 1:]} for item in labels],
        }
