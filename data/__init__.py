from .synthetic import SyntheticEdgeDataset

__all__ = ["SyntheticEdgeDataset"]

# COCO dataset (requires pycocotools): imported lazily to avoid hard dependency
def _load_coco():
    from .dataset import COCODetectionDataset, build_dataloader
    return COCODetectionDataset, build_dataloader

