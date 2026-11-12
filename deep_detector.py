"""Optional PyTorch detector adapter.

This file deliberately keeps deep learning separate from the OpenCV baseline so both can be
benchmarked with the same evaluation.py functions. Training requires an annotated dataset.
"""


def build_fasterrcnn(num_classes=4, pretrained_backbone=True):
    """Build Faster R-CNN. num_classes includes background."""
    try:
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:
        raise ImportError("Install torch and torchvision to use the deep-learning detector") from exc

    if pretrained_backbone:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TorchvisionDetector:
    def __init__(self, model, class_names=None, device=None, score_threshold=0.5):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.class_names = class_names or {1: "player", 2: "ball", 3: "referee"}
        self.score_threshold = score_threshold

    def __call__(self, bgr_image):
        import cv2
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with self.torch.no_grad():
            output = self.model([tensor.to(self.device)])[0]
        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            score = float(score.detach().cpu())
            if score < self.score_threshold:
                continue
            x1, y1, x2, y2 = map(float, box.detach().cpu().tolist())
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            detections.append({
                "bbox": (x, y, w, h),
                "center": (x + w/2.0, y + h),
                "class_name": self.class_names.get(int(label), str(int(label))),
                "score": score,
            })
        return detections


def save_checkpoint(model, path, metadata=None):
    import torch
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata or {}}, path)


def load_checkpoint(model, path, map_location=None):
    import torch
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
