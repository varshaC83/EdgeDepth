"""EdgeDepth: RGB-guided edge-preserving depth map upsampling and refinement."""

__version__ = "1.0.0"

METHODS = [
    "nearest",
    "bilinear",
    "bicubic",
    "joint_bilateral",
    "guided_filter",
    "deep_learning",
    "hybrid",
]

METHOD_LABELS = {
    "nearest": "Nearest neighbour",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "joint_bilateral": "Joint bilateral (JBU)",
    "guided_filter": "Guided filter",
    "deep_learning": "Deep learning (EdgeDepthNet)",
    "hybrid": "Hybrid (fusion + edge-aware refinement)",
}
