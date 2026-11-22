"""Small randomly initialised timm ViTs standing in for Hugging Face encoders.

A stand-in is a folder laid out like a Hub repository (`config.json` plus
`model.safetensors`), which timm loads with `local-dir:<folder>` and the
`local` loader hands to the `timm` runtime.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path


def virchow2_like(directory: Path, *, size: int = 56, seed: int = 0) -> Path:
    """A ViT shaped like Virchow2: 4 register tokens, SwiGLU MLP, every token out.

    The config is Virchow2's own, shrunk (width 32, depth 2, `size` px input
    at patch 14); every weight is drawn at random, so blocks change the tokens.

    Args:
        directory: The folder to write.
        size: The input size, a multiple of 14.
        seed: Seed of the weights.

    Returns:
        The folder.
    """
    import timm
    import torch
    from timm.layers.mlp import SwiGLUPacked

    model_args = {
        "img_size": size,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 2,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "mlp_ratio": 5.3375,
        "global_pool": "",
        "dynamic_img_size": True,
        "num_classes": 0,
    }
    config = {
        "architecture": "vit_huge_patch14_224",
        "model_args": model_args,
        "pretrained_cfg": {
            "tag": "virchow_v2",
            "custom_load": False,
            "input_size": [3, size, size],
            "fixed_input_size": False,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "center",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
        },
    }
    model = timm.create_model(
        config["architecture"],
        pretrained=False,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        **model_args,
    )
    return _write(directory, config, model, seed)


def prov_gigapath_like(directory: Path, *, seed: int = 0) -> Path:
    """A ViT shaped like Prov-GigaPath's tile encoder: class token out, 224 px in.

    The config is Prov-GigaPath's own (a DINOv2 ViT-g with SwiGLU at patch
    16), shrunk to width 32 and depth 2; every weight is drawn at random.

    Args:
        directory: The folder to write.
        seed: Seed of the weights.

    Returns:
        The folder.
    """
    import timm

    model_args = {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 2,
        "init_values": 1e-5,
        "mlp_ratio": 5.33334,
        "num_classes": 0,
    }
    config = {
        "architecture": "vit_giant_patch14_dinov2",
        "num_classes": 0,
        "global_pool": "token",
        "model_args": model_args,
        "pretrained_cfg": {
            "tag": "",
            "custom_load": False,
            "input_size": [3, 224, 224],
            "fixed_input_size": True,
            "interpolation": "bicubic",
            "crop_pct": 0.875,
            "crop_mode": "center",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
        },
    }
    model = timm.create_model(config["architecture"], pretrained=False, **model_args)
    return _write(directory, config, model, seed)


def _write(directory: Path, config: dict[str, Any], model: Any, seed: int) -> Path:
    """Write the config and random weights for the model's state dict."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(seed)
    state = {
        name: torch.randn(value.shape, generator=generator) * 0.3
        for name, value in model.state_dict().items()
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file(state, str(directory / "model.safetensors"))
    return directory
