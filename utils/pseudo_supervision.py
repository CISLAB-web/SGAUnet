"""
Pseudo targets for auxiliary **segmentation head** supervision (weak MDM/WDM / noise).

Training scripts apply this **only when** ``model: CustomUnet_att_ca``
(dual segmentation + regression). Other regressors omit a seg head.

Values are in **[0, 1]** for ``nn.BCEWithLogitsLoss``.
"""

from __future__ import annotations

import torch


def segmentation_pseudo_mask(
    seg_output: torch.Tensor,
    gmdm: torch.Tensor | None,
    gwdm: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """
    Parameters
    ----------
    seg_output
        Segmentation logits tensor ``[B, 1, H, W]`` (shape reference for noise modes).
    gmdm, gwdm
        Maps from dataloader after ``.to(device)``. If a modality is disabled in yaml,
        the loader fills **zeros** of the same shape — use ``use_gmdm`` / ``use_gwdm``
        explicitly for single-modality experiments.
    mode
        - ``mdm_wdm`` — ``clamp(gmdm + gwdm, 0, 1)`` (original combined target).
        - ``gmdm`` — MDM-only target (recommended: ``use_gmdm: True``, ``use_gwdm: False``).
        - ``gwdm`` — WDM-only (recommended: ``use_gmdm: False``, ``use_gwdm: True``).
        - ``gaussian`` — i.i.d. ``N(0,1)`` per element, then ``sigmoid`` → roughly Unif-like
          targets; **resampled every call** (each train step gets new noise).
        - ``uniform`` — ``Uniform(0,1)`` i.i.d. (optional unstructured baseline).
    """
    m = (mode or "mdm_wdm").strip().lower()
    if m == "gaussian":
        return torch.sigmoid(torch.randn_like(seg_output))
    if m == "uniform":
        return torch.rand_like(seg_output)

    if gmdm is None or gwdm is None:
        raise ValueError(
            "gmdm/gwdm must be tensors on device unless seg_supervision is "
            "'gaussian' or 'uniform'"
        )

    if m == "mdm_wdm":
        return torch.clamp(gmdm + gwdm, 0, 1)
    if m == "gmdm":
        return gmdm
    if m == "gwdm":
        return gwdm

    raise ValueError(
        "Unknown seg_supervision %r; use 'mdm_wdm', 'gmdm', 'gwdm', "
        "'gaussian', or 'uniform'." % (mode,)
    )
