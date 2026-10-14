# Structural Guidance Enhances Attention Networks for Multi-Dimensional Facial Skin Quality Assessment

<div align="center">
 <img width="413" height="253" alt="image" src="https://github.com/user-attachments/assets/c962b36e-3a88-42ff-b5ff-efe9d4290caf" />
</div>

This repository contains the official implementation of our work **“Structural guidance enhances attention networks for multi-dimensional facial skin quality assessment”** (submitted to *Scientific Reports*).
The proposed framework integrates structural cues—pore and wrinkle features—with an attention-enhanced U-Net backbone (SGA-UNet) to estimate four continuous facial skin-quality attributes: **moisture, glossiness, sebum, and elasticity**.

## 🔔 Code and Data Availability

### The dataset download    [Link](https://forms.gle/Df8r7Dxf9yanb95j7) 

* FaceSkin-Q : Decompression password: skinqskinq
* NIA :  Decompression password: niania

> **Note:** Some annotations in the FaceSkin-Q dataset contain missing values or formatting errors. Please refer to the data loading logic in `./data_load/load_skin_q.py` to inspect and handle these cases.
  
## 📚 Citation

If you find this work useful, please cite:
```bibtex
@article{chu2026structural,
  title   = {Structural guidance enhances attention networks for multi-dimensional facial skin quality assessment},
  author  = {Chu, X. and Kim, C. and Bai, S. J. and Song, Y. and Sung, J. and Choi, Y.},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-65848-z},
  url     = {https://doi.org/10.1038/s41598-026-65848-z}
}
```


