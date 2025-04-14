# MM-PHATE: Multiway Multislice PHATE

> Visualizing Hidden Dynamics of RNNs Through Training

---

## Overview

MM-PHATE (Multiway Multislice PHATE) is a novel dimensionality reduction framework designed to visualize and understand the internal dynamics of Recurrent Neural Networks (RNNs) throughout their training process.

Unlike standard visualization methods which only focus on the final network state, MM-PHATE captures how RNN hidden representations evolve over both time steps and training epochs, revealing rich dynamics, community structure, and phases of information processing and compression.

This repository contains code for implementing MM-PHATE on RNN hidden state data, along with examples and visualization utilities. For more information, please refer to [the paper](https://arxiv.org/abs/2406.01969)


---

## Method at a Glance

MM-PHATE extends the M-PHATE framework originally developed by [Gigante et al.](https://github.com/scottgigante/m-phate?tab=readme-ov-file) to handle the temporal nature of RNNs.

Key components include:
- Multiway multislice kernel construction across epochs, time steps, and units
- Structured graph affinities within and across time and epochs
- PHATE embedding for visualization
- Entropy-based analysis of information flow within RNNs

---

## Visualization Examples

![MM-PHATE Visualization](figures/embedding.png)

---

## Usage

1. Prepare your RNN hidden state tensor with dimensions:
```
[EPOCHS, TIMESTEPS, UNITS, SAMPLES]
```

2. Construct the multiway multislice kernel using provided scripts.

3. Apply MM-PHATE embedding.

4. Visualize and analyze your results.

Example usage:

```python
from mmphate import compute_kernel, apply_phate, plot_embedding

K = compute_kernel(hidden_tensor)
embedding = apply_phate(K)
plot_embedding(embedding)
```

---

## Acknowledgement & Citation

MM-PHATE builds upon the M-PHATE framework by Gigante et al. (2019), which pioneered multislice visualization of feedforward neural networks. 

If you use MM-PHATE in your research, please cite:

> S. Gigante, A. Charles, S. Krishnaswamy, G. Mishne. *Visualizing the PHATE of Neural Networks.* arXiv preprint arXiv:1908.02831, 2019.

M-PHATE repository: https://github.com/KrishnaswamyLab/MPHATE

---

## Paper

For details of the MM-PHATE method, see:

> *Multiway Multislice PHATE: Visualizing Hidden Dynamics of RNNs through Training.* [link to your paper or arXiv]

---

## License

MIT License.

---

## Contact

For questions or collaborations, please contact:

*Your Name*  
*Your Institution*  
*Email: your_email@domain*

---

## TODO
- [ ] Add example notebooks
- [ ] Add support for GRU/Vanilla RNN
- [ ] Optimize memory efficiency for large networks
- [ ] Expand to Transformer models
