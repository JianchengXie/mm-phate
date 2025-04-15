# MM-PHATE: Multiway Multislice PHATE

> Visualizing Hidden Dynamics of RNNs Through Training

---

## Overview

MM-PHATE (Multiway Multislice PHATE) is a novel dimensionality reduction framework designed to visualize and understand the internal dynamics of Recurrent Neural Networks (RNNs) throughout their training process.

Unlike standard visualization methods which only focus on the final network state, MM-PHATE captures how RNN hidden representations evolve over both time steps and training epochs, revealing rich dynamics, community structure, and phases of information processing and compression.

This repository contains code for implementing MM-PHATE on RNN hidden state data, along with examples and visualization utilities.

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

1. Prepare your RNN hidden state tensor (trace_data) with dimensions:
```
[EPOCHS*TIMESTEPS, UNITS, SAMPLES]
```

2. Construct the multiway multislice kernel using provided scripts.

3. Apply MM-PHATE embedding.

4. Visualize and analyze your results.

## Example usage

1. Clone the repository
```
git clone https://https://github.com/JianchengXie/mm-phate.git
cd mm-phate
```
2. Ensure the preprocessed data is available

The Area2_Bump folder is included and contains the preprocessed files:

- trainX.npy
- trainy.npy
- testX.npy
- testy.npy

If you need to download and preprocess the original data, visit the [DANDI Archive](https://dandiarchive.org/dandiset/000127)

3. Training the Model
   
To train the LSTM model and perform MM‑PHATE analysis, run:

```
python train_model.py
```

The train_model.py script will:
- Set up reproducible training by configuring GPU settings and fixing random seeds.
- Load the preprocessed data from the Area2_Bump folder.
- Build an LSTM model using TensorFlow/Keras.
- Train the model while recording hidden activations with the MM‑PHATE TraceHistory callback.
- Apply MM‑PHATE to reduce the dimensionality of recorded hidden activations.
- Display example 2D visualizations of the MM‑PHATE output (colored by epoch, timestep, hidden unit, and most active output).

---

## Acknowledgement

MM-PHATE builds upon the M-PHATE framework by Gigante et al. (2019), which pioneered multislice visualization of feedforward neural networks. 

If you use M-PHATE in your research, please cite:

> S. Gigante, A. Charles, S. Krishnaswamy, G. Mishne. *Visualizing the PHATE of Neural Networks.* arXiv preprint [arXiv:1908.02831, 2019](https://arxiv.org/abs/1908.02831).

---

## Citation

If you use MM‑PHATE in your research, please cite:

```bibtex
@misc{xie2024multiwaymultislicephatevisualizing,
      title={Multiway Multislice PHATE: Visualizing Hidden Dynamics of RNNs through Training}, 
      author={Jiancheng Xie and Lou C. Kohler Voinov and Noga Mudrik and Gal Mishne and Adam Charles},
      year={2024},
      eprint={2406.01969},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.01969}
}
```
