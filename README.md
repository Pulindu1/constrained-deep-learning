# Constrained Deep Learning: Efficient Discriminative and Generative Modelling

Two deep learning models built under **hard parameter and training-step budgets**, in PyTorch.
The question behind both: how much capability can you extract from a model that is deliberately
too small?

| | Task | Constraint | Result |
|:--|:--|:--|:--|
| **01** | AddNIST classification (20 classes) | < 100k params, ≤ 10k steps | **87.7% ± 3.6%** test accuracy @ 95,652 params |
| **02** | CIFAR-100 image generation | < 1M params, ≤ 50k steps | **FID 59.95** @ 840,588 params |

<p align="center">
  <img src="assets/gan-step-0.png" width="45%" alt="DCGAN samples at step 0 — noise" />
  <img src="assets/gan-step-50000.png" width="45%" alt="DCGAN samples at step 50,000" />
  <br />
  <em>Generator output at step 0 (left) and after the full 50,000-step budget (right).</em>
</p>

<p align="center">
  <a href="#part-1--classification-addnist"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

---

## Contents

- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Part 1 — Classification (AddNIST)](#part-1--classification-addnist)
  - [The task](#the-task)
  - [Architecture](#architecture)
  - [Training setup](#training-setup)
  - [Results](#results)
  - [Limitations](#limitations)
- [Part 2 — Generation (CIFAR-100)](#part-2--generation-cifar-100)
  - [The task](#the-task-1)
  - [Why a GAN, and why this GAN](#why-a-gan-and-why-this-gan)
  - [Architecture](#architecture-1)
  - [Training setup](#training-setup-1)
  - [Evaluation](#evaluation)
  - [Results](#results-1)
  - [Limitations](#limitations-1)
- [What I'd do differently](#what-id-do-differently)
- [References](#references)

---

## Repository structure

```
.
├── notebooks/
│   ├── 01_classifier_addnist.ipynb        # ResNet classifier under a 100k-parameter budget
│   └── 02_generative_dcgan_cifar100.ipynb # Spectrally normalised DCGAN under a 1M-parameter budget
├── assets/                                # Figures used in this README (exported from notebook outputs)
├── requirements.txt
└── README.md
```

Both notebooks are committed **with their outputs intact**, so you can read the training curves,
sample grids and metrics on GitHub without running anything.

---

## Setup

**Requirements:** Python 3.9+ and, for notebook 02, a CUDA-capable GPU (50,000 GAN steps is
impractical on CPU). Notebook 01 runs on CPU in a reasonable time.

```bash
git clone https://github.com/Pulindu1/constrained-deep-learning.git
cd constrained-deep-learning

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch that matches your driver instead of the
default wheel — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then launch Jupyter and run either notebook top to bottom:

```bash
jupyter lab
```

### Data

Nothing needs to be downloaded by hand — both notebooks fetch their own data on first run:

| Notebook | Dataset | Size | Downloaded to |
|:--|:--|:--|:--|
| 01 | [AddNIST](https://data.ncl.ac.uk/articles/dataset/AddNIST/24574354) | ~629 MB | `classification-data/` |
| 02 | CIFAR-100 (via `torchvision`) | ~170 MB | `data/` |

These directories, along with the 10k real/generated image folders used for FID, are
gitignored.

**Runtime:** notebook 01 takes roughly 10 minutes on a modern GPU; notebook 02 takes a few
hours for the full 50,000 steps, plus a few minutes for FID evaluation.

---

## Part 1 — Classification (AddNIST)

**Goal:** classify AddNIST images into 20 classes with **fewer than 100,000 parameters** and
**at most 10,000 training steps**.

<p align="center">
  <img src="assets/addnist-samples.png" width="52%" alt="Sample AddNIST images with their class labels" />
  <br />
  <em>AddNIST samples: three MNIST digits stacked as RGB channels, labelled by an arithmetic combination of the three — so the task needs all channels read together, not one digit recognised.</em>
</p>

### The task

Each AddNIST image is three MNIST digits stacked as the red, green and blue channels of a single
28×28×3 image. The label is an arithmetic function of the three digits:

```
l = (r + g + b) − 1
```

which gives 20 classes. Two consequences drive the whole design:

1. **All three channels must be read jointly.** Recognising one digit tells you nothing about the
   label, so the network cannot fall back on single-channel shape features.
2. **The signal lives in fine detail.** At 28×28, aggressive spatial reduction destroys the digit
   strokes the label depends on, so the usual "downsample early, widen channels" recipe is
   actively harmful here.

The dataset ships with pre-split train/valid/test `.npy` arrays and is downloaded by the notebook
on first run.

### Architecture

A custom ResNet, chosen because residual architectures are **parameter-efficient per unit of
depth** — depth is the cheap resource under a parameter cap, and skip connections are what make
depth trainable. He et al.'s insight is that adding an identity shortcut around each pair of
convolutions lets gradients flow past the weighted layers, mitigating the vanishing-gradient
problem Bengio et al. identified: without shortcuts, gradients through a deep stack shrink toward
zero and performance plateaus after a few thousand steps. Wu et al.'s comparison of wider vs.
deeper ResNets supports the same trade-off — depth buys accuracy more cheaply than width when
parameters are the binding constraint.

**Residual block** (the unit used six times):

| Component | Detail |
|:--|:--|
| conv1 | 3×3, stride `s`, padding 1, no bias |
| bn1 + ReLU | BatchNorm2d then ReLU |
| conv2 | 3×3, stride 1, padding 1, no bias |
| bn2 | BatchNorm2d |
| shortcut | identity, or 1×1 conv + BatchNorm when stride ≠ 1 or channels change |
| output | `ReLU(F(x) + x)` |

**Full network:**

```
Input 3×28×28
  3×3 conv, 16 channels, stride 1 → BatchNorm → ReLU        28×28
  Block 1:  16 → 16, stride 1                               28×28
  Block 2:  16 → 32, stride 2                               14×14
  Block 3:  32 → 32, stride 1                               14×14
  Block 4:  32 → 32, stride 2                                7×7
  Block 5:  32 → 32, stride 1                                7×7
  Block 6:  32 → 32, stride 1                                7×7
  AdaptiveAvgPool2d(1×1) → flatten                          32
  Linear 32 → 20                                            20 logits
```

Depth follows directly from the block count:

```
depth = (residual blocks × 2) + 1 = (6 × 2) + 1 = 13 weighted layers
```

**Design choices worth naming:**

- **Narrow channels, minimal downsampling.** Channels only ever reach 32, and just two of the six
  blocks use stride 2, so the feature map shrinks twice (28 → 14 → 7) before global average
  pooling. Spatial reduction is what would cost accuracy on this dataset, and compute was never
  the limiting factor — the parameter budget and the small data pool were.
- **Global average pooling instead of a flattened dense head.** A flatten-and-FC head over a 7×7×32
  map would spend tens of thousands of parameters on a single layer; GAP reduces it to a 32→20
  linear layer (660 parameters) and leaves the budget for depth.
- **BatchNorm everywhere.** Beyond regularisation, normalising activations stabilises training and
  lets a higher learning rate be used, which matters when the entire run is 10,000 steps.

The result is **95,652 parameters — 96% of the 100,000 budget**, spent almost entirely on
convolutional depth.

### Training setup

| Setting | Value |
|:--|:--|
| Optimiser | Adam (initial LR 0.001) |
| LR schedule | OneCycleLR — max LR 0.004, `pct_start=0.4` (40% warm-up), cosine anneal, `total_steps=10000` |
| Loss | Cross-entropy with **label smoothing 0.25** |
| Batch size | 128 (`drop_last=True`) |
| Steps | Exactly 10,000, evaluated on the full test set every 1,000 |
| Normalisation | Per-channel mean/std computed from the **training split only**, applied to all splits |
| Augmentation | None in the final run (see [Limitations](#limitations)) |

**Why label smoothing at 0.25.** Müller et al. show that label smoothing prevents a network from
becoming overconfident, tightening class clusters and improving generalisation. That is exactly the
failure mode here: the data pool is small enough for a 95k-parameter network to approach
memorisation, and smoothing keeps the softmax from collapsing onto one-hot targets. 0.25 is a
strong setting, chosen deliberately because overfitting — not underfitting — was the observed
problem.

**Why OneCycle.** With a hard 10,000-step cap there is no room for a long constant-LR phase
followed by manual decay. OneCycle warms up over the first 40% of the run to escape poor early
minima at a high learning rate, then cosine-anneals to a small LR so the final steps are spent
refining rather than bouncing.

### Results

| Metric | Value |
|:--|:--|
| Parameters | 95,652 / 100,000 |
| Training steps | 10,000 / 10,000 |
| Final train loss | 1.329 (label-smoothed, so it does not floor near 0) |
| Train accuracy | 98.8% ± 1.0% |
| **Test accuracy** | **87.7% ± 3.6%** |

<p align="center">
  <img src="assets/classifier-training-curve.png" width="46%" alt="Training and test accuracy over 10,000 steps" />
  <img src="assets/classifier-predictions.png" width="46%" alt="Per-class prediction confidences on test images" />
  <br />
  <em>Left: accuracy over the training budget, with shaded ±1 s.d. bands across batches. Right: per-class confidences on test images (blue = correct, red = incorrect).</em>
</p>

### Limitations

The high training accuracy (98.8%) with a tight uncertainty band shows the architecture learns the
task efficiently and stably despite the parameter cap and the small dataset. The ~11-point gap to
87.7% test accuracy is straightforward **overfitting**.

The training curve localises when it happens: both curves rise together and the model has largely
fit the training data by around **step 5,000**. Past that point train accuracy keeps climbing while
test accuracy flattens — the remaining 5,000 steps buy memorisation, not generalisation.

The fix is data, not capacity, since the parameter budget is already 96% spent. Mild augmentation
is the obvious lever, with one dataset-specific caveat: **random horizontal flips would be wrong
here**, because a flipped digit is not the same digit and the label is a function of digit
identity. Slight distortions and small rotations preserve the label and would enrich the limited
pool. An early augmentation attempt hurt accuracy and was cut for time; with a tuned strength and
schedule it should close most of the gap for free.

---

## Part 2 — Generation (CIFAR-100)

**Goal:** train a deep generative model on CIFAR-100 that synthesises new 32×32 images — measured
on realism, diversity, and how distinct they are from the originals — with **fewer than 1,000,000
parameters** and **at most 50,000 training steps**.

### The task

CIFAR-100 contains 60,000 32×32 colour images across 100 fine classes (600 images each), which are
also grouped into 20 coarse superclasses. Preprocessing is deliberately minimal: resize to 32×32,
convert to a tensor, and normalise each channel with `mean = std = 0.5`, mapping pixels to
**[−1, 1]** to match the generator's `tanh` output range. The model is **unconditional** — class
labels are never fed to either network; they only serve as a reference point when judging whether
samples resemble recognisable categories.

### Why a GAN, and why this GAN

The natural first candidate for "generate images" is an **autoencoder**, trained by minimising
squared L2 reconstruction error:

```
L_AE = E_x~p_data [ ‖ x − D(E(x)) ‖² ]
```

That objective is the problem. Minimising pixel-wise L2 against an uncertain target makes the
optimal prediction the *mean* of the plausible outputs, which is systematically blurry — and blur
is exactly what FID punishes. A GAN replaces the hand-written pixel loss with a learned one: a
discriminator that has to be *fooled*, which rewards sharp, plausible detail rather than average
detail. The GAN minimax objective is:

```
min_G max_D V(D, G) = E_x~p_data [ log D(x) ] + E_z~p_z [ log(1 − D(G(z))) ]
```

where G is the generator, D the discriminator, x real data and z a latent vector drawn from a prior
distribution.

Within GANs, the two choices that matter here:

- **DCGAN** (Radford et al.) for the backbone — an all-convolutional generator that raises spatial
  resolution *gradually*, 1×1 → 4×4 → 8×8 → 16×16 → 32×32, rather than jumping to full resolution
  in one step. Each transposed convolution then only has to add detail at one scale, which is a
  much easier function to learn with few parameters.
- **Spectral normalisation** (Miyato et al.) on the discriminator, making this an SN-GAN. Miyato
  et al. found SN-GAN architectures produced higher-quality CIFAR-10 images than the alternatives
  they compared against, and the mechanism explains why it matters *especially* under a parameter
  cap: discrimination is a far easier problem than generation, so a small discriminator converges
  much faster than the generator it supervises. Once it wins, its gradients saturate and the
  generator stops learning. Spectral normalisation divides each layer's weights by their largest
  singular value, bounding the discriminator's Lipschitz constant and keeping it a useful teacher
  instead of a perfect critic. This is the single most important stability lever in the model.

<p align="center">
  <img src="assets/gan-step-0.png" width="45%" alt="DCGAN samples at step 0" />
  <img src="assets/gan-step-50000.png" width="45%" alt="DCGAN samples at step 50,000" />
  <br />
  <em>Non-cherry-picked samples: step 0 (left) and step 50,000 (right).</em>
</p>

### Architecture

**Generator — 553,644 parameters.** Latent vector z ∈ ℝ¹⁰⁰, reshaped to 100×1×1, then four
transposed convolutions:

| Layer | Op | Output |
|:--|:--|:--|
| 1 | ConvTranspose2d(100 → 168, k=4, s=1, p=0) + BatchNorm + ReLU | 168×4×4 |
| 2 | ConvTranspose2d(168 → 84, k=4, s=2, p=1) + BatchNorm + ReLU | 84×8×8 |
| 3 | ConvTranspose2d(84 → 42, k=4, s=2, p=1) + BatchNorm + ReLU | 42×16×16 |
| 4 | ConvTranspose2d(42 → 3, k=4, s=2, p=1) + **tanh** | 3×32×32 |

BatchNorm stabilises training and ReLU supplies the non-linearity; `tanh` puts outputs in [−1, 1],
matching the normalised real data.

**Discriminator — 286,944 parameters.** A mirror of the generator, every convolution wrapped in
`nn.utils.spectral_norm`:

| Layer | Op | Output |
|:--|:--|:--|
| 1 | SN-Conv2d(3 → 42, k=4, s=2, p=1) + LeakyReLU(0.2) | 42×16×16 |
| 2 | SN-Conv2d(42 → 84, k=4, s=2, p=1) + LeakyReLU(0.2) | 84×8×8 |
| 3 | SN-Conv2d(84 → 168, k=4, s=2, p=1) + LeakyReLU(0.2) | 168×4×4 |
| 4 | SN-Conv2d(168 → 1, k=4, s=1, p=0) + Sigmoid | 1×1×1 → scalar |

**LeakyReLU(0.2) rather than ReLU** in the discriminator: Xu et al. found that a non-zero slope on
the negative part of a rectified unit improves results by preventing dead neurons. A dead unit in
the discriminator is worse than a dead unit elsewhere — it silently stops carrying gradient back to
the generator, which is the only training signal the generator has.

The 553k/287k split between generator and discriminator was arrived at by hand. Roughly a 2:1
allocation in the generator's favour reflects that generation is the harder half of the problem
while the discriminator only needs to stay a competent-but-beatable critic.

### Training setup

| Setting | Value |
|:--|:--|
| Loss | Binary cross-entropy (`nn.BCELoss`) on both networks |
| Optimiser | Adam, LR 0.0002, betas (0.5, 0.999) — separate optimisers for G and D |
| Batch size | 256 (`drop_last=True`) |
| Latent dim | 100, sampled from a standard normal |
| Steps | 50,000, sample grid logged every 5,000 |

Each step: sample z, generate a fake batch, update D on real (label 1) and detached fake (label 0)
batches, then update G with flipped labels — the standard non-saturating trick, where the generator
maximises `log D(G(z))` rather than minimising `log(1 − D(G(z)))` to keep gradients strong early in
training when its samples are easily rejected.

### Evaluation

Two metrics, measuring different things:

- **FID** (Fréchet Inception Distance) — *realism*. Computed with `clean-fid` in `mode="clean"`
  between 10,000 generated images and 10,000 real CIFAR-100 **test** images, both written to disk
  as PNGs. Comparing against the held-out test split rather than the training data means a model
  that simply memorised its training set gains nothing. Lower is better.
- **LPIPS** (Learned Perceptual Image Patch Similarity) — *diversity*. LPIPS measures perceptual
  similarity rather than quality, so applying it across generated samples answers "how different
  are these images from one another and from real images?" Reported as a mean and standard
  deviation over the generated set.

The pairing matters: FID alone can be gamed by a model that produces a handful of very good images
over and over. LPIPS is what exposes that failure, and it did.

Latent structure is checked separately with **spherical (slerp) interpolation** between two random
latent vectors. Slerp rather than linear interpolation because the prior is a high-dimensional
Gaussian, whose mass sits on a shell — linearly interpolating cuts through the sparse interior and
produces off-distribution latents. Smooth transitions along a slerp path indicate the generator has
learned a continuous mapping rather than memorising isolated points.

### Results

| Metric | Value |
|:--|:--|
| Parameters | 840,588 / 1,000,000 (553,644 G + 286,944 D) |
| Training steps | 50,000 / 50,000 |
| **FID** ↓ | **59.95** |
| LPIPS (mean ± std) ↑ | 0.5578 ± 0.0722 |

<p align="center">
  <img src="assets/gan-slerp-interpolation.png" width="90%" alt="Spherical latent-space interpolation strip" />
  <br />
  <em>Spherical (slerp) interpolation between two latent vectors — smooth transitions indicate a structured latent space rather than memorised samples.</em>
</p>

**Reading the two numbers together.** The FID of 59.95 says samples land in roughly the right
region of image space: global colour statistics and object-like structure are correct, and outputs
are at a comparable apparent resolution to the originals. The LPIPS **mean** of 0.5578 says
generated images are on average fairly different from real ones — the model is inventing images
rather than reproducing training examples, which is the distinctness half of the goal.

The LPIPS **standard deviation** of 0.0722 is the problem. A tight spread means the generated
images are all roughly equally distant from the reference in perceptual space — they lack
individual character, which is the fingerprint of **partial mode collapse**. It corroborates what
the sample grids show directly: a limited set of recurring textures and colour schemes rather than
100 distinguishable classes.

### Limitations

**The parameter budget is the dominant constraint.** At 840k parameters only compact architectures
are viable, and the real difficulty is not the total but the *split* between discriminator and
generator. Give either side too much relative to the other and performance degrades even as total
parameters rise — a stronger discriminator saturates the generator's gradients, a stronger
generator outpaces a critic too weak to correct it. Because each rebalancing costs a full training
run, and time and compute were finite, the split was not pushed further once a stable configuration
was found; parameters were left on the table rather than spent chasing an unstable balance.

**Sample quality is uneven.** The images are often blurry and frequently do not resemble any
identifiable CIFAR-100 category. Cherry-picking the best outputs turns up recognisable objects —
two flowers, a fish, a crab — but most of what the generator produces is not reliable enough to
name.

**Training converges early and then plateaus.** The logged losses settle into a stable equilibrium
(Loss_D ≈ 1.35, Loss_G ≈ 0.70) by roughly step 5,000 and barely move over the remaining 45,000, so
the bulk of the step budget buys refinement rather than new structure — consistent with a capacity
ceiling rather than an optimisation failure.

**Diversity, not fidelity, is the binding weakness.** The low LPIPS variance points at mode collapse
specifically, which is a loss-function and regularisation problem, not something more training
steps fix. Future work should target it directly (see below) rather than through stabilisation
alone.

---

## What I'd do differently

- **Classifier — augmentation.** Random crops, small rotations and mild distortions (never
  horizontal flips, which break the digit-identity label), plus mixup. An early augmentation
  attempt hurt accuracy and was cut for time; with a proper schedule and tuned strength it should
  close most of the train/test gap without costing a single parameter.
- **GAN — a loss that attacks mode collapse.** Swap BCE for a hinge or Wasserstein-GP objective and
  add a small amount of differentiable discriminator augmentation. Both target the low-diversity
  failure directly, rather than treating it as a side effect of instability.
- **GAN — search the G/D parameter split properly.** The 553k/287k allocation was hand-tuned over a
  handful of runs. A short sweep at reduced step counts would map the trade-off far more cheaply
  than full-length runs and would probably let more of the unused 160k parameters be spent safely.
- **Both — validate and seed.** Track a proper validation split for early stopping instead of
  reporting whatever the model happens to be at the step cap, and fix random seeds so results are
  exactly reproducible.

---

## References

**Datasets**

- Geada et al., *AddNIST Dataset* (2024) — <https://data.ncl.ac.uk/articles/dataset/AddNIST_Dataset/24574354>
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009) — CIFAR-100
- LeCun, Cortes and Burges, *The MNIST Database of Handwritten Digits*, IEEE TPAMI (1998) — <http://yann.lecun.com/exdb/mnist/>

**Architectures and training**

- He, Zhang, Ren and Sun, *Deep Residual Learning for Image Recognition* (2015) — <https://arxiv.org/abs/1512.03385>
- Wu, Shen and van den Hengel, *Wider or Deeper: Revisiting the ResNet Model for Visual Recognition* (2016) — <https://arxiv.org/abs/1611.10080>
- Bengio, Simard and Frasconi, *On the Difficulty of Training Recurrent Neural Networks* (1994) — vanishing gradients
- Ioffe and Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, arXiv (2015)
- Müller, Kornblith and Hinton, *When Does Label Smoothing Help?*, arXiv (2019) — <https://arxiv.org/abs/1906.02629>
- Xu, Wang, Chen and Li, *Empirical Evaluation of Rectified Activations in Convolutional Network* (2015) — <https://arxiv.org/abs/1505.00853>

**Generative modelling**

- Radford, Metz and Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs* (2015) — <https://arxiv.org/abs/1511.06434>
- Miyato et al., *Spectral Normalization for Generative Adversarial Networks* (2018) — <https://arxiv.org/abs/1802.05957>
- Kramer, *Nonlinear Principal Component Analysis Using Autoassociative Neural Networks*, AIChE Journal 37(2) (1991), pp. 233–243 — autoencoders
- Oanda, *A Review of the Image Quality Metrics Used in Image Generative Models* — LPIPS overview, <https://blog.paperspace.com/review-metrics-image-synthesis-models/#learned-perceptual-image-patch-similarity-lpips>

**Implementation guides**

- Ahmed, *Writing ResNet from Scratch in PyTorch* — <https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch>
- Kumar, *Spectrally Normalized Generative Adversarial Networks (SN-GAN)* (2023) — <https://medium.com/%40danushidk507/spectrally-normalized-generative-adversarial-networks-sn-gan-d40b27b5fc8a>
- Inkawhich, *DCGAN Tutorial* (PyTorch, 2018) — <https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>
- *Label Smoothing in PyTorch* (Stack Overflow) — <https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch>

Code adapted from external sources is attributed inline in the notebooks:
the ResNet scaffolding follows a
[DigitalOcean tutorial](https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch),
the SN-GAN structure follows
[this walkthrough](https://medium.com/%40danushidk507/spectrally-normalized-generative-adversarial-networks-sn-gan-d40b27b5fc8a),
and the slerp helper was drafted with GPT-4o and verified by hand.

---

## Licence

Code is released under the [MIT Licence](LICENSE).
