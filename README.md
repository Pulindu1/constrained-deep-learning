# Deep Learning Project
## Constrained Deep Learning: Efficient Discriminative and Generative Modelling

This project explores efficient implementations of **discriminative (classification)** and **generative (image synthesis)** models under strict parameter and training constraints.  
It focuses on performance–efficiency trade-offs and model stability across two tasks:

1. **Classification** using a ResNet-based architecture on the **AddNIST** dataset  
2. **Image Generation** using a DCGAN-based architecture on the **CIFAR-100** dataset

---

## Repository Structure

```
.
├── classifier.ipynb          # AddNIST classifier implementation (ResNet)
├── generative-model.ipynb    # CIFAR-100 generative model (DCGAN + SN)
├── paper.pdf                 # Full technical report and analysis
└── README.md                        # This file
```

---

## Part 1: Classification Model (Discriminative)

### **Goal**
Develop a deep classifier to categorise AddNIST images (20 classes) using fewer than **100,000 parameters** and **10,000 training steps**.

### **Architecture**
- **Backbone:** Custom **ResNet** with 6 residual blocks (13 layers total)
- **Activation:** ReLU  
- **Normalization:** Batch Normalization for stable convergence  
- **Regularization:** Label smoothing to prevent overconfidence  
- **Optimizer:** Adam  
- **Loss:** Cross-Entropy  

### **Key Implementation Details**
- Custom residual connections following He et al. (2015)  
- Parameter-efficient design leveraging *skip connections* for gradient flow  
- Limited training budget carefully optimized for maximum learning  

### **Results**
| Metric | Value |
|:--|:--|
| Parameters | 95,652 |
| Training Accuracy | 98.8% ± 1.0% |
| Testing Accuracy | 87.7% ± 3.6% |
| Steps | 10,000 |

The model effectively utilized the full parameter budget, achieving high training accuracy with stable convergence.  
**Limitation:** Overfitting evident from the training–test gap, suggesting the need for data augmentation or regularization.

---

## Part 2: Generative Model (Autoencoder + GAN)

### **Goal**
Generate realistic and diverse images from the **CIFAR-100** dataset using fewer than **1,000,000 parameters** and **50,000 training steps**.

### **Architecture**
- **Base:** **DCGAN** (Deep Convolutional GAN)
- **Enhancements:**
  - **Spectral Normalization** in the Discriminator for stability
  - **LeakyReLU** activations to mitigate dead neurons
  - **Batch Normalization** and **ReLU** in the Generator
- **Loss Function:** Binary Cross-Entropy  
- **Evaluation Metrics:** FID, LPIPS  

### **Key Implementation Details**
- Compact yet expressive **generator–discriminator pair**
- Progressive upsampling from 1×1 to 32×32 resolution
- Balanced parameter allocation between generator (≈553k) and discriminator (≈287k)

### **Results**
| Metric | Value |
|:--|:--|
| Parameters | 840,588 total |
| Training Steps | 50,000 |
| FID Score | **59.95** |
| LPIPS (mean ± std) | 0.5578 ± 0.0722 |

The model successfully generated CIFAR-like images, with smooth latent space interpolations.  
**Limitation:** Blurriness and limited variation indicate overfitting and generator mode collapse.

---

## Key Achievements

- Designed and trained two neural models under **strict computational and parameter budgets**.  
- Achieved **high discriminative accuracy (87.7% test)** using <100k parameters.  
- Implemented a **Spectrally Normalized DCGAN** with <1M parameters and a competitive **FID of 59.95**.  
- Incorporated advanced techniques: BatchNorm, Label Smoothing, LeakyReLU, and Spectral Normalization.  
- Delivered a comprehensive analysis of model performance, stability, and generalization.

---


## References
This project draws upon research from:
- He et al., *Deep Residual Learning for Image Recognition* (2015)  
- Radford et al., *Unsupervised Representation Learning with DCGANs* (2015)  
- Miyato et al., *Spectral Normalization for GANs* (2018)  
- Krizhevsky, *CIFAR-100 Dataset* (2009)  
- Geada et al., *AddNIST Dataset* (2024)  

Full citations are available in [`paper.pdf`](./paper.pdf).

