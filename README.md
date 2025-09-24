# Dog Breed Classification with EfficientNetV2

This repository contains my end-to-end Kaggle notebook for classifying **120 dog breeds** using **TensorFlow/Keras** and a fine-tuned **EfficientNetV2-B2** model.  
The project focused on systematically overcoming common deep learning pitfalls and squeezing maximum accuracy out of limited compute.

---

## Project Overview
- Dataset: ~20k labeled dog images across 120 classes  
- Backbone: **EfficientNetV2-B2** (ImageNet pretrained)  
- Training strategy: staged fine-tuning + progressive resizing + test-time augmentation (TTA)  
- **Final test accuracy: 90.3%** 

---

## Training Pipeline

### 1. **Phase A – Head Training**
- Backbone frozen, trained only classification head.  
- Very low learning rate to prevent NaNs.  
- ✅ ~88.7% validation accuracy.

### 2. **Phase B – Fine-Tuning**
- Unfroze top 40 layers of EfficientNetV2-B2.  
- LR schedule: 1e-5 → 1e-6.  
- ✅ ~89.0% test accuracy.

### 3. **Progressive Resizing**
- Increased input size: 300×300 → 320×320.  
- Helped model capture finer texture details.  
- ✅ ~89.7% test accuracy.

### 4. **TTA (Test-Time Augmentation)**
- Flips, rotations, zooms applied at inference.  
- Predictions averaged across multiple augmented views.  
- ✅ **Final accuracy: 90.3%**.

---

## Problems & Fixes

| Problem | Cause | Solution |
|---------|-------|----------|
| **NaN losses** | Wrong preprocessing (pixel scaling) | Fixed normalization pipeline |
| **OOM / Collective reduce errors** | Multi-GPU sync failures | Switched to single P100 GPU |
| **High early loss** | LR too high with frozen backbone | Two-phase training (head → fine-tune) |
| **Accuracy stagnation** | Resolution bottleneck | Progressive resizing |
| **Overfitting risk** | Train/val gap growing | Dropout + strong online augmentation |
| **Inference variance** | Single-pass prediction | TTA ensemble |

---

## Results

- **Phase A (300×300)** → val acc: ~88.7%  
- **Phase B (300×300 fine-tune)** → test acc: ~89.0%  
- **Progressive Resizing (320×320)** → test acc: ~89.7%  
- **TTA Ensemble (320×320)** → **test acc: 90.3%**

---

## Training Curves

### Phase A (Head Training, 300×300)
![Phase A Accuracy](<img width="1400" height="800" alt="phase_a_acc" src="https://github.com/user-attachments/assets/9101aafc-38a6-495e-aef6-f95e0cb04b7b" />
)  
![Phase A Loss](<img width="1400" height="800" alt="phase_a_loss" src="https://github.com/user-attachments/assets/b0832729-832d-46b8-9592-8c26a55183ff" />)


---

### Phase B (Fine-Tuning, 300×300)
![Phase B Accuracy](<img width="1400" height="800" alt="phase_b_acc" src="https://github.com/user-attachments/assets/098a9dcd-89d5-47fa-a1ff-4f4ac0fadfa1" />
)  
![Phase B Loss](<img width="1400" height="800" alt="phase_b_loss" src="https://github.com/user-attachments/assets/0512598d-9eb6-4f14-a5c0-d7d0e89503b6" />
)

---

### Progressive Resizing (320×320)
![Progressive Resizing Accuracy](<img width="1400" height="800" alt="resize_acc" src="https://github.com/user-attachments/assets/492c68af-7dc5-4819-a9c2-4314d11688bc" />
)  
![Progressive Resizing Loss](<img width="1400" height="800" alt="resize_loss" src="https://github.com/user-attachments/assets/37553d1d-3dc1-4812-8a7a-ebfbbc90e7ae" />
)

---

## Acknowledgements
- Kaggle (P100, T4 GPUs)  
- TensorFlow/Keras EfficientNetV2  
- [Vikas Chauhan Dataset]([http://vision.stanford.edu/aditya86/ImageNetDogs/](https://www.kaggle.com/datasets/vikaschauhan734/120-dog-breed-image-classification/data))  

---

### Final Note
Through **careful staged training, progressive resizing, and TTA**, EfficientNetV2-B2 achieved **90.3% test accuracy** on 120 dog breeds.  
This establishes a strong baseline — future improvements could come from **larger EfficientNetV2 variants (B3/B4)** or **hybrid ensembles**.

## Author
Devashish Mishra
