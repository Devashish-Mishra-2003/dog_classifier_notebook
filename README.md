# Dog Breed Classification with EfficientNetV2-B2

This repository documents the development of a high-performance deep learning pipeline for **120-class dog breed classification** using **EfficientNetV2-B2**.  
The project was conducted in a Kaggle environment with a focus on overcoming practical training challenges, systematically improving accuracy, and achieving near state-of-the-art results under limited compute.

---

## Project Overview
- **Dataset**: ~20k labeled dog images across 120 breeds  
- **Backbone**: EfficientNetV2-B2 pretrained on ImageNet  
- **Training Strategy**: staged fine-tuning, progressive resizing, and test-time augmentation (TTA)  
- **Final Test Accuracy**: **90.3%**

---

## Training Strategy

### Phase A – Head Training
- Backbone frozen, only classification head trained.  
- Used low learning rate to avoid NaNs.  
- Achieved **88.7% validation accuracy**.  

### Phase B – Fine-Tuning
- Unfroze top 40 layers of EfficientNetV2-B2.  
- Learning rate schedule: 1e-5 → 1e-6.  
- Achieved **89.0% test accuracy**.  

### Progressive Resizing
- Input resolution increased from 300×300 → 320×320.  
- Improved detail recognition and reduced stagnation.  
- Achieved **89.7% test accuracy**.  

### Test-Time Augmentation (TTA)
- Applied random flips, rotations, and zooms at inference.  
- Averaged predictions across multiple augmented views.  
- **Final test accuracy: 90.3%**.

---

## Key Challenges and Solutions

| Challenge | Cause | Solution |
|-----------|-------|----------|
| NaN losses during training | Incorrect preprocessing and scaling | Corrected normalization pipeline |
| Collective reduce errors (multi-GPU) | Sync failures across GPUs | Switched to single P100 GPU |
| High early loss | Learning rate too high | Adopted staged training (head → fine-tune) |
| Validation stagnation | Resolution bottleneck | Progressive resizing |
| Overfitting risk | Train–val gap widening | Strong online augmentation + dropout |
| Inference instability | Single-pass prediction | TTA ensemble |

---

## Results Summary

- **Phase A (300×300)** → Validation accuracy: ~88.7%  
- **Phase B (fine-tune, 300×300)** → Test accuracy: ~89.0%  
- **Progressive Resizing (320×320)** → Test accuracy: ~89.7%  
- **TTA Ensemble (320×320)** → **Final test accuracy: 90.3%**

---

## Training Curves

### Phase A (Head Training, 300×300)
![Phase A Accuracy](https://github.com/user-attachments/assets/9101aafc-38a6-495e-aef6-f95e0cb04b7b)  
![Phase A Loss](https://github.com/user-attachments/assets/b0832729-832d-46b8-9592-8c26a55183ff)  

### Phase B (Fine-Tuning, 300×300)
![Phase B Accuracy](https://github.com/user-attachments/assets/098a9dcd-89d5-47fa-a1ff-4f4ac0fadfa1)  
![Phase B Loss](https://github.com/user-attachments/assets/0512598d-9eb6-4f14-a5c0-d7d0e89503b6)  

### Progressive Resizing (320×320)
![Resizing Accuracy](https://github.com/user-attachments/assets/492c68af-7dc5-4819-a9c2-4314d11688bc)  
![Resizing Loss](https://github.com/user-attachments/assets/37553d1d-3dc1-4812-8a7a-ebfbbc90e7ae)  

---

## Acknowledgements
- Kaggle (P100, T4 GPUs)  
- TensorFlow/Keras EfficientNetV2  
- [Vikas Chauhan Dataset](https://www.kaggle.com/datasets/vikaschauhan734/120-dog-breed-image-classification/data)  

---

## Author
**Devashish Mishra**  

