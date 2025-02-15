# CapDec-Based Image Captioning with Noise Injection

## Project Overview  
This project is part of our Deep Learning Final Project in the Industrial Engineering & Management (IEM) Department. We explored and implemented **CapDec**, a method for text-only image captioning using noise-injected CLIP embeddings. The goal was to improve image captioning without direct image supervision by leveraging pre-trained language models.

## What is CapDec?  
**CapDec** is an image captioning model that does not require direct image supervision. Instead, it uses **text embeddings** derived from captions and applies **Gaussian noise injection** to simulate image embeddings. This approach helps bridge the gap between visual and textual modalities.

**Key Features:**  
- Uses **pre-trained CLIP models** to generate embeddings.  
- Trains with **only text data** rather than paired image-caption datasets.  
- Applies **noise injection** to improve performance on image captioning tasks.  

For more details, refer to the original **CapDec repository**:  
[CapDec GitHub Repository](https://github.com/DavidHuji/CapDec) (Nukrai, Mokady, & Globerson, 2022)

## CLIP Model  
CapDec relies on **CLIP**, a **multimodal model developed by OpenAI** (Radford et al., 2021). CLIP learns joint representations of images and text using contrastive learning, making it well-suited for tasks like **zero-shot learning** and **cross-modal retrieval**.

## Implementation and Training  
We followed a two-phase training process:  

1. **Text Embedding Creation**  
   - Generated **text embeddings** from image captions using a frozen CLIP model.  
   
2. **Training the Decoder**  
   - Trained a decoder-only model to reconstruct captions from noisy text embeddings.  
   - Applied **Gaussian noise injection** to the embeddings during training.

## Inference  
During inference, the model:  
1. Converts images into **CLIP embeddings** using a frozen CLIP model.  
2. Uses the trained decoder to generate captions based on the embeddings.  

## Noise Injection Techniques  
We experimented with different noise injection strategies:  
- **Gaussian Noise:** Random perturbations to embeddings.  
- **T-Distribution Noise:** Alternative statistical noise distribution.  
- **Normalized Gradient Noise:** Adjusts noise dynamically based on gradient updates.  

## Hyperparameter Tuning  
To improve the model’s performance, we explored:  
- Number of **attention heads**  
- Number of **transformer layers**  
- **Noise injection intensity**  

## Model Evaluation  
We evaluated the model using **cosine similarity** between generated captions and human-annotated ground truth captions. The evaluation process included:  
- Generating **five captions** per image.  
- Comparing them against **three to five real captions** per image.  
- Computing an average similarity score across all real captions.  

## Key Findings  
- **Pre-trained CapDec performed best overall.**  
- **Gradient noise injection improved performance** more effectively than other noise types.  
- **Transformer depth had mixed effects**, with no clear trend.  
- **Cosine similarity-directed noise was ineffective**, but further tuning might improve results.  

## References  
Nukrai, D., Mokady, R., & Globerson, A. (2022). Text-only training for image captioning using noise-injected CLIP. *arXiv preprint arXiv:2211.00575*.  
[CapDec GitHub Repository](https://github.com/DavidHuji/CapDec)  

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *International Conference on Machine Learning (ICML)*.  

Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. Y. (2022). Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. *Advances in Neural Information Processing Systems, 35*, 17612-17625.  
