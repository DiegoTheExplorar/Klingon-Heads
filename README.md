# Klingon Heads

**Project Title**: Klingon Heads  
**Authors**: Arvind Natarajan, Sivakumar Karthikraj  
**Date**: 2024

## Overview

Klingon Heads is a web application designed to facilitate translation between English and Klingon, a language from the Star Trek universe. Our project combines state-of-the-art machine learning techniques with an accessible and user-friendly interface to deliver accurate, context-aware translations. Despite the challenges faced during the development process, we have made significant progress towards creating a robust and reliable tool for both casual users and linguistic enthusiasts.

## Features

- **User Authentication**: Secure Google Auth integration to personalize the user experience.
- **Bidirectional Translation**: Real-time translation between English and Klingon via a GRU model.
- **Favorites and History**: Save and track your translations using Firestore.
- **Flashcards**: Learn Klingon phrases through interactive flashcards.
- **Quiz Feature**: Test your knowledge with different types of quizzes.
- **Speech to Text**: Convert spoken language to text for translation.
- **Image to Text**: Extract and translate text from images using Tesseract.js.

## Model Training

### Byte Pair Encoding (BPE) Tokenizer

For the translation tasks, we implemented a Byte Pair Encoding (BPE) tokenizer. BPE was chosen to effectively manage the out-of-vocabulary (OOV) issues that plagued earlier models, allowing for better generalization by recognizing both subwords and the complex words they form.

### GRU Model

Our initial approach involved training a GRU-based sequence-to-sequence (seq2seq) model. We trained the model locally, making several adjustments to the hyperparameters, learning rates, and data preprocessing methods. Unfortunately, we encountered persistent issues with overfitting and poor performance, regardless of the changes we implemented. The GRU model, while simpler and less computationally intensive, did not meet our performance expectations.

### Challenges and Solutions

- **Overfitting**: Despite experimenting with various regularization techniques (e.g., dropout, weight decay) and adjusting the learning rates, the GRU model consistently overfitted to the training data. This led to poor generalization on unseen data.
- **Hyperparameter Tuning**: Extensive hyperparameter tuning was conducted, including adjustments to the number of layers, units per layer, and learning rate schedules. Unfortunately, these efforts did not yield significant improvements in model performance.

Given these challenges, we eventually transitioned to using the T5 Transformer model for better performance, especially with the use of mixed precision training on more powerful hardware.  You may still find our code for the GRU models on backend(scrapped) directory.

## Deployment

The application is currently deployed and accessible at [Klingon Heads](https://klingon-heads.vercel.app/). However, please note:

- The **Quiz and Learning features** are not available at this time.
- The translation models, deployed on Hugging Face and accessed via its serverless API, may occasionally give an error. This is due to the limitations and the current state of the model deployment.

## Tech Stack

- **Backend**: Python, Pytorch, Hugging Face, Firebase
- **Frontend**: React, Heroku, Vercel
- **Model Deployment**: Hugging Face Spaces, Gradio

## Conclusion

While the GRU-based approach presented significant challenges, including overfitting and inadequate performance improvements despite extensive tuning, the overall project demonstrates a strong foundation for machine translation applications. Future work may involve further refining the model, exploring more advanced architectures like transformers, and enhancing the user experience based on feedback.

