-----

# VLM Readme

This repository contains the code for a Vision-Language Model (VLM) built using PyTorch. This VLM is designed to understand and process information from both visual (images) and textual (natural language) inputs, enabling tasks like image captioning and extracting features from an image.

-----

## Features

  * **Multimodal Understanding:** Seamlessly integrates visual and textual information.
  * **PyTorch Implementation:** Built with PyTorch for flexibility and performance.
  * **Modular Design:** Easily extensible to incorporate new architectures or components.
-----

## Installation

The following libraries were used:
pytorch==2.2.0
torchtext==0.17.0
torchvision==0.17.0
sklearn
google-genai
-----

### Inference
For inference please refer to the inference.ipynb file. The model takes an image, and generates features for the given image, the features are then passed into a GPT-2 model for further enhancement of the captions. This caption is then passed to gemini 2.0 Flash model along with the question posed by the user.


-----

## Model Architecture


Our VLM architecture comprises three main components:

1.  **Vision Encoder:** We use an vit_b16 vision encoder whoose `mlp` layers are finetuned using `LoRA`
2.  **Language Encoder:** For this we use a combination of positional embedding and an embedding layer which uses pre-trained GloVe weights.
3.  **Multimodal Fusion Module:** Combines the visual and textual features to enable joint understanding in the decoder-transformer section of the model.
4.  **Task-Specific Head:** After the tensors are processed by the decoder-transformer layer, the resultant tensors are passed into a linear layer before `logits` can be generated.

This VLM also uses GPT-2 and Gemini 2.0. Once the VLM gives features of the image, this is refined by the `GPT-2` model which is then combined with the query of the user and then passed to the `Gemini 2.0` model.

-----

## Training

### Datasets

The dataset used in the training of this VLM is Flicker 8k Dataset

### Configuration

  * `learning_rate`: 1
  * `batch_size`: 50
  * `num_epochs`: This can vary based on the availability of gpu's for you
  * `optimizer`: Adam
  * `loss_function`: CrossEntropy
  * `vision_encoder_weights`: We use pretrained vit_b16 encoder
  * `language_encoder_weights`: Pre-trained GloVe Embeddings

## Evaluation

The `evaluate()` returns the accuracy of the model, it evaluates the model for each data element and compares the textual output given by the model and the textual info present in the test-dataset. It gives the average accuracy over the entire test-dataset.

-----


## Contact

If you have any questions or suggestions, feel free to open an issue or contact akanish327@gmail.com

-----
