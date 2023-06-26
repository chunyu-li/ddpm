# DDPM Pytorch Implementation

This repo implemented a very basic version of diffusion model, also known as **Denoising Diffusion Probabilistic Models**.

Reference materials:

- Original paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Youtube tutorial video: [Diffusion models from scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=942s)
- Colab Notebook: [A Diffusion Model from Scratch in Pytorch](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=bpN_LKYwuLx0)

## Dataset

As dataset we use the StandordCars Dataset, which consists of around 8000 images in the train set. It can be downloaded from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset). Unzip the dataset compressed file and rename the folder as `stanford_cars`, place it into the root folder.

## Investigating the dataset

Display some images of the StandordCars Dataset.

```shell
python3 show_dataset.py
```

## Forward noising

We can perform forward noising to show the effect of diffusion intuitively.

```shell
python3 forward_noising.py
```

## Build the U-Net model

### Key Takeaways

- We use a simple form of a UNet for to predict the noise in the image
- The input is a noisy image, the ouput the noise in the image
- Because the parameters are shared accross time, we need to tell the network in which timestep we are
- The Timestep is encoded by the transformer Sinusoidal Embedding
- We output one single value (mean), because the variance is fixed

to show the architecture of u-net model, we can run:

```shell
python3 unet.py
```

## Train the model

Start training the U-Net model, after training, the model will save in folder `trained_models`.

```shell
python3 training_model.py
```

## Sample an image

```shell
python3 sampling.py
```