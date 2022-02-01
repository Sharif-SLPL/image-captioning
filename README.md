<div align="center">
<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3"/>
</a>
<a href="https://www.tensorflow.org/">
    <img src="https://github.com/aleen42/badges/raw/master/src/tensorflow.svg" alt="built with Tensorflow"/>
</a>
<a href="https://github.com/Sharif-SLPL/image-captioning">
    <img src="https://github.com/aleen42/badges/raw/master/src/github.svg" alt="hosted on Github"/>
</a>
<a href="https://colab.research.google.com/github/Sharif-SLPL/image-captioning/blob/master">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Document"/>
</a>
</div>

# Image Captioning
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions. This work (Mohammad Mohammadifar's master thesis) makes captions for images in Persian.

| Example 1 | Example 2 |
|:--:|:--:|
| <img src="https://user-images.githubusercontent.com/43045767/151662991-4f0d4e3d-f740-47fa-a67c-c171b7795461.jpg" alt="activities-for-younger-kids" width="300px" height="300px"> | <img src="https://user-images.githubusercontent.com/43045767/151663183-cf8f2f43-df89-41ae-9eaa-347d935e0af8.jpg" alt="activities-for-younger-kids" width="300px" height="300px"> |
| کودکی با لباس آبی در حال بازی با توپ قرمز است  | دختربچه ای در حال بازی است |


## Requirements
Install requirements by 

```
pip install -r requirements.txt
```

## Dataset
This work uses the Flicker8 Dataset which is available [here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip). You can donwload and unzip it by command bellow:

```
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip
```

## 1. Pre-processing
In this section three steps of pre-processing are needed which are as fallows

### Feature Extraction
First part will extract features from Flicker8 dataset and save them into a `features.pkl` file

```
python 1_feature_exctract.py
```

### Text Preparation
Second part will prepare texts from the `farsi_8k_human.txt` and save them into a `descriptions.txt` file

```
python 1_text_prep.py
```

### Tokenizer
Third part will train a tokenizer based on train set image descriptions and save it into a `tokenizer.pkl`

```
python 1_tokenizer.py
```

## 2. Training
In this section we train our Persion image captioning model based on `features.pkl`, `descriptions.txt` and `tokenizer.pkl`. This process may take a while. At the end it will make some files like `model-ep*-loss*-val_loss*-attention-final.h5` and each of them can be used for evaluation section. You should rename your preferred one to `model.h5`. You can either downlowad our pretrianed model from [here](https://drive.google.com/file/d/1sB_mfQ0k0amO_vL9JXJ859KrsJpVn0Kg/view).

```
python 2_train_nic2.py
```

## 3. Evaluation and test
In this section you can evaluate the trained model and then test it on any given image

### Evaluation
This part will evaluate the model on test data using [BLEU](https://en.wikipedia.org/wiki/BLEU) score

```
python 3_eval.py
```

### Test
You can get the caption of your images using bellow command

```
python test.py [path_to_image]
```

For example for bellow picture we should have

<div align="center">
<img src="https://raw.githubusercontent.com/Sharif-SLPL/image-captioning/main/test.jpg" alt="activities-for-younger-kids" width="200px" height="200px">
</div>

```
$ python test.py test.jpg
startseq یک زن در حال عکس گرفتن از یک صخره بزرگ است endseq
```
