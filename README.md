Inference Notebook (a few different models): <a href="https://colab.research.google.com/drive/1Jgj0uaALtile2iyqlN1r72UYRe9SZw-H?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>

## [Link to YouTube Presentation](https://www.youtube.com/watch?v=UG7Q50J6m74)

# ["CapDec: Text-Only Training for Image Captioning using Noise-Injected CLIP"](https://arxiv.org/abs/2211.00575), EMNLP 2022 (findings).
## Official implementation of the paper 
As shown in the paper, CapDec achieves SOTA image-captioning in the setting of training without even a single image.
This is the formal repository for CapDec, in which you can easily reproduce the papers results. 
You can also play with our [inference notebook]("https://colab.research.google.com/drive/1Jgj0uaALtile2iyqlN1r72UYRe9SZw-H?usp=sharing") <a href="https://colab.research.google.com/drive/1Jgj0uaALtile2iyqlN1r72UYRe9SZw-H?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a> to see how the model works, and try it on your OWN images with different CapDec-based models.

<img src="https://github.com/DavidHuji/CapDec/blob/main/figures/fig1.png" width=70% height=70%>


## FlickrStyle7k Examples
Examples for styled captions of CapDec on FlickrStyle10K dataset:

<img src="https://github.com/DavidHuji/CapDec/blob/main/figures/examples.png" width=50% height=50%>

## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/DavidHuji/CapDec && cd CapDec
conda env create -f others/environment.yml
conda activate CapDec
```

# Datasets
1. Download the datasets using the following links: [COCO](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits), [Flickr30K](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits), [FlickrStyle10k](https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip).
2. Parse the data to the correct format using our script parse_karpathy.py, just make sure to edit head the json paths inside the script.


# Training
Make sure to edit head the json or pkl paths inside the scripts.
1. Extract CLIP features using the following script:
```
python embeddings_generator.py -h
```

2. Training the model using the following script:
```
python train.py --data clip_embeddings_of_last_stage.pkl --out_dir ./coco_train/ --noise_variance 0.016
```

**There are a few interesting configurable parameters for training as follows. 
You can view it by running 'python train.py --help'**
```
optional arguments:
  -h, --help            show this help message and exit
  --data                path to clip embeddings of captions generated by the attached embeddings_generator script
  --val_pt              path to clip embeddings of validations set
  --pretrain_weights    path to pretrained weights, if not specified, will train from scratch
  --out_dir             path to output directory
  --add_modality_offset train with modality offset that was pre calculated at others/CLIP_embeddings_centers_info.pkl
  --prefix PREFIX       prefix for saved filenames
  --noise_variance      noise variance
  --uniform_noise       use uniform noise instead of gaussian
  --dont_norm           dont normalize CLIP embeddings
  --lr LR               learning rate
  --epochs EPOCHS       number of epochs
  --save_every          save every n epochs
  --prefix_length       prefix length
  --prefix_length_clip  prefix length for clip
  --bs BS               batch size
  --only_prefix         train only the mapper between CLIP and GPT, while GPT is frozen
  --mapping_type        type of architurctre between CLIP and GPT (mlp/transformer)
  --num_layers          number of layers in the mapper
  --is_not_rn           Choose the CLIP backbone: False for RN, True for ViT
  --use_image_embedding_as_clipcap       use image embedding as ClipCap
```

# Evaluation
For evaluation, we used a repository that adapts the [COCO evaluation script](https://github.com/tylin/coco-caption) to python 3 [here](https://github.com/sks3i/pycocoevalcap). \
In order to evaluate the model, you need to first generate the captions using the following command (just edit the images_root path inside in order to direct it to the right ground truth annotations).
```
python predictions_runner.py  --checkpoint path_to_checkpoints.pt --dataset_mode 0 
```

# Pre Trained Models
We upload the trained weights that we used for creating Fig.3 in the paper, so you can download it if you do not want to wait for training.
[Here](https://drive.google.com/drive/folders/17axuxZ90uRD3-ohhXBhXvQJf1R6kolVw?usp=sharing) are the trained weights of 9 different noise levels. 


# Open Text Training - Training on any corpus as Harry Potter Books, Shakespeare Plays, or The New York Times (was NOT presented at the paper).
A cool application of CapDec is to create captions in the style of a specific corpus that was not even in the form of captions. Ideally, any given text can be used to train CapDec's decoder to decode CLIP embeddings. It enables the elimination of the need to have any sort of captions textual data. Moreover, it enables captioning model that is in the specific style of the given text. for that, we can first pre-train with images as regular ClipCap, then we fine-tune as in CapDec with text only when the text data is a combination of half COCO captions and half sentences from the open text (HP or News) sentences in length between 4 to 20 words.

In order to reproduce that, all you need is to create sentences out of the open text, save them in the right format as the json we have for COCO and then repeat the steps mentioned above for training.
For that you can use the attached script at others/hp_to_coco_format.py.
Although you can use any sort of text for that, you can download the data we used, from the following links: [Harry Potter](https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7), [Shakespeare](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays), [News](https://www.kaggle.com/datasets/sbhatti/news-articles-corpus)
You can see an example of the correct format for training at others/parssed_sheikspir_alllines_111k.json

Here are a few examples of the results of training on the Harry Potter books, Shakespeare plays, and news articles:
<img src="https://github.com/DavidHuji/CapDec/blob/main/figures/opent1.png" width=50% height=50%>
<img src="https://github.com/DavidHuji/CapDec/blob/main/figures/opent2.png" width=50% height=50%>
<img src="https://github.com/DavidHuji/CapDec/blob/main/figures/opent3.png" width=50% height=50%>

# Fairness.
In principle, CapDec could be useful for creating captions that are fairer by fixing biases in the data. For example, we can de-bias the textual data by changing gender terms. That trick is possible only in our setting of text-only training (i.e. image editing is much more complex than text editing). More generally, any sort of bias in the data could be manipulated in the data by simple text editing.

In order to examine this idea we implemented text-data bias editing for gender terms. You can use it by adding the flag --fix_gender_imbalance_mode when you run the script of embeddings_generator.py. It has three modes: 0 - no fixing, 1 for both genders, 2 for men only, 3 for women only. For example when running: python embeddings_generator.py --fix_gender_imbalance_mode 2 any gender term of male will be exchanged with a probability of 0.5 to a female term, resulting in more balanced data (in COCO there are much more male captions than a woman as shown by ['woman also snowboard 2018'](https://arxiv.org/abs/1803.09797)).

## Citation
If you use this code for your research, please cite:
```
@article{nukrai2022text,
  title={Text-Only Training for Image Captioning using Noise-Injected CLIP},
  author={Nukrai, David and Mokady, Ron and Globerson, Amir},
  journal={arXiv preprint arXiv:2211.00575},
  year={2022}
}
```

## Acknowledgments
This repository is based on [CLIP](https://github.com/openai/CLIP), [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and [pycocotools](https://github.com/sks3i/pycocoevalcap) repositories.


## Contact
For any issue please feel free to contact me at: nukraidavid@mail.tau.ac.il.
