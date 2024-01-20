# MAGVLT: Masked Generative Vision-and-Language Transformer

--- 

The official PyTorch implementation of [**Masked Generative Vision-and-Language Transformer**](https://arxiv.org/abs/2303.12208), CVPR 2023

MAGVLT is a unified non-autoregressive generative Vision-and-Language (VL) model which is trained via 1) three multimodal masked token prediction tasks along with two sub-tasks, 2) step-unrolled masked prediction and 3) MixSel.

<div align="center">
<figure>
  <img alt="" src="./assets/main.png">
</figure>
</div>

## Requirements
We have tested our codes on the environment below
```angular2html
PyTorch 1.10.0
Python 3.7.11
Ubuntu 18.04
```
Please run the following command to install the other dependencies
```angular2html
pip install -r requirements.txt
```

## Coverage of Released Codes
- Implementation of MAGVLT
- Pretrained checkpoints of MAGVLT-base and MAGVLT-large
- Sampling pipelines of MAGVLT: 
  - Generate image from text
  - Generate text from image
  - Generate image from text and image (inpainting)
  - [ ] Generate text from text and image (infilling)
  - [ ] Generate text and image (unconditional generation)
- [ ] Evaluation pipelines of MAGVLT on downstream tasks 
- [ ] Training pipeline with data preparation example



## Pretrained Checkpoints
MAGVLT uses VQGAN (vqgan_imagenet_f16_16384) as the image encoder which can be downloaded from [this repo](https://github.com/CompVis/taming-transformers).

|      Model       |         #Parameters          | CIDEr (↑, coco) | CIDEr (↑, NoCaps) | FID (↓, coco) |
|:----------------:|:----------------------------:|:---------------:|:-----------------:|:-------------:|
| [MAGVLT-base](https://arena.kakaocdn.net/brainrepo/models/magvlt/magvlt-it2it-base.ckpt)  | 371M                         |      60.4       |       46.3        |     12.08     |
| [MAGVLT-large](https://arena.kakaocdn.net/brainrepo/models/magvlt/magvlt-it2it-large.ckpt) |             840M             |      68.1       |       55.8        |     10.14     |

## Sampling
We provide the following sampling codes.
```angular2html
python sampling_t2i.py  --prompt=[YOUR PROMPT] 
                        --config_path=configs/magvlt-it2it-base-sampling.yaml 
                        --model_path=[MAGVLT_MODEL_PATH] 
                        --stage1_model_path=[VQGAN_MODEL_PATH]

python sampling_i2t.py  --source_img_path=[YOUR_IMAGE_PATH] 
                        --config_path=configs/magvlt-it2it-base-sampling.yaml 
                        --model_path=[MAGVLT_MODEL_PATH] 
                        --stage1_model_path=[VQGAN_MODEL_PATH]

python sampling_it2i.py --prompt=[YOUR PROMPT] 
                        --source_img_path=[YOUR_IMAGE_PATH] 
                        --config_path=configs/magvlt-it2it-base-sampling.yaml 
                        --model_path=[MAGVLT_MODEL_PATH] 
                        --stage1_model_path=[VQGAN_MODEL_PATH]
```


## Evaluation
### T2I evaluation 
  - Setting
      - Download folders {"pycocoevalcap", "pycocotools"} and files {cocoeval.py, get_stanford_models.sh} from https://github.com/yxuansu/MAGIC/tree/main/image_captioning/evaluation, and place them under ./evaluation.
      - cd ./evaluation; bash get_stanford_models.sh 
```angular2html
python eval_i2t.py cfg_path=configs/magvlt-it2it-base-eval_i2t.yaml checkpoint_path=[MAGVLT_MODEL_PATH] result_path=none result_file_path=[RESULT_JSON_FILE_PATH]
```
### I2T evaluation 
````angular2html
python eval_t2i.py cfg_path=configs/magvlt-it2it-base-eval_t2i.yaml result_file_path=[RESULT_IMAGE_DIR_PATH] checkpoint_path=[MAGVLT_MODEL_PATH]
````
## Citation

```bibtex
@InProceedings{Kim_2023_CVPR,
    author    = {Kim, Sungwoong and Jo, Daejin and Lee, Donghoon and Kim, Jongmin},
    title     = {MAGVLT: Masked Generative Vision-and-Language Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23338-23348}
}
```

## Contact
Donghoon Lee, [dhlee@kakaobrain.com](dhlee@kakaobrain.com)  
Jongmin Kim, [jmkim@kakaobrain.com](jmkim@kakaobrain.com)  

## License
This project is released under [MIT license](./LICENSE).
