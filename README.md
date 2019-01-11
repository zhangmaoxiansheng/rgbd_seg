# RGBD_SEG FOR HUMAN

Implemented Architecture of Pyramid Scene Parsing Network in Keras.Ane using depth information to refine the mask.
(However, in this case the depth result is more reliable. Actually, maybe using PSPnet to fine the depth result is a better way)

### Setup
1. Install dependencies:
    * Tensorflow (-gpu)
    * Keras
    * numpy
    * scipy
    * pycaffe(PSPNet)(optional for converting the weights) 
    ```bash
    pip install -r requirements.txt --upgrade
    ```
2. Converted trained weights are needed to run the network.
Weights(in ```.h5 .json``` format) have to be downloaded and placed into directory ``` weights/keras ```


Already converted weights can be downloaded here:

 * [pspnet50_ade20k.h5](https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1)
[pspnet50_ade20k.json](https://www.dropbox.com/s/v41lvku2lx7lh6m/pspnet50_ade20k.json?dl=1)
 * [pspnet101_cityscapes.h5](https://www.dropbox.com/s/c17g94n946tpalb/pspnet101_cityscapes.h5?dl=1)
[pspnet101_cityscapes.json](https://www.dropbox.com/s/fswowe8e3o14tdm/pspnet101_cityscapes.json?dl=1)
 * [pspnet101_voc2012.h5](https://www.dropbox.com/s/uvqj2cjo4b9c5wg/pspnet101_voc2012.h5?dl=1)
[pspnet101_voc2012.json](https://www.dropbox.com/s/rr5taqu19f5fuzy/pspnet101_voc2012.json?dl=1)

## Convert weights by yourself(optional)
(Note: this is **not** required if you use .h5/.json weights)

Running this needs the compiled original PSPNet caffe code and pycaffe.

```bash
python weight_converter.py <path to .prototxt> <path to .caffemodel>
```

## Usage:


for large dataset
```bash
python rgbd_seg.py -m <model> -g <./rgb_folder/*.png> -gd<./depth_folder/*.png> -o <output_path>

python rgbd_seg.py -m <model> -i <input_image>  -id <input_depth> -o <output_path>

List of arguments:
```bash
 -m --model        - which model to use: 'pspnet50_ade20k', 'pspnet101_cityscapes', 'pspnet101_voc2012'
    --id           - (int) GPU Device id. Default 0
 -s --sliding      - Use sliding window
 -f --flip         - Additional prediction of flipped image
 -ms --multi_scale - Predict on multiscale images
```






