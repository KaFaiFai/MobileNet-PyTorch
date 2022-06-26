# MobileNet in PyTorch

## Main contribution

User can convert pretrained models **from official TensorFlow implementation** and many other versions to this **pytorch
implementation**

## Examples
Evaluation results on ImageNet   
The following pretrained models are converted from different sources using `convert.py`

| Model                                              | Top 1 Accuracy | Top 5 Accuracy | F1 score |
|----------------------------------------------------|----------------|----------------|----------|
| [MobileNet][net1] <br/>from [TensorFlow][net1_src] | 64.21%         | 85.12%         | 0.6411   |
| [MobileNet][net2] <br/>from [wjc852456][net2_src]  | 67.95%         | 88.13%         | 0.6762   |
| [MobileNetV2][net3] <br/>from [PyTorch][net3_src]  | 69.84%         | 89.26%         | 0.6952   |


[MobileNet][net1]

[net1]: (https://drive.google.com/file/d/1gFH0c6YregiiFctTFBIjr_7ffHZIUfxp/view?usp=sharing)

[net1_src]: (https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet)

[net2]:(https://drive.google.com/file/d/1CSSJi0brQZ0Le89XtYvrXXfpaFLyusSg/view?usp=sharing)

[net2_src]:(https://github.com/wjc852456/pytorch-mobilenet-v1.git)

[net3]:(https://drive.google.com/file/d/1CSSJi0brQZ0Le89XtYvrXXfpaFLyusSg/view?usp=sharing)

[net3_src]:(https://drive.google.com/file/d/1VExkcO5r7g3-jn4nu_Jhz5__pj4Zxqz8/view?usp=sharing)

## Todo

1. ~~implement conversion for different res, alpha for **MobileNet**~~  :white_check_mark:  
2. implement conversion for **MobileNetV2**
3. add **MobileNetV3** and **MobileViT**
4. allows loading pretrained model with different classes
5. clean up convert.py codes + accept command arguments
6. complete this README.md file
7. support quick start in Google Colab
8. apply tqdm to display progress
 