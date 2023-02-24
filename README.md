# MobileNet in PyTorch

## Main contribution

User can convert pretrained models **from official TensorFlow implementation** and many other versions to this **pytorch
implementation**

## Examples

Evaluation results on ImageNet   
The following pretrained models are converted from different sources using `convert.py` or `converter/converter_*.py`

| Model                                                                                                                                                                                                         | Config (α, dim) | # of mult-adds | Top 1 Accuracy | Top 5 Accuracy | F1 score |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|----------------|----------------|----------------|----------|
| [MobileNet](https://drive.google.com/file/d/11fuptwvQWr-G78RU7W5KugZlasYGNnwG/view?usp=sharing) <br/> from [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet) | (0.5, 160)      | 76.54 M        | 36.06%         | 59.91%         | 0.3711   |
| [MobileNet](https://drive.google.com/file/d/1gFH0c6YregiiFctTFBIjr_7ffHZIUfxp/view?usp=sharing) <br/>from [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet)  | (1.0, 224)      | 568.76 M       | 64.21%         | 85.12%         | 0.6411   |
| [MobileNetV2](https://drive.google.com/file/d/18ZOdYjS1uqMt1eUEsW6WQNwrVWDRL6cR/view?usp=sharing) <br/>from [PyTorch](https://drive.google.com/file/d/1VExkcO5r7g3-jn4nu_Jhz5__pj4Zxqz8/view?usp=sharing)     | (1.0, 224)      | 300.81 M       | 69.84%         | 89.26%         | 0.6952   |

## Todo

1. ~~implement conversion for different res, alpha for **MobileNet**~~  :white_check_mark:
2. ~~support alpha and input resolution hyperparameters in **MobileNetV2**~~  :white_check_mark:
3. implement conversion for **MobileNetV2**
4. add **MobileNetV3**
5. better abstract the save model function
6. allows loading pretrained model with different classes
7. clean up convert.py codes + accept command arguments
8. complete this README.md file
9. support quick start in Google Colab
10. apply tqdm to display progress
 