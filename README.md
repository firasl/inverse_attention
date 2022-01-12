# inverse_attention
This repository provides the official implementation of *['Learning to ignore: rethinking attention in CNNs'](https://arxiv.org/abs/2111.05684)*  accepted in *[BMVC 2021](https://www.bmvc2021.com/)*.


# Learning to ignore: rethinking attention in CNNs
*Abstract:* 

Recently, there has been an increasing interest in applying attention mechanisms in Convolutional Neural Networks (CNNs) to solve computer vision tasks. Most of these methods learn to explicitly identify and highlight relevant parts of the scene and pass the attended image to further layers of the network. In this paper, we argue that such an approach might not be optimal. Arguably, explicitly learning which parts of the image are relevant is typically harder than learning which parts of the image are less relevant and, thus, should be ignored.  In fact, in vision domain, there are many easy-to-identify patterns of irrelevant features. For example, image regions close to the borders are less likely to contain useful information for a classification task. Based on this idea, we propose to reformulate the attention mechanism in CNNs to learn to ignore instead of learning to attend. Specifically, we propose to explicitly learn irrelevant information in the scene and suppress it in the produced representation, keeping only important attributes. This implicit attention scheme can be incorporated into any existing attention mechanism. In this work, we validate this idea using two recent attention methods Squeeze and Excitation (SE) block and Convolutional Block Attention Module (CBAM). Experimental results on different datasets and model architectures show that learning to ignore, i.e., implicit attention, yields superior performance compared to the standard approaches.


## Dependencies
The project was tested in Python 3 and Tensorflow 2. Run pip install -r requirements.txt to install dependent packages. Parts of the code are based on *['CBAM-TensorFlow'](https://github.com/kobiso/CBAM-tensorflow)*  

## Running the code:
To test our approach on ImageNet, run main_imagenet.py. You need to: 
1/ specify *dataset_dir* the TF-record directory of the dataset.
2/ choose the attention model to use, i.e., *attention_module*.

To test our approach on CIFAR10 or CIFAR100, run main_CIFAR.py. You need to: 
1/ specify *dataset* and *num_classes*
2/ choose the attention model to use, i.e., *attention_module*.






## Cite This Work

```
@article{laakom2021learning,
  title={Learning to ignore: rethinking attention in CNNs},
  author={Laakom, Firas and Chumachenko, Kateryna and Raitoharju, Jenni and Iosifidis, Alexandros and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2111.05684},
  year={2021}
}
```
