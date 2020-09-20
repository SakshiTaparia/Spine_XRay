## Segmentation and Classification of Spine X-Rays

The first half of the draft documents segmentation of X-ray images and the second half experiments with segmented images as well as original X-rays to perform binary classification of spines into "normal" and "damaged" categories. U-Net inspired architectures outperform all other models for segmentation of biomedical images because of their ability to learn from limited data. A DICE score approximately between 60-80 is achieved in 6 out of 8 segmentation cases. Further, training ResNet-50 on X-ray images, coupled with transfer learning and feature extraction by pre-training on imagenet data set achieves classification accuracy of over 85%. Since pre-trained models have already learnt low-level features like edges and textures, they do not need to be re-learned during model fine-tuning.

All the details related to the experiments and accuracies obtained are included in the final report.
