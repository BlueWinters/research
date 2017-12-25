# Machine Learning Notes
Useful Links Of Machine Learning
- Neutral Network
- Conference Papers


## Neutral Network

### Autoencoders
- Adversarial Autoencoders ([pdf](https://arxiv.org/abs/1511.05644), [office-theano](https://github.com/hjweide/adversarial-autoencoder), [pytorch](https://github.com/fducau/AAE_pytorch))
- PixelGAN Autoencoders ([pdf]())

### Generative Model
- **GAN**: Generative Adversarial Networks ([pdf](https://arxiv.org/abs/1406.2661), code)
- **InfoGAN**: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets ([pdf](https://arxiv.org/abs/1606.03657), [code](https://github.com/openai/InfoGAN))
- **VAE**: Auto-Encoding Variational Bayes ([pdf](https://arxiv.org/abs/1312.6114), [tutorial](Tutorial on Variational Autoencoders), [tensorflow](https://github.com/y0ast/VAE-TensorFlow))
- **PixelRNN**: Pixel Recurrent Neural Networks ([pdf](https://arxiv.org/abs/1601.06759))
- **PixelCNN**: Conditional Image Generation with PixelCNN Decoders ([pdf](https://arxiv.org/abs/1606.05328))
- **GMM**: Generative Moment Matching Networks ([pdf]())

### Variants Of Convolution Network
- Convolution Guide: A guide to convolution arithmetic for deeplearning ([pdf](https://arxiv.org/abs/1603.07285))
- Deformable Convolutional Networks ([pdf](http://arxiv.org/abs/1703.06211), [mxnet](https://github.com/felixlaumon/deform-conv))
- Spatial Transformer Networks ([pdf](https://arxiv.org/abs/1506.02025), [tensorflow](https://github.com/tensorflow/models/tree/master/transformer))

### Architeture Of Deep Network
- **VggNet**: Very Deep Convolutional Networks for Large-Scale Image Recognition ([pdf](https://arxiv.org/abs/1409.1556))
- **ResNet**: Deep Residual Learning for Image Recognition ([pdf](https://arxiv.org/abs/1512.03385v1), [tensorflow](https://github.com/tensorflow/models/tree/master/resnet), [caffe](https://github.com/KaimingHe/deep-residual-networks), [mxnet](https://github.com/tornadomeet/ResNet))
- **Incption**
	+ **v1**: Going Deeper with Convolutions ([pdf](http://arxiv.org/abs/1409.4842))
	+ **v2**: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ([pdf](http://arxiv.org/abs/1502.03167))
	+ **v3**: Rethinking the Inception Architecture for Computer Vision ([pdf](http://arxiv.org/abs/1512.00567))
	+ **v4**: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning ([pdf](https://arxiv.org/abs/1602.07261))
- **DenseNet**: Densely Connected Convolutional Networks ([pdf](https://arxiv.org/abs/1608.06993), [code](https://github.com/liuzhuang13/DenseNet))
- **SiameseNet**: Siamese Neural Networks for One-shot Image Recognition ([pdf](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf))
	
### Tricks For Training Network
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ([pdf](https://arxiv.org/abs/1502.03167))

### Face Recognition
- DeepID1: Deep Learning Face Representation from Predicting 10,000 Classes ([pdf](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf))
- DeepID2: Deep Learning Face Representation by Joint Identification-Verification ([pdf](https://arxiv.org/abs/1406.4773))
- DeepID2+: Deeply learned face representations are sparse, selective, and robust ([pdf](https://arxiv.org/abs/1412.1265))
- DeepID3: Face recognition with very deep neural networks ([pdf](https://arxiv.org/abs/1502.00873)) 
- FaceNet: A Unified Embedding for Face Recognition and Clustering ([pdf](https://arxiv.org/abs/1503.03832.pdf))

### Deep Metric Learning
- TripletNet: Deep metric learning using Triplet network ([pdf](https://arxiv.org/abs/1412.6622))
	
### Uncategorized
- Deep Photo Style Transfer ([pdf](https://arxiv.org/abs/1703.07511), [code](https://github.com/luanfujun/deep-photo-styletransfer))
- Ladder Networks:
	+ Lateral Connections in Denoising Autoencoders (pdf)
	+ From Neural PCA to Deep Unsupervised Learning ([pdf](https://arxiv.org/abs/1411.7783))
	+ Semi-Supervised Learning with Ladder Networks ([pdf](https://arxiv.org/abs/1507.02672), [theano](https://github.com/CuriousAI/ladder), [tensorflow](https://github.com/rinuboney))
	+ Deconstructing the Ladder Network Architecture ([pdf](https://arxiv.org/abs/1511.06430))
- Learning Deep Features for Discriminative Localization ([pdf](https://arxiv.org/abs/1512.04150.pdf), [project](http://cnnlocalization.csail.mit.edu/))
- [A Year In Computer Vision](http://www.themtank.org/a-year-in-computer-vision)

### Image Inpainting
- Context Encoders: Feature Learning by Inpainting ([pdf](https://arxiv.org/abs/1604.07379), [lua](https://github.com/pathak22/context-encoder), [pytorch](https://github.com/BoyuanJiang/context_encoder_pytorch))


### Transfer Learning
- **CycleGAN**: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks ([pdf](https://arxiv.org/pdf/1703.10593.pdf), [office-pytorch](https://github.com/junyanz/CycleGAN))
- **DualGAN**: Unsupervised Dual Learning for Image-to-Image Translation ([pdf](https://arxiv.org/abs/1704.02510), [tensorflow](https://github.com/duxingren14/DualGAN))
- **DiscoGAN**: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks ([pdf](https://arxiv.org/pdf/1703.05192.pdf))


## Reading List
- Second Order
	- [x] Is Second-order Information Helpful for Large-scale Visual Recognition ([pdf]())
	- [x] Second-order Convolutional Neural Networks ([pdf]())
- Metric Learning
	- [ ] Improved Deep Metric Learning with Multi-class N-pair Loss Objective ([pdf]())
	- [ ] Deep metric learning using Triplet network ([pdf]())
- Others
	- [ ] DRAW:  A Recurrent Neural Network For Image Generation: ([pdf]())
	- High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis ([pdf](https://arxiv.org/abs/1611.09969), [lua](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting))
	- [x] Reconstruction of Hidden Representation for Robust Feature Extraction ([pdf]())
	- [x] Between-class Learning for Image Classification ([pdf]())
	- [x] Age Regression by Conditional Adversarial Autoencoder ([pdf]())
	- [x] Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction ([pdf]())
	- [x] Stacked Similarity-Aware Autoencoders ([pdf]())
	- [x] Joint Unsupervised Learning of Deep Representations and Image Clusters ([pdf]())
	- [ ] Variational Approaches for Auto-Encoding Generative Adversarial Networks ([pdf]())


## Conferentce Paper
- NIPS ([2017](https://nips.cc/Conferences/2017/Schedule?type=Poster), [2016](https://nips.cc/Conferences/2016/Schedule?type=Poster), [2015](https://nips.cc/Conferences/2015/Schedule?type=Poster), [2014](https://nips.cc/Conferences/2014/Schedule?type=Poster))
- ICML ([2017](https://2017.icml.cc/Conferences/2017/Schedule?type=Poster), [2016](http://icml.cc/2016/?page_id=1649), [2015]())
- CVPR ([2017](http://openaccess.thecvf.com/CVPR2017.py), [2016](http://www.cv-foundation.org/openaccess/CVPR2016.py))
- ICLR ([2017](https://openreview.net/group?id=ICLR.cc/2017/conference))
- ECCV ([2016](http://www.eccv2016.org/main-conference/))
- ICCV ([2017](http://openaccess.thecvf.com/ICCV2017.py), [2015](http://pamitc.org/iccv15/program.php))

