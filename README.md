# Classic-Image-GAN
A classic GAN that generates Images: A study on problems like Mode Collapse.

## Dataset
Only a sample hundred data has been provided, which isn't enough to actually generate GAN Images. So, kindly full flower dataset and place it in `./flowers` directory from : http://chaladze.com/l5/ (Linnaeus 5 dataset Project).

## Abstract
The project demonstrates a classic GAN (Generative adversarial network) made with only classic simple Dense Neurons, connected end to end in from of layers with simple, relu/tanh/sigmoid/linear activation functions. The project has a sample output included : sample-output-viewer Notebook file which has been used on Google Colab after hours of GPU Training. Analysis over outputs demonstrates simple issues with such Classical GANs, called Mode Collapse. Mode Collapse refers to the phenomenon where Generator Network of GANs converge to a single output or a special case vector, irrespective of input vectors fed to them. A good remedy to tackle such problem is demonstrated in DC-GAN Repo where, we replace some Dense layers to Convolutional Transpose Layers or Convolutional Layers. We are training out GAN with simple Adam Optimizer for Deep Nets.

## Results

**After 5 epochs**

![1](https://github.com/abhaskumarsinha/Classic-Image-GAN/raw/main/sample-outputs/1.png)

**After 5000 epochs**

![2](https://github.com/abhaskumarsinha/Classic-Image-GAN/raw/main/sample-outputs/2.png)

**After 10000 epochs**

![3](https://github.com/abhaskumarsinha/Classic-Image-GAN/raw/main/sample-outputs/3.png)

As we can see the phenomenon of *Model collapse* in some of the pixels where some of the pixels are dark black/Green/Blue/Red/White. It often happens because its easier for the *tanh* function to converge to end -1, +1 values more easily than the *sigmoid* activation function. The solution to Mode Collapse has been discussed better in DC-GAN Repo. The model fails to output multiple different outputs, instead, keeps rotating the noise on a single case.

# Biblography

- I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems, 2014, pp. 2672–2680.

- A. Radford, L. Metz, and S. Chintala, “Unsupervised representation learning with deep convolutional generative adversarial networks,” in Proceedings of the 5th International Conference on Learning Representations (ICLR) - workshop track, 2016.

- Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015.

- I. J. Goodfellow, “On distinguishability criteria for estimating generative models,” International Conference on Learning Representations - workshop track, 2015

- I. Goodfellow, “Nips 2016 tutorial: Generative adversarial networks,” 2016, presented at the Neural Information Processing Systems Conference. [Online]. Available: https://arxiv.org/abs/1701.00160

- J. D. Lee, M. Simchowitz, M. I. Jordan, and B. Recht, “Gradient descent only converges to minimizers,” in Conference on Learning Theory, 2016, pp. 1246–1257

- Thanh-Tung, Hoang, and Truyen Tran. "Catastrophic forgetting and mode collapse in gans." 2020 international joint conference on neural networks (ijcnn). IEEE, 2020.

- Kushwaha, Vandana, and G. C. Nandi. "Study of prevention of mode collapse in generative adversarial network (GAN)." 2020 IEEE 4th Conference on Information & Communication Technology (CICT). IEEE, 2020.

