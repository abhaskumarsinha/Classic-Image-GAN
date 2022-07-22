# Classic-Image-GAN
A classic GAN that generates Images: A study on problems like Mode Collapse.

## Dataset
Only a sample hundred data has been provided, which isn't enough to actually generate GAN Images. So, kindly full flower dataset and place it in ./flowers directory from : http://chaladze.com/l5/ (Linnaeus 5 dataset Project).

## Abstract
The project demonstrates a classic GAN (Generative adversarial network) made with only classic simple Dense Neurons, connected end to end in from of layers with simple, relu/tanh/sigmoid/linear activation functions. The project has a sample output included : sample-output-viewer Notebook file which has been used on Google Colab after hours of GPU Training. Analysis over outputs demonstrates simple issues with such Classical GANs, called Mode Collapse. Mode Collapse refers to the phenomenon where Generator Network of GANs converge to a single output or a special case vector, irrespective of input vectors fed to them. A good remedy to tackle such problem is demonstrated in DC-GAN Repo where, we replace some Dense layers to Convolutional Transpose Layers or Convolutional Layers. We are training out GAN with simple Adam Optimizer for Deep Nets.
