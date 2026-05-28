# Peco (Version 1.0) Guidebook

> Author: GHz
>
> Date: 2023/9/9



[TOC]

------





## 0 Why Peco

Peco is a training-free method to denoise the ARPES data. At present version, it can only process 2D+1D data. However, owning to its convenience of training without dataset, it can extend to 3D+1D data more easily than Deno which needs very large training dataset that consist of ARToF data. 

As a joke, the name Peco actually implies pickle which is not fresh vegetable anymore and also means a little bit of picking the data. It is because Peco mapping a white noise data to the clean data, which seems like a trick fabricating the data. This should be seriously considered. More Peco's mechanism will be illustrated in detail in the part of `How Peco Work`.  





## 1 Network

Peco use U-Net to denoise data.

![](.\assets\hor_netForwatch.onnx.svg)

As a very strong and powerful neural networks, U-Net is mainly compose of two parts, the down sampling encoder and the up sampling decoder, with some connections to keep the information volumes unchanged when executing down sampling.





## 2 How Peco Work

As it has been mentioned above, Peco mapping a fixed uniform noise to the clean data. For most of neural networks, they perform better imitating the ordered data than the noise. The ordered information is easy to fit but the disorganized information is hard to fit. But if training the neural network too much epochs, it  also can imitate the noise. This is why Peco doesn't need a training dataset, however, whose results strongly rely on the training epochs and learning rate. Generally, a well trained network should be irrelevant to these two, but what Peco does is just catch the best epoch that the network trained best. It is the core for Peco to capture the time window which is before Peco imitating the noise well and after Peco imitating clean data well.   

![image-20230909232447463](.\assets\image-20230909232447463.png)





## 3 Pannel

![image-20230909150351250](.\assets\image-20230909150351250.png)

- `Choose style`: Change the theme of Peco;
- `Data Folder`: Select your work place (different from Deno, Peco doesn't have the result fold);
- `Total iterations`: Fill in with a integer number confining the upper limit of the training times. More training iterations, more risk of overfitting;
- `Save(Plot) every`: Better to fill a integer number that can divide `Total iterations`. Every `Save(Plot) every`, a figure to check will pop up optionally and a data file named`<name>_<iterations>_<deno>.itx` will be saved definitely;
- `Process data format`: The Process data use the image format, and the author recommend `tiff` format out of its FLOAT accuracy. Bad accuracy(CHAR) for png but easier to train;
- `Plot process`: Get a teamwork with `Save(Plot) every`, optionally draw the process results;
- `Learning Rate`: Learning rate affects the smoothness of the denoised data. Adjusting this parameter is a tricky and challenged work, Careful! Generally, less learning rate bring more rough data and bad for denoising;
-  `Optimizer choice`: Altough LBFGS is a very great and advanced optimization algorithm, but here LBFGS often collapse thus not recommended;
- `Denoise data`, `Check results`, `Clean up`, `Clear all`: These four buttons are the same with those of Deno.



<u>**NOTICE: Peco only support .itx format in version 1.0!**</u>





## 4 Performance

### 4.1 Learning Rate set to 0.01

See default learning rate performance here.

**<u>0 iteration</u>**

![2400iterations](.\assets\0iteration.png)

**<u>800 iterations</u>**

![2400iterations](.\assets\800iterations.png)

**<u>1600 iterations</u>**

![2400iterations](.\assets\1600iterations.png)

**<u>2400 iterations</u>**

![2400iterations](.\assets\2400iterations.png)

### 4.2 Learning Rate set to 0.005

If set a smaller learning rate, the denoised data will be sharper with more details.

**<u>0 iteration</u>**

![0_2400iter](.\assets\0_0iter.png)

**<u>800 iterations</u>**

![0_2400iter](.\assets\0_800iter.png)

**<u>1600 iterations</u>**

![0_2400iter](.\assets\0_1600iter.png)

**<u>2400 iterations</u>**

![0_2400iter](.\assets\0_2400iter.png)

### 4.3 Imitate the noise (LR=0.01)

If it is hard for Peco to imitate the noise? Check it out here. 

**<u>0 iteration</u>**

![0](.\assets\0.png)

**<u>4800 iterations</u>**

![0](.\assets\4800.png)

**<u>9600 iterations</u>**

![0](.\assets\9600.png)

**<u>10400 iterations</u>**

![0](.\assets\10400.png)

### 4.4 Other data (LR=0.005)

Looking for more performances on the other data? Here it is. 

<u>**sample 1**</u>

![1_2400iter](.\assets\1_2400iter.png)

<u>**sample 2**</u>

![1_2400iter](.\assets\2_2400iter.png)

<u>**sample 3**</u>

![1_2400iter](.\assets\3_2400iter.png)

<u>**sample 4**</u>

![1_2400iter](.\assets\4_2400iter.png)
