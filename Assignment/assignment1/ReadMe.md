#k-Nearest Neighbor (kNN) exercise

在本作业中，将基于k-最近邻（k-Nearest Neighbor）或者SVM/Softmax分类器实践一个简单的图像分类流程。本作业的目标如下：

- 理解基本的**图像分类流程**和数据驱动方法（训练与预测阶段）。
- 理解训练、验证、测试分块，学会使用验证数据来进行**超参数调优**。
- 熟悉使用numpy来编写向量化代码。
- 实现并应用k-最近邻（**k-NN**）分类器。
- 实现并应用支持向量机（**SVM**）分类器。
- 实现并应用**Softmax**分类器。
- 实现并应用一个**两层神经网络**分类器。
- 理解以上分类器的差异和权衡之处。
- 基本理解使用**更高层次表达**相较于使用原始图像像素对算法性能的提升（例如：色彩直方图和梯度直方图HOG）。

## Setup

在[这](http://cs231n.github.io/assignments/2018/spring1718_assignment1.zip)获取代码压缩包。

安装[教程](http://cs231n.github.io/setup-instructions/)。

### Download data:

无论选择哪种方式开始执行代码前你都需要下载CIFAR-10数据集。在`assignment1`目录下执行如下命令：

```
cd cs231n/datasets
./get_datasets.sh
```

### Start IPython:

在下载CIFAR-10数据之后，在`assignment1`目录使用`jupyter notebook`启动IPython notebook服务器。

如果对IPython不熟悉，你可以参考 [IPython教程](http://cs231n.github.io/ipython-tutorial)。

### Some Notes

**注意1:** `assignment1`的代码在`Python3.6`下测试通过。设置虚拟Python环境的时候记得设置正确的版本号。确认python版本的方式（1）启动virtulenv（2）运行`which python`

**注意2:**如何你的运行环境是OSX，那matplotlib有可能会发生错误[issues described here](http://matplotlib.org/faq/virtualenv_faq.html)。

### Q1: k-Nearest Neighbor classifier (20 points)

IPython Notebook **knn.ipynb**文件会指导你实现一个kNN分类器。 

kNNclassifier包含两个阶段：

- 在训练阶段，分类器训练数据只是简单的把数据存储下来。
- 在测试阶段，kNN分类器会将测试图像和所有的训练数据进行比较，最终将最标签设置为最接近的训练样本。
- k值是通过交叉验证得到的。

在本练习中，您将理解基本的图像分类流程，交叉验证以及熟练编写高效的矢量化代码。

从初始化环境

```python
# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from __future__ import print_function

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
```

从本地加载数据

```python
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

Training data shape:  (50000, 32, 32, 3)
Training labels shape:  (50000,)
Test data shape:  (10000, 32, 32, 3)
Test labels shape:  (10000,)

查看部分图像数据

```python
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):		// 函数可以把一个list变成索引-元素对
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)		// 随机选择
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
```

转换数据结构

```python
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
```

(5000, 3072) (500, 3072)

训练数据

```python
from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
```

#### 作业

计算测试样本和训练样本之间的距离

```python
 def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists
```







### Q2: Training a Support Vector Machine (25 points)

The IPython Notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier (20 points)

The IPython Notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Two-Layer Neural Network (25 points)

The IPython Notebook **two_layer_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features (10 points)

The IPython Notebook **features.ipynb** will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

### Submitting your work

There are **two** steps to submitting your assignment:

**1.** Submit a pdf of the completed iPython notebooks to [Gradescope](https://gradescope.com/courses/17367). If you are enrolled in the course, then you should have already been automatically added to the course on Gradescope.

To produce a pdf of your work, you can first convert each of the .ipynb files to HTML. To do this, simply run

```
ipython nbconvert --to html FILE.ipynb
```

for each of the notebooks, where `FILE.ipynb` is the notebook you want to convert. Then you can convert the HTML files to PDFs with your favorite web browser, and then concatenate them all together in your favorite PDF viewer/editor. Submit this final PDF on Gradescope, and be sure to tag the questions correctly!

**Important:** *Please make sure that the submitted notebooks have been run and the cell outputs are visible.*

**2.** Submit a zip file of your assignment on AFS. To do this, run the provided `collectSubmission.sh` script, which will produce a file called `assignment1.zip`. You will then need to SCP this file over to Stanford AFS using the following command (entering your Stanford password if requested):

```
# Run from the assignment directory where the zip file is located
scp assignment1.zip YOUR_SUNET@myth.stanford.edu:~/DEST_PATH
```

`YOUR_SUNET` should be replaced with your SUNetID (e.g. `jdoe`), and `DEST_PATH` should be a path to an existing directory on AFS where you want the zip file to be copied to (you may want to create a CS231N directory for convenience). Once this is done, run the following:

```
# SSH into the Stanford Myth machines 
ssh YOUR_SUNET@myth.stanford.edu

# Descend into the directory where the zip file is now located
cd DEST_PATH

# Run the script to actually submit the assignment
/afs/ir/class/cs231n/submit
```

Once you run the submit script, simply follow the on-screen prompts to finish submitting the assignment on AFS. If successful, you should see a “SUBMIT SUCCESS” message output by the script.

- [ cs231n](https://github.com/cs231n)
- [ cs231n](https://twitter.com/cs231n)
- [karpathy@cs.stanford.edu](mailto:karpathy@cs.stanford.edu)