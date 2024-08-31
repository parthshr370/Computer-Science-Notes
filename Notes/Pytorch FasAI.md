 

1. Tensor broadcasting is the process of performing operations between a tensor and a smaller tensor .
2. You can broadcast two tensors if - 
     - Two dimensions are equal
     - One of the dimension is 1

3. Addition of [2,2] and [3,3] will not work but anything with a 1 will because 1 can be expanded to cover other dimensions.
4. But we can add [1,3] tensor to [3,3] without any trouble 
5. Better than manually expanding the tensor each time 
6. Pytorch needs an abstraction of defining any type of dataset - 
```python 
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

7. We need a way to determine what is a cat and what is a fish. Thats easy enough for us but is harder for computers to determine . We use labels attached to data and train . This is called supervised learning.
8. Image-net already has too much information for us to use.
9. We use this dataloader type of labelling things - 
   - A function that implements the size of our dataset using `(len)`
   - A method that can retrieve an item from out dataset in a `(label,tensor)` pair. This pushes the data in  the neural network
   - The body for `getitem` that can take an image and transform it into a tensor and return that and the label back so Pytorch can operate on it.

10. The **torchvision** package has a class called `ImageFolder` that does pretty much everything for us, providing our images in a structure where each directory is labelled.
```python
import torchvision 
from torchvision import transforms

train_data_path = "./train/"


transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=train_data_path,tranform = transforms)
```

11. `torchvision.transforms` allows us to convert a normal image into pytorch tensors so that the NN can read it . It also makes them of the same size for NN to interpret easily.
12. For GPUs to perform calculations at a efficient manner we scale the images to a common resolution via the `Resize(64)` transform . We then convert the images to a tensor and finally normalise the tensor around a specific set of mean and standard deviations.
13. Normalising is important because a lot of matrix multiplication happens in the input layer and prevents the value from getting too large by keeping them between 0 and 1 ( prevents exploding gradient problem )
14. We use the standard deviation and mean values for the dataset.
15. Overfitting is a common problem in deep learning models and our model gets accustomed to one type of image and we need a network to have a validation set to verify at the end of each epoch.
16. Code to create a validation set 
```python 
val_data_path = "./val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,
                                            transform=transforms)
```

17. Along with a validation set we also create a test set 
```python 
test_data_path = "./test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                             transform=transforms)
```
\
18. We often complete our dataloader with a few more lines of python 
```python 
batch_size = 64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader  = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader  = data.DataLoader(test_data, batch_size=batch_size)
```

19. Batch size tells us how many images it will go through the network before we train and update it 
20. Smaller batch size or miny batches help us to make training faster 

![nodes](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0203.png)

21. Every node is connected to another node in next layer 
22. Each connection has a weight that determines the strength of the signal from that node going into the next layer ( It is the weights that get updated when training a NN )
23. We can find the input of the system by performing the matrix multiplication of the weights and biases

## Creating a Network 

24. We inherit from a class called `torch.nn.Network` and fill out the `__init__` and `__forward__` method
```python 
class SimpleNet(nn.Module):

def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(12288, 84)
    self.fc2 = nn.Linear(84, 50)
    self.fc3 = nn.Linear(50,2)

def forward(self):
    x = x.view(-1, 12288)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.fc3(x))
    return x

simplenet = SimpleNet()
```

25. The `init()` method calls 3 fully connected layers which is called `Linear` in pytorch and `Dense` in keras 
26. The `forward()` method tells how data flows through the network 
27. First we need to convert the 3-D tensor in an image with three channels red green blue into 1D tensor so that it can be fed into the first `Linear` Layer 
28. We do that using the `view()` command 
29. Finally the `softmax()` function gives us the output probalities
30. If a layer is going to, say, **50** inputs to **100** outputs, then the network might _learn_ by simply passing the 50 connections to **50** of the **100** outputs and consider its job done. By reducing the size of the output with respect to the input, we force that part of the network to learn a representation of the original input with fewer resources, which hopefully means that it extracts some features of the images that are important to the problem we’re trying to solve; for example, learning to spot a fin or a tail.
31. We meed a loss function in the code to find just how right or wrong each label is 
32. Cross entropy tells you about the entropy or the uncertaininty of the information 
33. The closer the value is to 1 the better the probablity and the cross entropy of the function 
34. Cross entropy loss is great for image classification because - It measures the difference between the probability distribution of the model's predictions and the actual distribution of the correct class. 
35. It is great for multi class problem where we might need to classify multiple things like dogs cats birds 
36. We need `softmax()` function in the cross entropy loss and it is incorporated in the `forward()` function 
```python 
def forward(self):
    # Convert to 1D vector
    x = x.view(-1, 12288)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

37. Optimiser helps you find the lowest point int he loss function so that you can use that to find the least error value 
38. Adam uses stochastic gradient descent with momentum and RMSProp
39. Stochastic gradient descent takes the average of the region past gradient to find the best updating parameters 
40. RMSProp optimizer tackles the problem of constantly changing learning rates for parameters with fluctuating gradients by using an exponentially decaying average of squared gradients.
41. We might get trapped in the point of local minima which looks like the shallowest part of the gradient but are not 
42. Adam uses exponentially decaying list of gradients and square of those gradient and uses those to scale the global learning rate that adam is working with 
43. For using adam - 
```python 
import torch.optim as optim 
optimiser = optim.Adam(simplenet.parameters(),lr = 0.001)
```

44. The final step is to create a training loop - 
```python
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input, target = batch
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

45. We compute the gradient of the model by calling the `backward()` function 
46. The `optimiser.step()` method uses those gradients afterwards to perform the adjustment of weights 
47. The `zero_grad()` method resets the gradients at the end of each loop 
48. Since the gradients accumulate at each iteration it is important to reset them 
49. In more technical terms, the gradient is the vector of partial derivatives of the error function with respect to each of the network's parameters.  These partial derivatives tell you how much each parameter contributes to the overall error.
50. To make use of GPU more we need to check if there is cuda on our machine . 
```python 
if torch.cuda.is_available():
        device = torch.device("cuda")
else
    device = torch.device("cpu")

model.to(device)
```

51. We need to have a training method that takes in a model that will be re used in the rest of the codes - 
```python 
def train(model, optimizer, loss_fn, train_loader, val_loader,
epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
            target = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_iterator)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
							   target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(valid_iterator)

        print('Epoch: {}, Training Loss: {:.2f},
        Validation Loss: {:.2f},
        accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
```

52. This is our training method and now we just need to intitate this to kick off our model with required parameters
```python 
train(simplenet, optimizer, torch.nn.CrossEntropyLoss(),
      train_data_loader, test_data_loader,device)
```

53. Now we need to use this NN to perform classification . This python code will load an image from the filesystem and print out whether our network says cat or fish 
```python
from PIL import Image

labels = ['cat','fish']

img = Image.open(FILENAME)
img = transforms(img)
img = img.unsqueeze(0)

prediction = simplenet(img)
prediction = prediction.argmax()
print(labels[prediction])
```

54. However, because our network uses batches, it actually expects a 4D tensor, with the first dimension denoting the different images within a batch. We don’t have a batch, but we can create a batch of length 1 by using `unsqueeze(0)`, which adds a new dimension at the front of our tensor.
55. Save the current state of a model in Python’s _pickle_ format by using the `torch.save()` method. Conversely, you can load a previously saved iteration of a model by using the `torch.load()` method.
56. These functions save the parameter and structure of the model in a file. This might be a problem when you might later want to edit the model . It is a common practice to save a models `state_dict` instead . This is a standard python dict that contains the map of each layers parameters in the model 
```python 
torch.save(model.state_dict(), PATH)
```
57. To restore, create an instance of the model first and then use `load_state_dict`. For `SimpleNet`: 
```python 
simplenet = SimpleNet()
simplenet_state_dict = torch.load("/tmp/simplenet")
simplenet.load_state_dict(simplenet_state_dict)
```

58. The benefit here is that if you extend the model in some fashion, you can supply a `strict=False` parameter to `load_state_dict` that assigns parameters to layers in the model that do exist in the `state_dict`, but does not fail if the loaded `state_dict` has layers missing or added from the model’s current structure. Because it’s just a normal Python `dict`, you can change the key names to fit your model, which can be handy if you are pulling in parameters from a completely different model altogether.

\

## Convolutional Neural Network

59. The basic code for a CNN is - 
```python
class CNNNet(nn.Module):

    def __init__(self, num_classes=2):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x
```

60. We use `nn.Sequential()` to create a chain of layers.When we use this in the `forward()` the input goes through each element of the array of layers in succession. You can use this to break the model into more logical arrangements.
61. The `conv2d` layer is a 2D convolution layer. The pixels are x pixels wide and y pixel high, Where each value indicates whether it is black or white.
62. We introduce something called a filter that is a smaller matrix which we drag across our matrix 
63. To produce our output we take our small filter and pass it over the original input from top left to bottom right 
64. We just multiply each element of the matrix to the corresponding element in the matrix 
65. We do that again on the output matrix and create a feature map 

![featuremap](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0301.png)


66. you can see how this works graphically, with a 3 × 3 kernel being dragged across a 4 × 4 tensor and producing a 2 × 2 output (though each segment is based on nine elements instead of the four in our first example).
67. All the filter share the same value of bias 
68. To invoke the convolutional layer - 
```python 
nn.Conv2d(in_channels,out_channels, kernel_size, stride, padding)
```

69. The `in_channels` is the number of input channels we will be recieving in the layer. When the image is RGB the number of inputs is 3 
70. The `out_channels` represent the number of output channels in out conv layer 
71. The `kernel_size` defines the height and width of our filter 
72. The `strive` attribute tells us about how many steps we move while adjusting the filter to a new position. When we take 2 strides it makes the output tensor the size half of it .
73. Sometimes the filter we move with a certain stride has no place to move or shift after reaching the end point. This is where **padding** comes into place 
74. If we do not pad then the last columns of input is simply thrown away 
75. We have a pooling layer in a CNN that lowers the dimension of the of previous input layer and helps and gives us fewer parameters. 
76. `MaxPool` takes the minimum value from each of these tensors giving us an output tensor of lower dimension. There is also a padding operation in `MaxPooling` that creates a border of zero values around the tensor in case the stride goes outside the tensor window.
77. We can also pool the values instead of just taking the maximum values in the case.
78. To prevent the NN to overfit we have something called a droput layer. We feed random data at the NN which would help us to generalise more and more data 
79. By default the `Dropout` layer in our example CNN network aer intialized with 0.5 meaning 50 percent of the values are randomly zerod out. If we want to change that value then - `Dropout(p=0.2)` 
80. Dropout only occurs during the training time because if we try to use it during testing you would lose a chunk of networks reasoning power.
81. Alexnet is one of the first major breakthrough in CNN. It was the first to introduce `maxpool` and `dropout` 
82. The **Inception network** instead runs a series of convolutions of different sizes all on the same input, and concatenates all of the filters together to pass on to the next layer. Before it does any of those, though, it does a 1 × 1 convolution as a _bottleneck_ that compresses the input tensor, meaning that the 3 × 3 and 5 × 5 kernels operate on a fewer number of filters than they would if the 1 × 1 convolution wasn’t present
![inception](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0302.png)


83. Then we had Resnet. Its key improvement is allowing information to flow directly through the network which was not possible in VGG or inception. This also solved the problem of vanishing gradient problem 
![resnet](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0304.png)

84. If we want to use these pretrained models we just need to import them from the torchvision library 
```python 
import torchvision.models as models
alexnet = models.alexnet(num_classes=2)
```
85. We can also use the pre-trained model with the necessary weights and biases by just - 
```python 
models.alexnet(pretrained=True)
```

86. Deep neural networks can be difficult to train for a couple of reasons. One issue is that the distribution of the data flowing through the network can change drastically between layers. This is known as internal covariate shift. As a result, subsequent layers have to constantly adapt to these changes, slowing down the learning process.
87. Batch normalisation Tackles this problem by standardizing the inputs to each layer in a network. This is done by subtracting the mean value of a mini batch and then dividing by the standard deviation.
88. This forces the data to have a constant mean and variance 
89. `BatchNorm` might not be extremely useful for smaller layers but as they get larger the effect on any layer say 20 can be vast because due to repeated matrix multiplication you either will have vanishing gradient or exploding


## Transfer Learning 

90. Our pretrained ResNet model already has a bunch of information encoded into it for image recognition and classification needs, so why bother attempting to retrain it? Instead, we fine-tune the network. We alter the architecture slightly to include a new network block at the end, replacing the standard 1,000-category linear layers that normally perform ImageNet classification.
91. We then _freeze_ all the existing ResNet layers, and when we train, we update only the parameters in our new layers, but still take the activations from our frozen layers. This allows us to quickly train our new layers while preserving the information that the pretrained layers already contain
```python
from torchvision import models
transfer_model = models.ResNet50(pretrained=True)
```
92. We now need to freeze the layers and that is by stopping the accumulated gradients - using `requires_grad()` we need to do this for each parameter but we rather use `parameters()` function 
```python
for name,param in transfer_model.named_parameters():
	param.requires_grad = False
```

93. We should also stop the `BatchNorm` parameter as is will eventually train the data on the standard deviation of the dataset that originially it was trained on and not on the actual dataset you want to fine-tune on 
94. Some of the _signal_ from your data may end up being lost as `BatchNorm` _corrects_ your input. You can look at the model structure and freeze only layers that aren’t `BatchNorm` like this:
```python 
for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
```

95. We then need to replace the final classification block with a new one that we will train for detecting cats or dogs. We replace it with couple of `Linear` or `Relu`  or `Dropout` But we can also have an extra CNN layer too 
96. The definition of Pytorch's implementation of the ResNet architecture stores the final classifier block in an instance variable `fc` all we need to replace is the new structure ( either using `fc` or `classifier` ) 

```python 
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
nn.ReLU(),
nn.Dropout(), nn.Linear(500,2))
```

97. Here we take advantage of `in_features` variable allows us to control the number of activation coming into a layer . We can also use `out_features` to check out all the activation coming out.
98. We then just go to our training loop and see large jumps in our accuracy even within a few epochs 
99. Learning rate is one of the most important parameters we can change 
100. We need to find the perfect learning rate that can be done by grid search where it exaustively searches through each pair validation dataset which is extremely time consuming ( still done by a lot of people )
101. a learning rate value that has empirically been observed to work with the Adam optimizer is 3e-4. This is known as Karpathy’s constant, after Andrej Karpathy (currently director of AI at Tesla)
102. One of the better learning rates finding method is - Over the course of an epoch , start out with a small learning rate and increase to a higher learning rate over each mini batch resulting a high rate at the end of each epoch. Calculate the loss for each rate and then look at the plot to find the greatest decline in the plot. This is roughly the point where the gradient descent is the steepest.

![lr](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0401.png)

103. Try looking at the bottom of the curve where you might find the value you are looking for 
104. Code for the same - 
```python 
import math
def find_lr(model, loss_fn, optimizer, init_value=1e-8, final_value=10.0):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, labels = data
        inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        # Do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]
```

105. We first slice the `lr` learning rate and losses. We do that because first bits of training and the last few tend not to tell much information \
106. We can even do better with differential learning rates 
107. We might also want to add some changes not just to the ending last classifiers in the NN but rather in between fine-tuning- 
```python 
optimizer = optimizer.Adam([
{ 'params': transfer_model.layer4.parameters(), 'lr': found_lr /3},
{ 'params': transfer_model.layer3.parameters(), 'lr': found_lr /9},
], lr=found_lr)
```

108. This sets the learning rate for `layer4` to a third of the founding learning rate and ninth for `layer3` 
109. We might also wanna un freeze the pre trained layer by giving them additional different learning rate
```python 
unfreeze_layers = [transfer_model.layer3, transfer_model.layer4]
for layer in unfreeze_layers:
    for param in layer.parameters():
        param.requires_grad = True
```

110. Now the parameters may take gradient once more 
111. Data augmentation - The NN learns different ways to see image and still able to classify them 
![cat](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0402.png)
Fig - Cat image in normal orientation

![Cat reversed](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0403.png)
Fig - Still a cat but inverted

112. Code utilising `torchivion.transforms` to transform the image 
```python 
transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225] )
        ])
```

113. `torchvision` comes up with large collection of `transforms` functions that can be used for data augmentation but we will look at some of the most useful data augmentation techniques
```python 
torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```
114. `ColorJitter` randomly changes the contrast , brightness , saturation of the image we can either supply float or tuple of floats with non negative range of 0 to 1 
115. If we want to flip an image , these two transforms randomly flip the image - 
```python 
torchvision.transforms.RandomHorizontalFlip(p=0.5)
torchvision.transforms.RandomVerticalFlip(p=0.5) # p here is probality , here we give random chances with p=0.5
```

116. `RandomGrayscale` is a similar type of transformation, except that it randomly turns the image grayscale, depending on the parameter _p_ (the default is 10%):
```python 
torchvision.transforms.RandomGrayscale(p=0.1)
```

117. `RandomCrop` and `RandomResizeCrop`, as you might expect, perform random crops on the image of `size`, which can either be an int for height and width, or a tuple containing different heights and width
```python 
torchvision.transforms.RandomCrop(size, padding=None,
pad_if_needed=False, fill=0, padding_mode='constant')
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0),
ratio=(0.75, 1.3333333333333333), interpolation=2)
```

118. We need to be a bit careful because if you crop the image too small then there is risk of cutting out important parts of images 
119. If you’d like to randomly rotate an image, RandomRotation will vary between [-degrees, degrees] if degrees is a single float or int, or (min,max) if it is a tuple:
```python 
torchvision.transforms.RandomRotation(degrees, resample=False,expand=False, center=None)
```

120. `Pad` is a general-purpose padding transform that adds padding (extra height and width) onto the borders of an image:
```python 
torchvision.transforms.Pad(padding, fill=0, padding_mode=constant)
```
121. A single value in `padding` will apply padding on that length in all directions. A two-tuple `padding` will produce padding in the length of (left/right, top/bottom), and a four-tuple will produce padding for (left, top, right, bottom). By default, padding is set to `constant` mode, which copies the value of `fill` into the padding slots
122. The other choices are `edge`, which pads the last values of the edge of the image into the padding length; `reflect`, which reflects the values of the image (except the edge) into the border; and `symmetric`, which is `reflection` but includes the last value of the image at the edge
123. `RandomAffine` allows you to specify random affine translations of the image (scaling, rotations, translations, and/or shearing, or any combination). 
![cat](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0409.png)

124. HSV is a another alternative to RGB where we have 8 bit values with hue, saturation and value 
125. It might capture some information better than RGB coding. A mountain may be mountain but the tensor that gets formed on each space representation will be different and one space may capture something about you data better than the other.
126. We can use Image.convert() to translate a PIL image from one color space to another. We could write a custom transform class to carry out this conversion, but PyTorch adds a transforms.Lambda class so that we can easily wrap any function and make it available to the transform pipeline. Here’s our custom function:
```python 
def _random_color_space(x):
	output = x.convert("HSV")
	return output
```

127. We can then use the `transforms.Lambda` class and used any standard transformation pipeline like we have seen before
```python 
colour_transform = transforms.Lambda(lambda x: _random_color_space(x))
```

128. We can change the images in different color space depending on the epoch for better information from the same image 
```python 
random_color_transform = torchvision.transform.RandomApply([colour_transform])
```

129. By default `RandomApply` has the value 0.5 
130. Some times simple lambda isn't enough we have some initialisation or state that we want to keep track of 
131. Such a class has to implement two methods - 
     - `__call__` which the transform pipeline will initiate during the transformation process
     - `__repr__` which should return a string representation of the transform, along with any state that may be useful for diagnostic purposes 

132. The Code - 
```python 
class Noise():

	def __init__(self,mean,stddev):
		self.mean = mean
		self.stddev = stddev
	def __call__(self,tensor):
		noise = torch.zeros_like(tensor).normal_(self.mean,self.stddev)
		return tensor.add_(noise)
	
	def __repr__(self):
		repr = f"{self.__class__.__name__  }(mean={self.mean},
        stddev={self.stddev})"
        return repr
```

133. If we add this to the pipeline we can see the results of the `__repr__` 
```python 
transforms.Compose([Noise(0.1,0.05)])
```

134. Tip seems odd but obtain real results - start smaller and get bigger. What means is if we are traiing at 256 x 256 image we can create a few more datasets that have scaled to 64 x 64 and 128 x 128 . Create a model , fine tune it with the smaller resolution images and then once you squeeze from smaller model then keep the same parameters and then scale up again and again.You’ll probably find a percentage point or two improvement in accuracy.
135. This happens because by training the lower resolutions , the model learns about the overall structure of the image and can refine that knowledge as the incoming images expand.
136. If we don't want to have multiple copies of the same dataset hanging around in the storage then we can use the `torchvision` transforms method to do this on the fly using `resize`.
```python 
resize = transforms.Compose([transforms.Resize(64)
 …_other augmentation transforms_…
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

137. The penalty here is that we tend to spend more time training as pytorch has to apply the resize function every time. If you resize all the images before hand then we can get a quicker training run 
138. The concept of starting small and then getting bigger also applies to architectures. Using a ResNet architecture like ResNet-18 or ResNet-34 to test out approaches to transforms and get a feel for how training is working provides a much tighter feedback loop than if you start out using a ResNet-101 or ResNet-152 model
139. There are plenty of approaches to ensembles, and we won’t go into all of them here. Instead, here’s a simple way of getting started with ensembles, one that has eeked out another 1% of accuracy in my experience; simply average the predictions:
```python 
# Assuming you have a list of models in models, and input is your input tensor

predictions = [m[i].fit(input) for i in models]
avg_prediction = torch.stack(b).mean(0).argmax()
```

140. The `stack` method concatenates the array of tensors together, so if we were working on the cat/fish problem and had four models in our ensemble, we’d end up with a 4 × 2 tensor constructed from the four 1 × 2 tensors
141. The mean takes the mean as expected but we have to pass in a dimension of 0 to ensure that it takes an average across the first dimension instead of simply adding up all the tensor elements 

## Pytorch Lightning

PyTorch Lightning is a lightweight PyTorch wrapper that aims to organize PyTorch code and reduce boilerplate. It provides a structured way to build and train neural networks while maintaining the flexibility of PyTorch


Skeleton of Basic Lightning Code - 
```python 
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Data
train_loader = DataLoader(...)
val_loader = DataLoader(...)

# Model
model = MyLightningModule()

# Trainer
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)
```

## Text Classification

142. Original Google translate code was of 500000 lines but the new one that compensates transformers is of just 500.
143. CNN has no concept of time  and one thing that is important while dealing with data is temporal domain.
144. RNN give this memory and temporal concept with the help of something called the `hidden state`

![RNN](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0501.png)



145. This is like a for loop with hidden output with step of `t` and hidden output of state `ht`
![full](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0502.png)


146. Here we have a group of fully connected layer with shared parameters. Series of inputs and outputs. 
147. The input data is fed into the network and the next item in the sequence is predicted as output 
148. `Relu` is inserted in between layers to introduce non linearity 
149. The backprop goes like on individual networks and summing all the gradients together 

![LSTM](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0503.png)



150. **LSTM** on the other hand is something with a forget gate. The parameters decide how much percent to forget 
151. The cell ends up being the memory of the network layer and input and output and forget gates will determine how data flows through the layer 

![GRU](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0504.png)


152. **GRU** is like a modified version of the LSTM where the forget gate directly goes into the output so the parameter count is even lesser than LSTM 

![BISTM](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492045342/files/assets/ppdl_0505.png)

153. **Bi-Directional LSTM** . Both RNN and LSTM Helps you work with the past data but sometimes you also need to see the future as well. Sometimes used in cases such as translation and handwriting recognition 
154. BiLSTM solves this problem by stacking 2 LSTMs in opposite direction one in forward and other in backward.
155. You can create BiLSTM in pytorch with the help of - `bidirectional=True` parameter while creating the `LSTM()`
156. The simplest approach is still one that you’ll find in many approaches to NLP, and it’s called _one-hot encoding_. It’s pretty simple! Let’s look at our first sentence from the start of the chapter:

```
The cat sat on the mat.
```

157. If we consider that this is the entire vocabulary of our world, we have a tensor of `[the, cat, sat, on, mat]`. One-hot encoding simply means that we create a vector that is the size of the vocabulary, and for each word in it, we allocate a vector with one parameter set to 1 and the rest to 0:

```python
the — [1 0 0 0 0]
cat — [0 1 0 0 0]
sat — [0 0 1 0 0]
on  — [0 0 0 1 0]
mat — [0 0 0 0 1]
```

158. We’ve now converted the words into vectors, and we can feed them into our network. Additionally, we may add extra symbols into our vocabulary, such as `UNK` (unknown, for words not in the vocabulary) and `START/STOP` to signify the beginning and ends of sentences.
159. One hot encoding requres some modification to work on 
     - When we work on a realistic set of words our vectors are going to be very long with no set of information 
     - We need to consider words with strong association like `kitty` and `cat` and its impossible to represent with one hot encoding 

160. One popular approach is to replace one hot encoding with `embedding matrix` . One hot encoding is an embedding matrix itself but without any information and relation
161. We squash the dimensionality of the matrix to something small 
162. For example, if we have an embedding in a 2D space, perhaps _cat_ could be represented by the tensor `[0.56, 0.45`] and _kitten_ by `[0.56, 0.445]`, whereas _mat_ could be `[0.2, -0.1]`. We cluster similar *words together in the vector space* and can do distance checks such as Euclidean or cosine distance functions to determine how close words are to each other. And how do we determine where words fall in the vector space?
163. `Embedding layer` is similar to all other layers in DL , we initialise random values and hopefully the training process updates the parameters so that similar words or concepts gravitate towards each other 
164. To use embedding in pytorch - 
```python 
embed = nn.Embedding(vocab_size, dimension_size)
```

165. This will create a tensor of size `vocab_size` x `dimension_size` which is initialised randomly.
166. Each word in your vocabulary indexes into an entry that is a vector of `dimension_size`, so if we go back to our cat and its epic adventures on the mat, we’d have something like this:
```python 
cat_mat_embed = nn.Embedding(5,2)
cat_tensor = Tensor([1])
cat_mat_embed.forward(cat_tensor)
## output
> tensor([[ 1.7793, -0.3127]], grad_fn=<EmbeddingBackward>)
```

167. We create out embedding , a tensor that contains the position of a cat in our vocabulary and pass it through the layers `forward()` method. That gives our random embedding. The result also points out that we have gradient function that we can use for updating the parameters, after we combine with the loss function 


### Torchtext 

168. Just like `torchvision` we have `torchtext` for handling text processing pipelines 
169. We will implement the twitter dataset - `Sentiment140Dataset` 
170. Download the zip archive and unzip. We use the file _training.1600000.processed.noemoticon.csv_. Let’s look at the file using pandas:

```python
import pandas as pd
tweetsDF = pd.read_csv("training.1600000.processed.noemoticon.csv",
                        header=None)
```

171. You may at this point get an error like this:

```python
UnicodeDecodeError: 'utf-8' codec can't decode bytes in
position 80-81: invalid continuation byte
```

172. This means that the C based csv parser that pandas uses is not going to work so we switch to python based one


## A Journey into Sound

173. . Sound stores one value at a time t , this is different from an image which requires 2 values x and y ( for a grey scale image ) 
174. If we use convolutional filters in our neural network, we need a 1D filter rather than the 2D filters we were using for images.
175. The _Environmental Sound Classification_ (ESC) dataset is a collection of field recordings, each of which is 5 seconds long and assigned to one of 50 classes (e.g., a dog barking, snoring, a knock on a door).
176. We will be using `torchaudio` to simplify loading and manipulation of audio. 
177. Clone the repo of ESC-50 dataset in the form of WAV files
```git
git clone https://github.com/karoldvl/ESC-50
```
178. Instead of using music player we can utilise the jupyter music player inside the notebook with the help of - ``IPython.display.Audio`` 
```python 
IPython.display.Audio
display.audio('ESC-50/audio/1-10032.wav)
```

179. To play a song using jupyter 
```python 
import librosa 
import pandas as pd 
import numpy as np 
import os 
from IPython.display import Audio

file_path = '/home/parthshr370/Downloads/The Weeknd - Blinding Lights.wav'

def read_audio_and_play (file_path):
    
    audio,sr = librosa.load(file_path,sr = None)
    
    print(f"duration:{librosa.get_duration(y= audio, sr= sr):,.2f}seconds")
    
    return Audio(audio,rate=sr)

read_audio_and_play(file_path)
```
180. To play and produce an audio with numpy array - 
```python 
def gen_and_play(frequency = 440, duration= 3, sample_rate = 22050):
    t = np.linspace(0, duration,int(sample_rate*duration), False) # Linspace presents you with evenly spaced samples in a matrix/array
    sine_wave = np.sin(2*np.pi*frequency*t)
    
    # convert to tensor 
    tensor  = torch.from_numpy(sine_wave.astype(np.float32))
    
    numpy_array = tensor.numpy()
    
    
        # Play the generated sound
    print(f"Playing generated sine wave at {frequency} Hz")
    return Audio(numpy_array, rate=sample_rate)

# Generate and play a 440 Hz sine wave
display(gen_and_play(50))
```


181. `Torchaudio` comes with `load()` and `save()` , `load()` will be the most used one but we might also use save as when generating a new audio track with a specific `file_path`
182. When using most `torchvision` and `torchtext` based image and text processing , pytorch did most of the data processing for us but we need to do some heavy lifting while working with `torchaudio`
183. To create a custom dataset you need to apply two classes - `__getitem__` and `__len__` we will also need an `__init__` method to use the file paths or anything we need to reuse multiple time 
```python 
class data_audio(Dataset):
    
    def __init__(self,path):
        files = Path(path).glob('*.wav') #uses pythons path library to find the location of all the wav files 
        
                # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [(f,int(f.name.split("-")[-1]
                    .replace(".wav",""))) for f in files]
        self.length = len(self.items)
        
        
    def __getitem__(self,index):
        filename,label = self.item[index]
        audio_tensor , sample_rate = torchaudio.load(file)
        return audio_tensor, label 
    
    def len(self):
        return self.length
```

184. The majority of the work in the class happens when a new instance is created 
185. The `__init__` method takes the path parameter finds all the `.wav` files inside the path and then produces tuples with `(filename,label).
186. When Pytorch requests an `item` from the `dataset` we index into the items list and use `torchaudio.load` to make `torchaudio` load in the audio file and turn it into a tensor 
187. Enough for us to start so we now create an ESC50 Object 
```python 
test_esc50 = ESC50(PATH_TO_ESC50)
tensor, label = list(test_esc50)[0]

tensor
tensor([-0.0128, -0.0131, -0.0143,  ...,  0.0000,  0.0000,  0.0000])

tensor.shape
torch.Size([220500])

label
'15'
```

188. We can now construct a pytorch dataloader with basic pytorch constructs 
```python 
example_loader = torch.utils.data.Dataloader(test_esc50,batch_size = 64,shuffle = True)
```

189. We also need to create a train test split  with validation and test sets 
190. The compilers of the dataset separated the data into five equal balanced _folds_, indicated by the _first_ digit in the filename. We’ll have folds `1,2,3` be the training set, `4` the validation set, and `5` the test set. But feel free to mix it up if you don’t want to be boring and consecutive! Move each of the folds to _test_, _train_, and _validation_ directories:






















### Some interesting things about Resnet Trained on ImageNet Dataset
The mean and standard deviation values used for normalizing images in pre-trained models like ResNet have a specific origin and purpose. Let me explain:

1. Origin of the Values: These specific values (mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]) come from the ImageNet dataset, which was used to pre-train models like ResNet.
2. Calculation Process:
    - These values were calculated by computing the mean and standard deviation of each color channel (R, G, B) across all images in the ImageNet training set.
    - The calculation was done after scaling the pixel values to the range [0, 1].
3. Significance:
    - They represent the average color distribution of the ImageNet dataset.
    - The mean values being close to 0.5 indicate that, on average, the dataset images are neither too bright nor too dark.
    - The standard deviation values show the spread of color intensities in each channel.
4. Purpose:
    - By normalizing input images using these values, we're adjusting new images to have a similar color distribution to what the model was trained on.
    - This helps the pre-trained model perform more effectively on new data, as it's seeing inputs with a similar statistical distribution to its training data.
5. Dataset Specificity:
    - These values are specific to ImageNet and models trained on it.
    - Different datasets might have different optimal normalization values.