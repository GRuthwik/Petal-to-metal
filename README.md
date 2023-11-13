There are over 5,000 species of mammals, 10,000 species of birds, 30,000 species of fish – and astonishingly, over 400,000 different types of flowers.
We are  challenged to build a machine-learning model that identifies the type of flowers in a dataset of images (for simplicity, we’re sticking to just over 100 types).
To achieve this result we will use the concept of TPU(Tensor Processing Unit).
Tensor Processing Unit (TPU) is an AI accelerator application-specific integrated circuit (ASIC) developed by Google for neural network machine learning, using Google's own TensorFlow software. Google began using TPUs internally in 2015, and in 2018 made them available for third-party use, both as part of its cloud infrastructure and by offering a smaller version of the chip for sale.


We use this concept because in most cases while building an image classifier we are mostly bottlenecked by the hardware of the person using it. By using TPU we can accelerate the processing time of the model vastly which will result in faster results.


I have completed implementing this work using Kaggle as my compiler in which there is a competition in which we can participate for free and practice with the resources available and I have chosen Python as my computing language.


I executed this work by building an image classifier in Keras and training it on a Tensor Processing Unit (TPU)


Firstly We begin by importing several Python packages that are needed


A TPU has eight different cores and each of these cores acts as its accelerator. (A TPU is sort of like having eight GPUs in one machine.) We tell TensorFlow how to make use of all these cores at once through a distribution strategy. Run the following cell to create the distribution strategy that we'll later apply to our model.



When used with TPUs, datasets need to be stored in a Google Cloud Storage bucket. You can use data from any public GCS bucket by giving its path just like you would data from '/kaggle/input'. The following will retrieve the GCS path for this competition's dataset.


When used with TPUs, datasets are often serialized into TFRecords. This is a format convenient for distributing data to each of the TPU cores. We've hidden the cell that reads the TFRecords for our dataset since the process is a bit long. You could come back to it later for some guidance on using your datasets with TPUs and creating data pipelines.


Now we're ready to create a neural network for classifying images! We'll use what's known as transfer learning. With transfer learning, you reuse part of a pre-trained model to get a head start on a new dataset.

For this tutorial, we'll use a model called VGG16 pre-trained on (ImageNet).

The distribution strategy we created earlier contains a context manager, strategy. scope. This context manager tells TensorFlow how to divide the work of training among the eight TPU cores. When using TensorFlow with a TPU, it's important to define your model in a strategy.scope() context.



The 'sparse_categorical' versions of the loss and metrics are appropriate for a classification task with more than two labels, like this one.


We'll train this network with a special learning rate schedule. with a learning rate of 0.005 and a starting learning rate of 0.005.


And now we're ready to train the model. After defining a few parameters we can get the result.


For a better understanding of the result data let's plot a simple graph with the result plotted.

My Contribution:

After the initial model prediction, I used a new model called the VGG16 model found the prediction value, and plotted it for reference and understanding of the difference.



Challenges Faced and their Solutions :

while loading the data initially the data was in the format of TF records. so I used the tensor flow method to load the data

While training the model the major problem was hardware limitation and high execution time. so to solve that I used TPU accelerators that are in-built into the Kaggle work environment.


Source Code: https://www.kaggle.com/ruthwikgangavelli/petals-to-metals 


References:

https://en.wikipedia.org/wiki/Tensor_Processing_Unit 

https://www.kaggle.com/competitions/tpu-getting-started 

https://www.kaggle.com/docs/tpu 

https://www.kaggle.com/code/ryanholbrook/create-your-first-submission/notebook 

https://www.kaggle.com/code/georgezoto/computer-vision-petals-to-the-metal 

https://www.kaggle.com/code/sathviknitap/tpu-flower-classification-assignment#load-my-data 
