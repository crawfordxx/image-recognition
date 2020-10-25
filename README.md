
# Source code of training flower recognition model based on TensorFlow

Next, we will retrain the existing Google Inception-V3 model and train the 4 kinds of flower sample data to complete a model that can recognize 4 kinds of flowers, and test the newly trained model.

![sample image](https://img-blog.csdn.net/20180602195623764?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FueW1ha2VfcmVu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


### Install TensorFlow 

```
# Python 3
➜ pip3 install tensorflow 
```
[Installing TensorFlow](https://www.tensorflow.org/install/)


#### Check if tensorflow is installed succeed or not

After entering the Python environment, enter the following code. When `"Hello, TensorFlow!"` appears, it indicates that the installation is successful and TensorFlow can be used normally.

```
➜ python3
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
Hello, TensorFlow!

```

### Download sample test images

The sample test images are in the "images.zip" file, it contains 4 kinds of weeds in Victoria。Download and unzip.

Open the training sample folder images, there are 4 categories of flowers: `African daisy, African Boxthorn, artichoke thistle, boneseed`, a total of 3672 photos, each category has about 600-900 training sample pictures.



### Start trainning

**Download the retrain script used to train the model**
This script will automatically download the google Inception v3 model related files. `retrain.py` is a script provided by Google that uses the ImageNet image classification model as the basic model and uses the flower_photos data migration to train the flower recognition model.

```
 cd flower_demo
 curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py

```
**Start the training script and start training the model**

When running the `retrain.py` script, you need to configure some running command parameters, such as specifying the input and output related names of the model and other training requirements. Among them, the `--how_many_training_steps=4000` configuration represents the number of training iterations. The default value is 4000. If the machine performance is good enough, you can reduce this value appropriately.

```
➜ cd flower_demo
➜ python3 retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=images

```
I used 20000 steps to train the model, because what we want is a precise model and my laptop has high performance with GPU drive，It can be seen that `Final test accuracy = 92.1%` on the test set, which means that the 4 types of flower recognition models we trained have a 92% recognition accuracy rate on the test set. The generated `retrained_labels.txt` and `retrained_graph.pb` are model-related files.
```
2018-06-02 15:47:00.266119: Step 3950: Train accuracy = 94.0%
2018-06-02 15:47:00.266159: Step 3950: Cross entropy = 0.135385
2018-06-02 15:47:00.327843: Step 3950: Validation accuracy = 93.0% (N=100)
2018-06-02 15:47:00.976543: Step 3960: Train accuracy = 94.0%
2018-06-02 15:47:00.976591: Step 3960: Cross entropy = 0.234760
2018-06-02 15:47:01.038559: Step 3960: Validation accuracy = 91.0% (N=100)
2018-06-02 15:47:01.667255: Step 3970: Train accuracy = 97.0%
2018-06-02 15:47:01.667372: Step 3970: Cross entropy = 0.167394
2018-06-02 15:47:01.731935: Step 3970: Validation accuracy = 87.0% (N=100)
2018-06-02 15:47:02.355780: Step 3980: Train accuracy = 96.0%
2018-06-02 15:47:02.355818: Step 3980: Cross entropy = 0.151201
2018-06-02 15:47:02.418314: Step 3980: Validation accuracy = 91.0% (N=100)
2018-06-02 15:47:03.042364: Step 3990: Train accuracy = 99.0%
2018-06-02 15:47:03.042402: Step 3990: Cross entropy = 0.094383
2018-06-02 15:47:03.103718: Step 3990: Validation accuracy = 91.0% (N=100)
2018-06-02 15:47:03.667861: Step 3999: Train accuracy = 99.0%
2018-06-02 15:47:03.667899: Step 3999: Cross entropy = 0.106797
2018-06-02 15:47:03.729215: Step 3999: Validation accuracy = 94.0% (N=100)
Final test accuracy = 92.1% (N=353)
```
### Test model

Similarly, we first download the test model script `label_image.py`, and then select the images(1).jpg from the images/African daisy/ folder to test the recognition accuracy of our trained model. Of course, you can also search for one on Google Test the recognition effect on any picture of the 4 types of flowers. I defined if the similarity is lower than 75%, then it shows "negative", otherwise it shows the most similar weed.

```
➜ cd flower_demo
➜ python label_image.py images/frican daisy/images(1).jpg

African daisy
```


