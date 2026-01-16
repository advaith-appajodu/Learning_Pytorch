# Learning_Pytorch

Notes Of what I've learned:

Day 1:
Was just practice for the tools and not much tinkering so I didn't build much ML intuition


Day 2:

If I add more layers to my NN, using SGD as my optimizer causes significantly less learning. A model with 9 layers barely has a decreasing in Loss witha  6.5% accuracy for 3-5 epochs and not much significant increase but a model with 5 layers reaches 80% by the end of the first epoch and ends with 85- 87%

5-7 epochs seems to be the most optimal as the loss seems to increase and the accuracy decrease from there. There is an occasion rise by the 10th epoch but it always ends below the peak.

Learning rate for SGD seems best at 0.001 or 0.0001 as 0.01 and 0.1 is a faster way to higher accuracy but high loss fluctuations in later epochs

Changing Optimizers to Adam increases learning exponentially. Seems to be a better optimizer so far. Accuracy doesn't seem to regress at higher epochs but rather stabilize at 87.9%

Someone Suggested adding dropout, so i added a dropout of 0.3 and increased the dimensions of each matrix while decreasing the depth. I also reduced learning rate for Adam based on that friend's suggestion I am also going to try and train for 30 epochs cause law of large numbers.

At 30 epochs, there was an overall increase to 89.9% with occasional decreases and platues.

I'm going to add weight decay, which limits the model from relying too much on one feature but that works against Adam, which gives table weights larger updates, so I'm going to switch to AdamW which accomodates this.

Still can't break 90% so I'm going to try and add Batch Normalization layers.

Finaly broke to 90.4%, which is consistent when I replicate it. Reached my goal for the day and learned more than I planned.
Started at 6.4% accuracy and ended the day with 90.4%

Day 3:
Learning to train/build CNNs today. Trying to get as high an accuracy on CIFAR-10 as I can.
Loading the dataset is the same process and traning/testing seems to be the same process. Only difference is the model architechture. 

Started with a simple 5 layer CNN. Two convulution layers, one pooling layer, one linear layer, and one flattening layer. Started with Adam as the optimizer and Cross Entropy Loss as the Loss function. Baseline Accuracy is 57.7% with Final mean loss of 1.22 over 5 epochs. 

Added 2 more convolution layers and 1 more pooling layers. Channels go from 3->8->16->32->64, then flattened for the linear layer. This should extract more features each step and increase accuracy. Ended with an accuracy of 64% and final mean loss of 1.22 over 5 epochs which is an improvemnt.

Im going to increase the epochs so I can observe more trends and change the optimizer to AdamW. Changing the optimizer shouldn't make any difference since the weight decay is 0. Ended with a training accuracy of 74.5% and a testing accuracy of 67.7% I think I should focus on a higher training accuracy before I worry about test, so unless there is a huge disparity between the two(i.e lot of overfitting), I'm not going to be recording test accuracy until I am satisfied with the training. Goal is around 95%+

Idea 1: Batch normalize
Since that leads to a more uniform distribution per channel and helps keep the important features, I'm going to add a Batch Normalization layer after every conv layer. Ended ith a training accuracy of 78%, but the test loss started to trend upwards towards the end, so there is some overfitting going on. Will deal with it later.

Idea 2: More Channels

Find more features to learn sounds good. New progression now looks like: 32 → 64 → 128 → 256
This reached 97.8% training accuracy but test loss decreased and then increased again, so now I'm going to deal with the overfitting problem.

Since I already switched to AdamW, setting weight decay to 1e-4 is the easiest approach. Ok So the accuracy on train is 97.3% but 73.6% on the test set which is not too shabby, but the test loss increased every epoch while test accuracy fluctuated in the same 68-73% range meaning overfitting is still occuring.

Next strategy is to introduce dropout to reduce overfitting. Worst case scenario, I reduce the number of epochs trained since the model isn't learning a whole lot in later epochs. Dropout of 0.3 seems nice and medium. The results are almost the exact same, but just pushes it to later epochs. 16-30 have strong overfitting. Since both the accuracy and loss are increasing, I'm assuming that the model is mostly correct, but is VERY wrong when it is wrong.

Before I reduce epochs, let me introduce ReLU/non-linearity to the model. I'm suprised I waited this long. I'll add a RelU activation after every batch normalization to see if this helps with the overfitting since this helps the linear layer learn better and memorize less. That helps slightly, but still overfitting.

Ok so major realization. From my understanding, CNNs extract features from the picture that it learns are important then feeds it into a linear layer for classfficiation, which is why its better than a simple linear NN at images since those only take a flattened vector of an image. So, why am I only using one linear layer to try and process so many features? That might be the reason my model is memorizing and not learning. So I'm going to add two more linear layers with non-linear activations between them. That seemed to work worse. Test accuracy reached 80% and loss is below one, but then fell from epochs 20->30, so I'm going to reduce traning time to 20 epochs and reduce depth of FCL layer. Working theory is that adding more parameters than images in the dataset causes more memorization since there are way too many learnable parameters. My initial thinking was completely wrong and misguided. 

After looking through the PyTorch documentation and a LOT of google, I think I found my solution. For my classifier I can use Global addaptive pooling, which returns a tensor taht indicates whether a feature learned by a filter exists in the picture  regardless of its location. Simply using flatten seems to treat the same feature differently depending on the location since it "flattens" the pixels. This reduced the overfitting but Accuracy is at 77.5% and a loss of 1.03, so the goal is to get a loss between 0.3-0.7 and an accuracy above 85%.

Since my CNN is built with all of it's bells and whistles, I want to experiment with the hyperparameters now. Since overfitting is the problem, I'll start by raising the weight decay from 1e-4 to 1e-3. Dramatic, but I can always bring it down if my train accuracy goes below 90%. Feeling a little stumped here. Increasing Decay should decrease overfitiing, and my traning accuracy was at 95.4 with a loss of 0.08 and a test accuracy of 76.8% and loss of 1.08, which still indicates overfitting.

Since the loss seems to be very jumpy, there's a chance that the learning rate is too high se let's test reducing that to 1e-4. So a slower learniing rate is leading to a more stable increase, but ended with a and accuracy in 65% in both, so I'm going to increase the epochs again. A lower learning rate is slow, but train and test accuracy as well as loss are much closer together so it might be worth it. Might graduate College before this finished training though. All my strategies seem to reduce the gap in overfitting but doesn't fight it comepletely.

So final idea of the night: Data Augmentation. I wanted to save data cleaning and manipulation for future projects but it seems like I can't get to my goal without it, so I am going to incorporate random crop and rotate in my traning set to introduce diversity in the images. OH MY GOD. Finally. The Accuracy for both sets is stably increasing and the Loss is stably decreasing. So after 225 epochs, I ended with Train Epoch: Avg Loss: 0.3882, Accuracy: 86.47% Test: Avg Loss: 0.5155, Accuracy: 82.41%. The loss is within my acceptable range and my accuracy is pretty close to each other, so I know I solved the overfitting issue, but I can't seem to push the model further. I think I need more advanced techniques to create a more powerful CNN or add more dimensions, but I think this toy exercise taught me a lot more than I thought I was going to learn, so I'm going to put a pause on this for now and branch to multi headed attention transformers tomorrow.

Forgot to use WandB, will do it next time. Final Graph:
<img width="515" height="307" alt="image" src="https://github.com/user-attachments/assets/127e1c6e-a6c6-448a-b83d-7ed30fc50e90" />



