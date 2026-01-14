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
