# hotdog-classifier
Funny problem from "Silicon Valley" series. There is dataset of images.
Need to classify them into two classes: with and without hotdog.

- Dataset volume: 1669 images
- Images with hotdog: 707
- Images without hotdog: 962

##Used training params:
- Test dataset volume: 20% (333 images)
- Random state: 17
- Batch size: 32
- ConvNN epochs count: 20
- ConvNN criterion: Log loss
- ConvNN optimizer: Adam

###ConvNN statistics:
- Train loss: 0.12099046260118484
- Train accuracy: 0.96875
- Test loss: 0.06452243775129318
- Test accuracy: 0.862012987012987
- Learning time: ~10 min

###SVM statistics:
- Test accuracy: 0.7035928143712575
- Learning time: ~16 min

###SVM with pre-trained VGG16 statistics:
- Test accuracy: 0.9431137724550899
- Learning time: ~7 sec

##Folders
- data - train and score datasets
- logs - tensorboard logs for ConvNN
- models - serialized models
- result - final scores of models on unmapped data

##Conclusion
The worst result was given by SVM. Its accuracy was only 70%.
The convolutional neural network gave a good result in 86%, 
which in general coincides with the results of its other implementations.
SVM with pre-trained VGG16 shows itself as the most suitable model, 
giving an accuracy of 94%. Changing of tolerance for stopping criterion 
for SVM gives better result in almost 95% (0.9491017964071856).