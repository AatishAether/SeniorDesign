# SeniorDesign

## Objective 
Compare the efficacy of various methods of computer vision in combination with machine learning as applied to signed translation tasks in real time. 
## Methods 
The following methods will be used in coordination with the MediaPipe vectorization library. The 
goal there is to be able to translate pertinent information from the image or video stream into 
usable but more spatially efficient data. The hands, in this case, are skeletonized and then 
stored as vectors to be used in a variety of ways. 
### Rule Based Recognition 
This is the "simplest" of the methods examined here, in a sense. It is tricky to code and finite in its scope, however, provides us with insight as 
to the inner workings and mechanisms of models which are trained to predict the "outcome" of the signs given to them as input. In this method we 
simply take the vectors we want and then define them ourselves as being a specific letter or not. 
### Recognition of Dynamic Signs using DTW 
Dynamic Time Warping (DTW) relies on taking a set of data as reference for the action or activity that one would like to capture and then applying 
certain knwoledge gleaned from this "training" set to any input that matches the requirements of the program. The input is taken and its pertinent 
features are extracted and compared to the existing training set. This was very successful for our translation of moving gestures/signs. This is 
because DTW can take in a set of video frames or images in sequence as the reference or training set and then compare that to another set of images 
taken as input and it will compare these two in a non euclidean manner so that the length of the action you are looking to capture or the speed at 
which it occurs is inconsequential. 
### Recognition of Static Images Using a Convolutional Neural Network 
Convolutional neural networks are wuite good at classification tasks, not least of all, image classification tasks. Our group took it as a proof of 
concept to work on a neural network which would function based off of static images. This has at least two major shortcomings when it comes to our 
end goal in that this method is computationally expensive when all frames in a video sequence are considered, and as such, making this happen in 
real time is not as feasible. Secondly, the network itself does not take in time as a consideration and as such cannot identify sequences of actions. 
A third significant shortcoming of this approach is that it requires training data in several resolutions and at various positions to be most effective. 
### Current Work in Neural Networks 
Seeking to resolve the problem of the excessively large datasets, we investigate the usage of MediaPipe to abstract the issue of both scaling (how large or 
small a hand is) as well as that of the large datasets required for optimal results by instead training a neural network on the pertinent points in 3D space 
captured by MediaPipe. These could be stored and labeled in a .csv or .xlsx file thus reducing the size of our training data on the disk from several gigabytes 
in static images to under a hundred megabytes.  <br> 

&emsp; Considerations: <br>
&emsp;&emsp; 1. Using a neural network as described above leads us to investigate how we could incorporate the time factor into our training set so as to be 
&emsp;&emsp; able to capture sequences of gestures. 
