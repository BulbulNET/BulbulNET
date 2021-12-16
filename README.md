# BulbulNET – 
## neural network for bulbul call detection

### Introduction –
Convolutional neural network for detection of White Spectacled Bulbul calls.
The network's architecture is attached at the end of the file.


"main" script calls following functions:
-	Bulbul_conv_net: run network on training data from directory "train"
-	Gen_label + gen_label_text: extract a new file with no labels from directory "to_label", and generate txt file with labels according to the model predictions.
-	Run_data: runs the model on new data for detection of Bulbul calls. Extracts files from "all" directory and returns dataframe of predictions.
-	Cut_words: runs the function on "run_data" outputs. Cut all detected calls from audio file and save in directory "saved_words" 

### Dependencies –
- Python 3.8
- Tensorflow
- Keras 
- librosa

### Launch –
Run each script through the main scripts. 

### Examples of use –
Download directories and scripts at the same order as presented here, choose at "main" script which analysis to run by selecting "True":


### Generate labels:
runNetwork = True

generateLabels = True

runData = False

cutWords = False

in this example, after the network is trained, the model is saved in "data/…", and write txt labels to new audio files from directory "to_label". These labels can be imported and used in Audacity. 

### Detect bulbul calls in a new audio file and save the detected calls:
runNetwork = False

generateLabels = False

runData = True

cutWords = True

In this example, if you have a saved model, there is no need to run the network again. Run_data uses the saved model and creates predictions on new data, then cut the detected event with "cut_words" and saves them in directory.

![Figure7 cnn configuration](https://user-images.githubusercontent.com/96051637/146358336-5b1263f4-20eb-4415-85b6-b597cc67e0da.jpg)

