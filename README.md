# Face Recognition using Python and OpenCV
Built on OpenCV 3.2.0 and Python 3.6.0/Anaconda 4.3.0. Code to detect faces using Haar Cascade and match faces using LBPH (Local Binary Patterns Histogram) Recognition on a live web camera.
1. Creating your dataset
Camera window opens and faces are detected using Haar Cascade (Frontal-Face) on execution. Press 'c' to capture the face. The program creates the directory 'dataset'. Input the name of the person as label when prompted. A new directory inside dataset is created by this new label. The image of the frame is stored in a numbered order in PNG format. To add pictures of the same person, use the same label name when prompted. Simply change the label name to save pictures of a new face. Another directory in 'datasets' is created. Hence each person gets a directory in 'dataset' to store our images in an ordered and organised manner.

2. Detecting the faces
Once some data is added in the dataset directory, the program automatically trains using the LBPH Face Recognition. The recognizer is trained on the selected cropped faces from the dataset. The faces in front of the web cam are used to predict labels on the trained model. Set a minimum confidence score to display labels under detected faces. The faces in front of the web cam are automatically labelled over the video window according to the predictions. A confidence score is also displayed over the frame.

3. Press 'escape' to exit execution
