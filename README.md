Introduction
------------

This module is written by Jiacheng Hou (300125708). The module trains a Graph Neural Network (GNN) to 
classify human activities based on smartphone sensors information only. 

The Human Activity Recognition (HAR) dataset [1],[2] is available in the UCI Machine Learning Repository.

The Graph Neural Network model is from paper [3]. It's a 
graph classification model where includes three layers of Graph Convolutional
Networks(GCN) [4], one pooling layer and two fully connected layers. 

The idea of applying a graph classification model on the 
smartphone sensor data is inspired by paper [5]. They claim
they are the first ones to use GNN for smartphone sensor-based
HAR.

The dependency requirement of running the code (GNN.py) is available in the "req.txt" file. The dataset visualization and GNN model performance on the training and validation dataset can be found
in the "plot/" folder.
<br />
<br />


References
------------
* [1] HAR Dataset. UCI Machine Learning Repository: Human Activity Recognition using smartphones data set. (2012). Retrieved March 8, 2022, from https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones 
* [2] Roobini, M. S., & Naomi, M. J. F. (2019). Smartphone sensor based human activity recognition using deep learning models. Int. J. Recent Technol. Eng, 8(1), 2740-2748.
* [3] Monti, F., Frasca, F., Eynard, D., Mannion, D., & Bronstein, M. M. (2019). Fake news detection on social media using geometric deep learning. arXiv preprint arXiv:1902.06673.
* [4] Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
* [5] Mondal, R., Mukherjee, D., Singh, P. K., Bhateja, V., & Sarkar, R. (2020). A new framework for smartphone sensor-based human activity recognition using graph neural network. IEEE Sensors Journal, 21(10), 11461-11468.
