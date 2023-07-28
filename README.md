# AutoEER: Automatic EEG-based Emotion Recognition with Neural Architecture Search
Code implementation of the paper AutoEER: Automatic EEG-based Emotion Recognition with Neural Architecture Search.
## Data Preparation
##### Before running AutoEER, download the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html), unzip it and put it in the correct directory. 
##### And run `deap_process.py` to segment the EEG signal and extract the DE features of the four frequency bands.
## Architecture Search
`python search.py \\`
## Architecture Evaluation
`python train_DE.py \\`
