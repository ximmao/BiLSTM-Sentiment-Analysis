# BiLSTM-Sentiment-Analysis

NLP project
Understanding the impact of hypernym on sentiment analysis
this is the repo for a biLSTM implemented for the project

Dataset naming is following [Dataset_Name][|-both|-root]-[train|valid|test].csv. For example, Twitter-root-train.csv is Twitter's training set while converting to root hypernym

Unzip the code, in the code folder, create ./dataset folder and put all the dataset; create emtpy ./model folder for future saved final model
train.py can be ran with two argument --datasetToTrain & --levelOnHypernym
--datasetToTrain control which training and validation dataset to run, choice from: IMDB, Twitter
--levelOnHypernym control which hypernym level on the dataset to run, choice from: NoneUsed, ToRoot, KeepBoth

Sample scipt using nohup:
 nohup python -u train.py --datasetToTrain Twitter --levelOnHypernym ToRoot 2>&1 &

Training will stop if validation is not improving for 5 consecutive times, model will be saved


Only Twitter dataset is provided in ./dataset
