import sys
import numpy as np
import pandas as pd

####################
### Set the hyperparameters in you myanswers.py file ###
####################
def MSE(y, Y):
    return np.mean((y-Y)**2)

from my_answers import iterations, learning_rate, hidden_nodes, output_nodes, NeuralNetwork

def get_datasets():
    # Load Data
    data_path = '/Users/ujones/Dropbox/Data Science/Python Project Learning/Udacity/Deep Learning/Course/deep-learning-v2-pytorch-master/project-bikesharing/Bike-Sharing-Dataset/hour.csv'
    rides = pd.read_csv(data_path)

    # One Hot Encoding
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                    'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)

    # Scaling features
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std

    # Save data for approximately the last 21 days 
    test_data = data[-21*24:]

    # Now remove the test data from the data set 
    data = data[:-21*24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    # Hold out the last 60 days or so of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]

    return train_features, train_targets, val_features, val_targets


def main():
    train_features, train_targets, val_features, val_targets = get_datasets()
    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train':[], 'validation':[]}
    for ii in range(iterations):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']

        # For Testing
        if ii > 3000:
            print(f"TESTING iteration {ii}")
                                
        network.train(X, y)
        
        # Printing out the training progress
        train_loss = MSE(np.array(network.run(train_features)).T, train_targets['cnt'].values)
        val_loss = MSE(np.array(network.run(val_features)).T, val_targets['cnt'].values)
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                        + "% ... Training loss: " + str(train_loss)[:5] \
                        + " ... Validation loss: " + str(val_loss)[:5])
        sys.stdout.flush()
        
        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

if __name__ == "__main__":
    main()