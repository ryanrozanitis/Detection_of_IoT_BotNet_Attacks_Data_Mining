from sklearn import svm
import pandas
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Depending on the size of the dataset imported, I have doubled or even tripled the amount of data points to get a more
# accurate result.
# Read in the benign traffic samples.
with open(
        'C:\\benign_traffic.csv',
        'rt')as BenignTrafficCSV:
    BenignTrafficFrame = pandas.read_csv(BenignTrafficCSV, index_col=0, header=0)
    BenignTrafficFrame = pandas.concat([BenignTrafficFrame, BenignTrafficFrame, BenignTrafficFrame])

# Read in the malignant traffic samples.
with open(
        'C:\\combo.csv',
        'rt')as MalignantTrafficCSV:
    MalignTrafficFrame = pandas.read_csv(MalignantTrafficCSV, index_col=0, header=0)
    MalignTrafficFrame = pandas.concat([MalignTrafficFrame, MalignTrafficFrame])

with open(
        'C:\\udp.csv',
        'rt')as MalignantTrafficUDPCSV:
    MalignTrafficFrameUDP = pandas.read_csv(MalignantTrafficUDPCSV, index_col=0, header=0)

with open(
        'C:\\tcp.csv',
        'rt')as MalignantTrafficTCPCSV:
    MalignTrafficFrameTCP = pandas.read_csv(MalignantTrafficTCPCSV, index_col=0, header=0)

with open(
        'C:\\scan.csv',
        'rt')as MalignantTrafficScanCSV:
    MalignTrafficFrameScan = pandas.read_csv(MalignantTrafficScanCSV, index_col=0, header=0)
    MalignTrafficFrameScan = pandas.concat([MalignTrafficFrameScan, MalignTrafficFrameScan, MalignTrafficFrameScan])

with open(
        'C:\\junk.csv',
        'rt')as MalignantTrafficJunkCSV:
    MalignTrafficFrameJunk = pandas.read_csv(MalignantTrafficJunkCSV, index_col=0, header=0)
    MalignTrafficFrameJunk = pandas.concat([MalignTrafficFrameJunk, MalignTrafficFrameJunk, MalignTrafficFrameJunk])

NewMalignTrafficFrameUDP = MalignTrafficFrameUDP.drop_duplicates(keep=False)
NewMalignTrafficFrameTCP = MalignTrafficFrameTCP.drop_duplicates(keep=False)

# Grab the length of each data set. (Simply put, get the row count)
# I am using 0 as the indicator for benign traffic, and 1 as an indicator for malignant traffic. This builds an array
# filled with 0s up to the count of Benign Traffic samples and 1s up to the count of Malignant Traffic Samples.
bucket = ([0] * len(BenignTrafficFrame.index) + [1] * len(MalignTrafficFrame.index) +
          [2] * len(NewMalignTrafficFrameUDP.index) + [3] * len(NewMalignTrafficFrameTCP.index) +
          [4] * len(MalignTrafficFrameScan.index) + [5] * len(MalignTrafficFrameJunk.index))

# Combine the frames. Only using the benign and combo attack packets for making the model
TrafficFramesForTesting = [BenignTrafficFrame, MalignTrafficFrame, NewMalignTrafficFrameUDP, NewMalignTrafficFrameTCP,
                           MalignTrafficFrameScan, MalignTrafficFrameJunk]
TotalTraffic = pandas.concat(TrafficFramesForTesting)

# Split the data. Here, 25% is used for training, 75% for testing.
X_Train, X_Test, y_train, y_test = train_test_split(TotalTraffic, bucket, random_state=0, train_size=0.25)

sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.fit_transform(X_Test)

print("# Tuning hyper-parameters")  # for %s % score)
print()

# This part here is the comprehensive test I did to find the best fit for the data. This set ran for about
# 7-10 hours(ran over night) searching all possibilities based off of the parameters indicated
#
param_grid = [{'C': [1, 10], 'gamma': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], 'kernel': ['rbf'],
               'shrinking': [False], 'decision_function_shape': ['ovr'], 'break_ties': [False]},
             {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'shrinking': [False], 'decision_function_shape': ['ovr'],
               'break_ties': [False]},
             {'C': [1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
              'kernel': ['sigmoid'], 'shrinking': [False], 'decision_function_shape': ['ovr'], 'break_ties': [False],
              'coef0': [0.0, 1.0]},
              {'C': [1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
               'kernel': ['poly'], 'shrinking': [False], 'decision_function_shape': ['ovr'], 'break_ties': [False],
               'coef0': [0.0, 1.0], 'degree': [3, 4, 5, 6, 7, 8, 9, 10]}]

# Iterate through the param grid and feed it into SVM
clf = GridSearchCV(
    svm.SVC(cache_size=4096, class_weight=None, max_iter=500, probability=False, random_state=0,
            tol=0.001, verbose=False), param_grid)

clf.fit(X_Train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.5f (+/-%0.05f) for %r"
          % (mean, std * 2, params))
print()

correctValues = [y_test]
testingDataSamples = [X_Test]

for CorrectTestingValues, TestingDataForPrediction in zip(correctValues, testingDataSamples):
    y_true, y_pred = CorrectTestingValues, clf.predict(TestingDataForPrediction)
    print(classification_report(y_true, y_pred))
    print()

    print()
    print(confusion_matrix(CorrectTestingValues, y_pred))

    TotalTestSamples = 0
    correctTestSamples = 0

    for correct, predicted in zip(CorrectTestingValues, y_pred):
        TotalTestSamples += 1
        if correct == predicted:
            correctTestSamples = correctTestSamples + 1

    print()
    print((correctTestSamples / TotalTestSamples) * 100, "% correct")
