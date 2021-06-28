from sklearn import svm
import pandas
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Read in the benign traffic samples.
with open(
        'C:\\benign_traffic.csv','rt')as BenignTrafficCSV:
    # about 40000 samples, 10000 is 25% of the benign data
    BenignTrafficFrame = pandas.read_csv(BenignTrafficCSV, index_col=0, header=0)  # nrows=10000
    # print(BenignTrafficFrame, "\n")

# Read in the malignant traffic samples.
with open(
        'C:\\combo.csv', 'rt')as MalignantTrafficCSV:
    # about 50000 samples, 12500 is 25% of the malignant data
    MalignTrafficFrame = pandas.read_csv(MalignantTrafficCSV, index_col=0, header=0)  # nrows=12500
    # print(MalignTrafficFrame, "\n")

with open(
        'C:\\udp.csv','rt')as MalignantTrafficUDPCSV:
    MalignTrafficFrameUDP = pandas.read_csv(MalignantTrafficUDPCSV, index_col=0, header=0)

    # bucketUDP = [1] * len(MalignTrafficFrameUDP.index)

with open(
        'C:\\tcp.csv','rt')as MalignantTrafficTCPCSV:
    MalignTrafficFrameTCP = pandas.read_csv(MalignantTrafficTCPCSV, index_col=0, header=0)

    # bucketTCP = [1] * len(MalignTrafficFrameTCP.index)

with open(
        'C:\\scan.csv','rt')as MalignantTrafficScanCSV:
    MalignTrafficFrameScan = pandas.read_csv(MalignantTrafficScanCSV, index_col=0, header=0)

    # bucketScan = [1] * len(MalignTrafficFrameScan.index)

with open(
        'C:\\junk.csv','rt')as MalignantTrafficJunkCSV:
    MalignTrafficFrameJunk = pandas.read_csv(MalignantTrafficJunkCSV, index_col=0, header=0)

    # bucketJunk = [1] * len(MalignTrafficFrameJunk.index)

# print(TrafficFrames)
# print(TotalTraffic)

# Grab the length of each data set. (Simply put, get the row count)
# print(len(BenignTrafficFrame.index))
# print(len(# MalignTrafficFrame.index))
# I am using 0 as the indicator for benign traffic, and 1 as an indicator for malignant traffic. This builds an array
# filled with 0s up to the count of Benign Traffic samples and 1s up to the count of Malignant Traffic Samples.
# bucket = ([0] * len(BenignTrafficFrame.index) + [1] * (len(MalignTrafficFrame.index) + len(MalignTrafficFrameUDP.index)
#                                                        + len(MalignTrafficFrameTCP.index)
#                                                        + len(MalignTrafficFrameScan.index)
#                                                        + len(MalignTrafficFrameJunk.index)))

# bucket = [2] * len(MalignTrafficFrameUDP.index) + [3] * len(MalignTrafficFrameTCP.index)

# bucket = [0] * len(BenignTrafficFrame.index) + [1] * len(MalignTrafficFrame.index)

bucket = ([0] * len(BenignTrafficFrame.index) + [1] * len(MalignTrafficFrame.index)
          + [2] * len(MalignTrafficFrameUDP.index) + [2] * len(MalignTrafficFrameTCP.index)
          + [3] * len(MalignTrafficFrameScan.index) + [4] * len(MalignTrafficFrameJunk.index))
# print(buckets)

# Combine the frames. Only using the benign and combo attack packets for making the model
# TrafficFramesForTesting = [MalignTrafficFrameUDP, MalignTrafficFrameTCP]
TrafficFramesForTesting = [BenignTrafficFrame, MalignTrafficFrame, MalignTrafficFrameUDP, MalignTrafficFrameTCP,
                           MalignTrafficFrameScan, MalignTrafficFrameJunk]
TotalTraffic = pandas.concat(TrafficFramesForTesting)
# , MalignTrafficFrameUDP, MalignTrafficFrameTCP,
#                        MalignTrafficFrameScan, MalignTrafficFrameJunk]

# TotalTraffic = np.concatenate((BenignTrafficFrame, MalignTrafficFrame))

# Split the data. Here, 25% is used for training, 75% for testing.
X_Train, X_Test, y_train, y_test = train_test_split(TotalTraffic, bucket, random_state=0, train_size=0.75)

sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.fit_transform(X_Test)
# MalignTrafficFrameUDP = sc.fit_transform(MalignTrafficFrameUDP)
# MalignTrafficFrameTCP = sc.fit_transform(MalignTrafficFrameTCP)
# MalignTrafficFrameScan = sc.fit_transform(MalignTrafficFrameScan)
# MalignTrafficFrameJunk = sc.fit_transform(MalignTrafficFrameJunk)

# for score in scores:
print("# Tuning hyper-parameters")  # for %s % score)
print()

# this is to find the best fit for the data. I have tested a ton of different attribute combinations
#   but have since shrunk it down so that it runs more quickly.
param_grid = [{'C': [0.1, 1, 10], 'break_ties': [False], 'coef0': [0.0], 'decision_function_shape': ['ovr'],
               'gamma': [0.01, 0.1, 1], 'kernel': ['rbf'], 'shrinking': [False]}]
#               {'C': [1, 10], 'gamma': [1, 0.01, 0.001], 'kernel': ['sigmoid'], 'shrinking': [False],
#                'decision_function_shape': ['ovr'], 'break_ties': [False], 'coef0': [0.0, 1.0]},
#               {'C': [1, 10], 'gamma': [1, 0.01, 0.001], 'kernel': ['poly'], 'shrinking': [False],
#                'decision_function_shape': ['ovr'], 'break_ties': [False], 'coef0': [0.0, 1.0],
#                'degree': [3, 4, 5]}]

# param_grid = {'C': [1], 'break_ties': [False], 'coef0': [1.0], 'decision_function_shape': ['ovr'], 'degree': [4],
#           'gamma': [0.001], 'kernel': ['poly'], 'shrinking': [False]}

# This part here is the comprehensive test I did to find the best fit for the data. This set ran for about
# 7-10 hours(ran over night) searching all possibilities based off of the parameters indicated
#
# param_grid = [{'C': [1, 10], 'gamma': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], 'kernel': ['rbf'],
#                'shrinking': [False], 'decision_function_shape': ['ovr'], 'break_ties': [False]},
#              {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'shrinking': [False], 'decision_function_shape': ['ovr'],
#                'break_ties': [False]},
#              {'C': [1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
#               'kernel': ['sigmoid'], 'shrinking': [False], 'decision_function_shape': ['ovr'], 'break_ties': [False],
#               'coef0': [0.0, 1.0]},
#               {'C': [1, 10, 100, 1000], 'gamma': [100, 10, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
#                'kernel': ['poly'], 'shrinking': [False], 'decision_function_shape': ['ovr'], 'break_ties': [False],
#                'coef0': [0.0, 1.0], 'degree': [3, 4, 5, 6, 7, 8, 9, 10]}]

clf = GridSearchCV(
    svm.SVC(cache_size=4096, class_weight=None, max_iter=100, probability=False, random_state=0,
            tol=0.001, verbose=False), param_grid)
# kernel='rbf', shrinking=True,  decision_function_shape='ovr', break_ties=False, degree=3, coef0=0.0,

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

correctValues = [y_test]  # , bucketUDP, bucketTCP, bucketScan, bucketJunk]
testingDataSamples = [X_Test]
# , MalignTrafficFrameUDP, MalignTrafficFrameTCP, MalignTrafficFrameScan, MalignTrafficFrameJunk]

for CorrectTestingValues, TestingDataForPrediction in zip(correctValues, testingDataSamples):
    # print("Detailed Classification Report:")
    print()

    y_true, y_pred = CorrectTestingValues, clf.predict(TestingDataForPrediction)
    print(classification_report(y_true, y_pred))
    print()

    print()
    print(confusion_matrix(CorrectTestingValues, y_pred))

    TotalTestSamples = 0
    correctTestSamples = 0

    for p, t in zip(CorrectTestingValues, y_pred):
        TotalTestSamples += 1
        if p == t:
            correctTestSamples = correctTestSamples + 1

    print()
    print((correctTestSamples / TotalTestSamples) * 100, "% correct")

    # y_true, y_pred = bucketsUDP, clf.predict(MalignTrafficFrameUDP)
