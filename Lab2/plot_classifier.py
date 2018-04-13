import svm_basic

def plot_fit(fit_line, datamatrix, labelmatrix):
    import matplotlib.pyplot as plt
    import numpy as np

    weights = fit_line

    dataarray = np.asarray(datamatrix)
    n = dataarray.shape[0]

    # Keep track of the two classes in different arrays so they can be plotted later...
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmatrix[i]) == 1:
            xcord1.append(dataarray[i, 0])
            ycord1.append(dataarray[i, 1])
        else:
            xcord2.append(dataarray[i, 0])
            ycord2.append(dataarray[i, 1])
    fig = plt.figure()

    # Plot the data as points with different colours
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # Plot the best-fit line
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


line = []
dataMatrix, labelMat = svm_basic.loadDataSet('data/linearly_separable.csv')
clf = svm_basic.svmClassifier(b=None, alphas=None, C=200, toler=0.0001, maxIter=1000)
clf = clf.fit(dataMatrix, labelMat)
print("Classifier after fit", "b=",  clf.b, "Alpha=", clf.alphas)
hyp = clf.predict(dataMatrix)
print("Hypothesis=", hyp)
line.append(clf.b.getA()[0])
line.append(clf.weights[0])
line.append(clf.weights[1])
plot_fit(line, dataMatrix, labelMat)

