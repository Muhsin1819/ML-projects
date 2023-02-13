#load libraries
import pickle
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
df = read_csv(url, names=names)

#visualization of the dataset
#univariate plots
#box and whisker plots
df.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

#histogram
df.hist()
pyplot.show()

#multivariate plot
#scatterplot
scatter_matrix(df)
pyplot.show()

#seperate validation dataset
array = df.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

#Evaluating algorithms
models = []
models.append(('LR', LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(('LDR', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma="auto")))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    names.append(name)
    results.append(cv_results)
    print(f"{name}: {'{:.6f}'.format(cv_results.mean())} ({'{:.6f}'.format(cv_results.std())})")

#comparing algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Evaluating best model
model = SVC(gamma="auto")
model.fit(X_train, y_train)
filename = "finalized_model.pickle"
pickle.dump(model, open(filename, "wb"))
#model = pickle.load(open(filename, "rb"))
predictions = model.predict(X_test)

print(accuracy_score(predictions, y_test))
print(confusion_matrix(predictions, y_test))
print(classification_report(predictions, y_test))