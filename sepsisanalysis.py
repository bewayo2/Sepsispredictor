import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
dataset = pd.read_csv(r"C:\Users\timsi\OneDrive\Bewaji HealthCare Solutions\training\JAIA AI Masterclass in Health\sepsis.csv")

# Initial class distribution
print(dataset['SepsisLabel'].value_counts())
plt.pie(dataset['SepsisLabel'].value_counts(), labels=['0','1'], autopct='%1.1f%%', shadow=True)
plt.show()
sns.countplot(x=dataset['SepsisLabel'], label="Count")
plt.show()

# Upsample minority class
df_majority = dataset[dataset.SepsisLabel==0]
df_minority = dataset[dataset.SepsisLabel==1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print(df_upsampled.SepsisLabel.value_counts())
plt.pie(df_upsampled['SepsisLabel'].value_counts(), labels=['0','1'], autopct='%1.1f%%', shadow=True)
plt.show()
sns.countplot(x=df_upsampled['SepsisLabel'], label="Count")
plt.show()

# Prepare data
X = df_upsampled[df_upsampled.columns[0:40]].values
Y = df_upsampled[df_upsampled.columns[40:]].values.ravel()  # .ravel() to flatten

print("sepsis dimensions : {}".format(df_upsampled.shape))
print("sepsis dimensions without label : {}".format(X.shape))
print("sepsis dimensions only label : {}".format(Y.shape))

# Encode labels
labelencoder_Y = preprocessing.LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print("Training data dimensions :{}".format(X_train.shape))
print("Testing data dimensions :{}".format(X_test.shape))

# Classifiers
classifiers = [
    MLPClassifier(
        activation='tanh',
        solver='lbfgs',
        early_stopping=False,
        hidden_layer_sizes=(40,10,10,10,10,2),
        random_state=1,
        batch_size='auto',
        max_iter=13000,
        learning_rate_init=1e-5,
        tol=1e-4,
    ),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

mlp_clf = classifiers[0]
mlp_clf.fit(X_train, Y_train)
# Save the trained MLPClassifier
joblib.dump(mlp_clf, "sepsis_mlp_model.joblib")

for clf in classifiers:
    clf.fit(X_train, Y_train)
    name = clf.__class__.__name__
    print("="*30)
    print(name)
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(Y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    if hasattr(clf, "predict_proba"):
        train_predictions_proba = clf.predict_proba(X_test)
        ll = log_loss(Y_test, train_predictions_proba)
    else:
        ll = None
    print("Log Loss: {}".format(ll))
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = pd.concat([log, log_entry], ignore_index=True)

print("="*30)

# Bar Graphs
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()