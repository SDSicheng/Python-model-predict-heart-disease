import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('/Users/sichengyu/Desktop/CS677 Final project/heart.csv')

# Step 1: Data Cleaning and Preprocessing
# Checking for any missing values in the dataset
missing_values = df.isnull().sum()

# Splitting the dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# There are no missing values, so we proceed to scaling the features
# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Feature Selection
# Using SelectKBest to select top features based on ANOVA F-test
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X_scaled, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

# Concat two dataframes for better visualization and display the best features
featureScores = pd.concat([df_columns,df_scores],axis=1)
featureScores.columns = ['Feature','Score']

# We will select features with the highest score to use in our model
selected_features = featureScores.nlargest(10, 'Score')['Feature'].values

# Updating X to include only the selected features
X_selected = X[selected_features]
# 2graph
plt.figure(figsize=(12,8))
sns.barplot(x='Score', y='Feature', data=featureScores.sort_values(by="Score", ascending=False))
plt.title('Feature Importance based on ANOVA F-test Scores')
plt.show()

# Step 3: Train-Test Split
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)

# Step 4: Applying Bayes' Theorem - Gaussian Naive Bayes
# Fitting the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Step 5: Probability Calculation
# Predicting the Test set results
y_pred = gnb.predict(X_test)

# Calculating the accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(missing_values, featureScores.sort_values(by="Score", ascending=False), accuracy, report)

# 5 confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title('Confusion Matrix with Gradient Color')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 5 Receiver Operating Characteristic
# Calculate the false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(y_test, gnb.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()






