import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from google.colab import files
store=files.upload()
import io
import pandas  as pd
df=pd.read_excel(io.BytesIO(store['BreastCancer.xlsx']))
print(df)

print(df.head()) 
print(df.info()) 
print(df.describe()) 

print(df.isnull().sum())

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

sns.countplot(x='diagnosis', data=df)
plt.title('Distribution of Diagnosis')
plt.show()

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

features = df.columns[2:] 
target = 'diagnosis'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
    voting='soft'
)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(estimator=SVC(probability=True, random_state=42), param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
print("Best parameters for SVC:", grid_search.best_params_)


best_svc = grid_search.best_estimator_
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', best_svc)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

rf.fit(X_train, y_train)
feature_importances = pd.Series(rf.feature_importances_, index=features)
feature_importances.nlargest(10).plot(kind='barh', title='Top 10 Important Features')
plt.show()

