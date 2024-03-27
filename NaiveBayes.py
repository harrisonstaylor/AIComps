import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, f1_score

df = pd.read_csv('student_data.csv')

category_columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
                    'guardian',
                    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                    'higher',
                    'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

mapping_functions = dict()

# Fit Columns
for column in category_columns:
    values = df[column].unique()
    mapping_function = dict()
    for value_idx, value in enumerate(values):
        mapping_function[value] = value_idx

    mapping_functions[column] = mapping_function

# Apply Mapping
for column in category_columns:
    df[column] = df[column].map(mapping_functions[column])

# Isolate features and target Performance
features = df.drop(['Performance'], axis=1)
target = df['Performance']

# Split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

best_num_features = None
best_f1_score = 0

for num_features_to_select in range(1, len(category_columns) + 1):
    # Select best feature set for each iteration
    selector = SelectKBest(f_classif, k=num_features_to_select)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    naive_bayes_model = GaussianNB()
    # Fit training data to NB Model
    naive_bayes_model.fit(X_train_selected, y_train)

    # Test against testing data and compute F1
    y_pred = naive_bayes_model.predict(X_test_selected)
    f1_selected = f1_score(y_test, y_pred, average='weighted')

    # Save best F1 and number of features
    if f1_selected > best_f1_score:
        best_num_features = num_features_to_select
        best_f1_score = f1_selected

print(f'Best Number of Features: {best_num_features}')
print(f'Weighted F1 Score (Selected Features): {best_f1_score}')
