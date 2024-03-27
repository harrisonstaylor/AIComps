import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

df = pd.read_csv('student_data.csv')

label_encoder = LabelEncoder()
category_columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
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

# Remove Performance from features and make it the target
features = df.drop(['Performance'], axis=1)
target = df['Performance']

#Training testing split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Use a scaler for KNN as the model relies on distance between datapoints, so all columns should be scaled roughly the same
# This is important to the accuracy of KNN specifically
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
best_k = None
best_num_features = None
best_f1 = 0

# Test neighbors in range of 1 to 50
for k_value in range(1, 50):
    # Test features in range of 1 to all
    for num_features in range(1, len(category_columns) + 1):
        #Use selector for best features
        selector = SelectKBest(f_classif, k=num_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        # Train knn model
        knn_model = KNeighborsClassifier(n_neighbors=k_value)
        knn_model.fit(X_train_selected, y_train)

        # Predict with testing data and calculate F1 score
        y_pred_selected = knn_model.predict(X_test_selected)

        f1 = f1_score(y_test, y_pred_selected, average='weighted')

        #Save best F1 score, number of neighbors, and feature count
        if f1 > best_f1:
            best_k = k_value
            best_num_features = num_features
            best_f1 = f1

print(f'Best Number of Neighbors (K): {best_k}')
print(f'Best Number of Features: {best_num_features}')
print(f'Weighted F1 Score: {best_f1}')
