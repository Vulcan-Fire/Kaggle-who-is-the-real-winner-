import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

def plot_party_criminal_cases_distribution(train_file_path):
    education_data = pd.read_csv(train_file_path)

    party_criminal_cases = education_data.groupby('Party')['Criminal Case'].sum()

    threshold = 41
    party_criminal_cases['Others'] = party_criminal_cases[party_criminal_cases < threshold].sum()
    party_criminal_cases = party_criminal_cases[party_criminal_cases >= threshold]

    # Plotting the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(party_criminal_cases, labels=party_criminal_cases.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.savefig('party_criminal_cases_distribution.jpg', format='jpg')

def plot_education_level_distribution(csv_file_path):
    education_data = pd.read_csv(csv_file_path)

    # Define education level encoding
    education_encoding = {
        'Doctorate': 15,
        'Post Graduate': 14,
        'Graduate Professional': 13,
        'Graduate': 12,
        '12th Pass': 11,
        '10th Pass': 10,
        '8th Pass': 8,
        '5th Pass': 5,
        'Literate': 1,
        'Others': 0
    }
    # Encode
    education_data['Education Level'] = education_data['Education'].replace(education_encoding)

    education_level_counts = education_data['Education Level'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(education_level_counts, labels=education_level_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')

    # Add legend
    legend_labels = {v: k for k, v in education_encoding.items()}
    legend_labels = [f"{value}: {key}" for key, value in legend_labels.items()]
    plt.legend(legend_labels, loc='lower right', bbox_to_anchor=(0.5, -0.05), fontsize='small')

    plt.savefig('party_net_worth_distribution.jpg', format='jpg')

def plot_party_net_worth_distribution(train_file_path):
    education_data = pd.read_csv(train_file_path)

    # Preprocessing
    education_data['Total Assets'] = education_data['Total Assets'].str.replace(' Crore+', 'e+7').str.replace(' Lac+', 'e+5').str.replace(' Thou+', 'e+3').str.replace(' Hund+', 'e+2')
    education_data['Liabilities'] = education_data['Liabilities'].str.replace(' Crore+', 'e+7').str.replace(' Lac+', 'e+5').str.replace(' Thou+', 'e+3').str.replace(' Hund+', 'e+2')
    education_data['Total Assets'] = pd.to_numeric(education_data['Total Assets'])
    education_data['Liabilities'] = pd.to_numeric(education_data['Liabilities'])

    # Net worth calculation
    education_data['Net Worth'] = education_data['Total Assets'] - education_data['Liabilities']

    # Sum up
    party_net_worth = education_data.groupby('Party')['Net Worth'].sum()

    threshold = 3000000000
    party_net_worth['Others'] = party_net_worth[party_net_worth < threshold].sum()
    party_net_worth = party_net_worth[party_net_worth >= threshold]

    # Plotting the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(party_net_worth, labels=party_net_worth.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.savefig('education_level_distribution.jpg', format='jpg')

def plot_state_percentage_higher_education(csv_file_path):
    education_data = pd.read_csv(csv_file_path)

    higher_education_candidates = education_data[education_data['Education'].isin(['12th Pass', 'Graduate', 'Post Graduate'])]
    state_counts = education_data['state'].value_counts()
    higher_education_state_counts = higher_education_candidates['state'].value_counts()

    # Calculate the percentage
    state_percentage = (higher_education_state_counts / state_counts) * 100

    # Sort
    state_percentage_sorted = state_percentage.sort_values()

    # Plotting the horizontal histogram
    plt.figure(figsize=(10, 10))
    state_percentage_sorted.plot(kind='barh')
    plt.xlabel('Percentage of Candidates with Higher Education')
    plt.ylabel('State')
    plt.savefig('state_percentage_higher_education.jpg', format='jpg')

def train_model_and_predict(train_file_path, test_file_path):
    # Read the training data
    education_data = pd.read_csv(train_file_path)

    # Preprocessing
    label_encoder1 = LabelEncoder()
    label_encoder2 = LabelEncoder()
    label_encoder3 = LabelEncoder()

    education_data['Party'] = label_encoder2.fit_transform(education_data['Party'])
    education_data['state'] = label_encoder3.fit_transform(education_data['state'])
    education_data['Total Assets'] = education_data['Total Assets'].str.replace(' Crore+', 'e+7').str.replace(' Lac+', 'e+5').str.replace(' Thou+', 'e+3').str.replace(' Hund+', 'e+2')
    education_data['Liabilities'] = education_data['Liabilities'].str.replace(' Crore+', 'e+7').str.replace(' Lac+', 'e+5').str.replace(' Thou+', 'e+3').str.replace(' Hund+', 'e+2')
    education_data['Total Assets'] = pd.to_numeric(education_data['Total Assets'])
    education_data['Liabilities'] = pd.to_numeric(education_data['Liabilities'])

    education_encoding = {
        'Doctorate': 15,
        'Post Graduate': 14,
        'Graduate Professional': 13,
        'Graduate': 12,
        '12th Pass': 11,
        '10th Pass': 10,
        '8th Pass': 8,
        '5th Pass': 5,
        'Literate': 3,
        'Others': 0
    }
    education_data['Education'] = education_data['Education'].replace(education_encoding)

    # Feature Engineering: Added Assets to Liabilities Ratio
    education_data['Assets_to_Liabilities_Ratio'] = education_data['Total Assets'] / (education_data['Liabilities'] + 1)  # +1 to avoid division by zero


    # Define features and target variable
    features = ['Party', 'Criminal Case', 'state', 'Liabilities', 'Total Assets', 'Assets_to_Liabilities_Ratio']
    X = education_data[features]
    y = education_data['Education']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    #Grid_Search for best params
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': range(1, 31), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'chebyshev']}
    clf = GridSearchCV(knn, parameters, cv=5)

    # Fit the model
    clf.fit(X_scaled, y)
    print("Best parameters:", clf.best_params_)
    print("Cross-validated score:", clf.best_score_)

    # Read the test data
    test_data = pd.read_csv(test_file_path)

    # Preprocess test data
    test_data['Party'] = label_encoder2.transform(test_data['Party'])
    test_data['state'] = label_encoder3.transform(test_data['state'])
    test_data['Total Assets'] = test_data['Total Assets'].str.replace(' Crore+', 'e+7').str.replace(' Lac+', 'e+5').str.replace(' Thou+', 'e+3').str.replace(' Hund+', 'e+2')
    test_data['Liabilities'] = test_data['Liabilities'].str.replace(' Crore+', 'e+7').str.replace(' Lac+', 'e+5').str.replace(' Thou+', 'e+3').str.replace(' Hund+', 'e+2')
    test_data['Total Assets'] = pd.to_numeric(test_data['Total Assets'])
    test_data['Liabilities'] = pd.to_numeric(test_data['Liabilities'])

    #Added Assets to Liability Ratio for test_data
    test_data['Assets_to_Liabilities_Ratio'] = test_data['Total Assets'] / (test_data['Liabilities'] + 1)  # +1 to avoid division by zero

    # Scale test features
    knn.fit(X_scaled, y)

    # Scaled and made predictions
    X_test_scaled = scaler.transform(test_data[features])
    predictions = knn.predict(X_test_scaled)


   # test_data['Education'] = np.where(test_data['Candidate'].str.contains('Dr. ', case=False), 'Doctorate', predictions)
    education_decoding = {v: k for k, v in education_encoding.items()}
    decoded_predictions = [education_decoding[int(prediction)] for prediction in predictions]
    predicted_df = pd.DataFrame({'ID': test_data['ID'], 'Education': decoded_predictions})

    for idx, candidate in enumerate(test_data['Candidate']):
      if 'Adv. ' in candidate:
          predicted_df.loc[idx, 'Education'] = 'Graduate Professional'

    arr=['8th Pass','9th Pass','10th Pass','12th Pass','Literate','Others']
    
    for idx, candidate in enumerate(test_data['Candidate']):
      if 'Dr. ' in candidate:
        if predicted_df.loc[idx, 'Education'] in arr:
          predicted_df.loc[idx, 'Education'] = 'Doctrate'
    
    return predicted_df
    

# Train the model and make predictions
predicted_df = train_model_and_predict('/content/train.csv', '/content/test.csv')

#store in csv file
predicted_df.to_csv('predicted_education.csv', index=False)

#plots
plot_party_criminal_cases_distribution('/content/train.csv')
plot_party_net_worth_distribution('/content/train.csv')
plot_education_level_distribution('/content/train.csv')
plot_state_percentage_higher_education('/content/train.csv')
