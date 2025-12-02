# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/bhumitps/amlops/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# --- Impute Missing Values ---
# Impute missing 'Age' with the median
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Impute missing 'MonthlyIncome' with the median
median_income = df['MonthlyIncome'].median()
df['MonthlyIncome'].fillna(median_income, inplace=True)

# Impute missing 'NumberOfChildrenVisiting' with the mode
mode_children = df['NumberOfChildrenVisiting'].mode()[0]
df['NumberOfChildrenVisiting'].fillna(mode_children, inplace=True)

# Impute missing 'PreferredPropertyStar' with the mode
mode_property_star = df['PreferredPropertyStar'].mode()[0]
df['PreferredPropertyStar'].fillna(mode_property_star, inplace=True)

# Impute missing 'TypeofContact' with mode
mode_typeofcontact = df['TypeofContact'].mode()[0]
df['TypeofContact'].fillna(mode_typeofcontact, inplace=True)

# Impute missing 'Gender' with mode
mode_gender = df['Gender'].mode()[0]
df['Gender'].fillna(mode_gender, inplace=True)

# Impute missing 'DurationOfPitch' with median
median_duration = df['DurationOfPitch'].median()
df['DurationOfPitch'].fillna(median_duration, inplace=True)

# --- Encode Categorical Features ---
# Label Encoding for binary categorical features (Gender: Male/Female)
label_encoder_gender = LabelEncoder()
# Fit and transform, including 'Other' which might be present
df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

# One-Hot Encoding for multi-class categorical features
categorical_cols_ohe = [
    'Occupation',
    'Designation',
    'MaritalStatus',
    'ProductPitched',
    'TypeofContact'
]

df = pd.get_dummies(df, columns=categorical_cols_ohe, drop_first=True)

# Target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform stratified train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload the split files to Hugging Face
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="bhumitps/amlops",
        repo_type="dataset",
    )

print("Data preparation complete and files uploaded to Hugging Face.")
