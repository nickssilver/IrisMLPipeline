# Machine Learning Pipeline Example

This repository demonstrates the creation of a machine learning pipeline using scikit-learn with the Random Forest Regression algorithm. The pipeline includes data preprocessing, model training, and evaluation. The sources used for creating this example are as follows:

   - [turing.com: Building an ML Pipeline in Python with scikit-learn]()
   - [freecodecamp.org: Machine Learning Pipeline]()
   - [towardsdatascience.com: Building a Machine Learning Pipeline]()
   - [analyticsvidhya.com: Build your first Machine Learning Pipeline using scikit-learn]()

## Getting Started

   1. Clone this repository to your local machine.
   2. Install the required packages:

pip install scikit-learn pandas numpy

    Run the pipeline script:

python src/pipeline.py

## Pipeline Design

The pipeline is designed in three stages:

    Data preprocessing: The dataset is cleaned by dropping unnecessary columns, filling missing values, and encoding categorical features.

    Model training: The preprocessed data is split into training and testing sets, and a Random Forest Regression model is trained using the training set.

    Model evaluation: The trained model is evaluated on the testing set to measure its performance.

## Data Preprocessing

The data preprocessing stage includes the following steps:

   - Dropping unused columns: df.drop(['record_id', 'casual', 'registered', 'datetime', 'temp'], axis=1, inplace=True)
   - Creating pipelines for numerical and categorical features using Pipeline(steps=[('step name', transform function), …])
   - Filling missing values with SimpleImputer
   - Scaling numerical features with MinMaxScaler
   - Encoding categorical features with OneHotEncoder(handle_unknown='ignore')

## Model Training

In this stage, the preprocessed data is split into training and testing sets, and a Random Forest Regression model is trained using the training set. The pipeline is built using Pipeline(steps=[('scaler', StandardScaler()), ('classifier', RandomForestClassifier())]), and the model is trained with the fit() method.
Model Evaluation

The trained model is evaluated on the testing set using accuracy_score and balanced_accuracy_score from scikit-learn's metrics module. The results are printed to the console.
### Authors

 [Nicks M. Gitobu, Software Engineer]()

License

This project is licensed under the **MIT License**.
