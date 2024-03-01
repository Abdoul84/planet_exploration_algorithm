### ML_Kepler: Exoplanet Classification Model

#### Overview
This script is crafted for the classification of candidate exoplanets from a dataset, utilizing Support Vector Classification (SVC). It incorporates GridSearchCV for optimizing the SVC model's parameters, aiming to enhance accuracy in predicting potential exoplanets.

#### Requirements
- Python 3.6+
- scikit-learn
- numpy (for dataset preprocessing, if necessary)
- TensorFlow (for deep learning model utilities)

#### Setup
1. Install Python on your system.
2. Install the necessary Python packages using pip:
   ```sh
   pip install scikit-learn numpy tensorflow
   ```

#### Highlighted Imports
Before diving into the script's operation, it's important to note some key Python imports that play a significant role in data preparation:

- `from sklearn.model_selection import train_test_split`: Splits the dataset into training and testing sets.
- `from sklearn.preprocessing import LabelEncoder, MinMaxScaler`: These are used for preprocessing labels and scaling features respectively, ensuring that your model receives data in a format it can work with effectively.
- `from tensorflow.keras.utils import to_categorical`: Converts class vectors to binary class matrices, critical for models that output predictions across multiple categories.

#### How to Run
1. Ensure your dataset is segmented into features (X) and targets (yy).
2. Save the script in the directory where your dataset is located.
3. Execute the script with the following command:
   ```sh
   python exoplanet_classification.py
   ```

#### Script Details
- **Data Preprocessing**: Utilizes `LabelEncoder` for labels and `MinMaxScaler` for feature scaling. This is crucial for models to perform optimally.
- **Model Training**: Defines a parameter grid for the SVC model, adjusting values for 'C', 'gamma', and 'kernel'.
- **Optimization**: Employs GridSearchCV to discover the best parameters for the SVC model.
- **Prediction**: Fits the model to the training data, enabling it to predict new candidate exoplanets accurately.

#### Note
Make sure your dataset is preprocessed before running this script. Adjust the `param_grid` to explore different parameters for the SVC model to potentially enhance the model's prediction accuracy further.