
# ML_Kepler: Exoplanet Classification Model

## Overview

The **ML_Kepler** project is designed to classify candidate exoplanets using machine learning techniques. Specifically, it leverages a Support Vector Classification (SVC) model and deep learning utilities from TensorFlow to enhance the prediction accuracy. The model is optimized through GridSearchCV, allowing for fine-tuning of parameters to achieve the best possible performance in identifying potential exoplanets.

## Requirements

- Python 3.6+
- scikit-learn
- numpy (for dataset preprocessing, if necessary)
- TensorFlow (for deep learning model utilities)

## Setup

1. Ensure Python is installed on your system.
2. Install the necessary Python packages using pip:
   ```sh
   pip install scikit-learn numpy tensorflow
   ```

## Key Script Details

### Data Preparation and Preprocessing

The script begins by reading the exoplanet dataset (`exoplanet_data.csv`) and performing basic data cleaning. Null columns and rows are removed to ensure the dataset is clean and ready for analysis.

- **Feature Selection**: The script identifies and selects key features relevant to the classification task. These features include astrophysical parameters such as period errors, impact errors, and stellar temperatures, among others.

- **Data Splitting**: The dataset is split into training and testing sets using `train_test_split`. This allows the model to learn from one portion of the data and validate its performance on another.

- **Preprocessing**: The features are scaled using `MinMaxScaler` to normalize the data, which is crucial for ensuring that the SVC model performs optimally. Additionally, the target labels are encoded using `LabelEncoder` and converted into a one-hot encoded format using TensorFlow's `to_categorical`, preparing the data for classification.

### Model Training

The classification model is built using a neural network with TensorFlow's `Sequential` API. The model architecture consists of:

- **Input Layer**: With 100 units and ReLU activation.
- **Hidden Layer**: Another 100 units with ReLU activation.
- **Output Layer**: 2 units with Softmax activation for binary classification.

The model is compiled with the Adam optimizer and categorical crossentropy loss function, and then trained over 40 epochs. The training process adjusts the model weights to minimize the loss and maximize accuracy.

### Evaluation and Hyperparameter Tuning

After training, the model is evaluated on the test set, yielding metrics for loss and accuracy. For further optimization, the script uses `GridSearchCV` to tune the SVC model's hyperparameters (`C`, `gamma`, and `kernel`). This process involves testing different parameter combinations to find the best set that maximizes classification accuracy.

### Saving the Model

Once the best model is identified, it is saved to disk using the `joblib` library for future use. This allows for easy loading and deployment of the trained model in other applications or environments.

## How to Run

1. Ensure your dataset is segmented into features (X) and targets (yy).
2. Save the script in the directory where your dataset is located.
3. Execute the script with the following command:
   ```sh
   python exoplanet_classification.py
   ```

## Findings

Through this approach, the SVC model, enhanced by TensorFlow's deep learning capabilities, was optimized to classify exoplanet candidates effectively. The use of GridSearchCV allowed fine-tuning of hyperparameters, significantly improving the model's performance. This methodology can be applied to similar classification tasks in astrophysics or other domains requiring robust machine learning solutions.
