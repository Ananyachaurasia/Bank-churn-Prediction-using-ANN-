# Bank-churn-Prediction-using-ANN

A deep learning project that predicts whether a bank customer is likely to churn, built with TensorFlow/Keras and deployed as an interactive web app using Streamlit.

## Project Structure
```
├── datapreprocessing.ipynb   # Data preprocessing & ANN model training
├── prediction.ipynb          # Single customer churn prediction (testing)
├── app.py                    # Streamlit web application
├── model.h5                  # Trained Keras model
├── scaler.pkl                # Fitted StandardScaler
├── label_encoder_gender.pkl  # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl    # Fitted OneHotEncoder for Geography
└── Churn_Modelling.csv       # Dataset
```


## Model Architecture

The ANN model is built with TensorFlow/Keras:

- **Input Layer:** 12 features (after encoding)
- **Hidden Layer 1:** 64 neurons, ReLU activation
- **Hidden Layer 2:** 32 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation (binary classification)
- **Optimizer:** Adam (learning rate = 0.01)
- **Loss Function:** Binary Crossentropy
- **Callbacks:** EarlyStopping (patience=10), TensorBoard

##  Dataset

The model is trained on the [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) with the following features:

| Feature | Description |
|---|---
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Active member status (0/1) |
| EstimatedSalary | Estimated annual salary |

**Target:** `Exited` — whether the customer churned (1) or not (0)

##  Data Preprocessing

Steps performed in `datapreprocessing.ipynb`:

1. Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
2. Label Encoding for `Gender`
3. One-Hot Encoding for `Geography`
4. Train/test split (80/20)
5. Feature scaling using `StandardScaler`
6. Saved encoders and scaler as `.pkl` files

##  Streamlit App

The web app (`app.py`) lets users input customer details through an interactive UI and get an instant churn probability prediction.

**Input fields:**
- Geography, Gender, Age, Balance, Credit Score
- Estimated Salary, Tenure, Number of Products
- Has Credit Card, Is Active Member

**Output:**
- Churn probability (0.00 – 1.00)
- Plain-language prediction: *"The customer is likely to churn"* or *"not likely to churn"*

## Installation & Setup

1. **Clone the repository**
```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the app**
```bash
   streamlit run app.py
```


## Training Monitoring

TensorBoard is integrated for tracking training metrics. To launch:
```bash
tensorboard --logdir logs/fit
```

## Notes

- The trained model (`model.h5`) and all preprocessing artifacts (`.pkl` files) must be present in the root directory for `app.py` to run.
- The model uses a threshold of **0.5** to classify churn vs. no churn.
