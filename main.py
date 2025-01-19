import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import nn
import pickle

# Detect device (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

# Define the neural network class
class MultiOutputNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiOutputNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load pre-trained model components
# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load PCA
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Initialize and load the trained model
input_size = pca.n_components_  # Match PCA output size
output_size = 16  # Number of targets
model = MultiOutputNN(input_size, output_size)
model.load_state_dict(torch.load("multi_output_nn_model.pth", map_location=device))
model.to(device)
model.eval()

# Define feature list
features = ['sku16', 'sku12', 'sku2', 'sku1', 'sku15', 'sku13', 'sku14', 'sku9', 'sku11',
            'compidx1lt10', 'compidx1lt20', 'timeunit', 'sku10', 'demandseg3', 'compidx1lt6',
            'demandseg2', 'compidx1lt30', 'compidx10lt30', 'demandseg1', 'compidx1lt2',
            'compidx9lt30', 'compidx0lt30', 'compidx14lt30', 'compidx6lt30', 'compidx11lt30']

# Streamlit app layout
st.title("Neural Network Model Prediction")
st.markdown("Provide input values for the features and get predictions.")

# User input form
st.markdown("### Input Features")
input_data = {feature: st.number_input(f"{feature}", value=0.0) for feature in features}

# Prediction logic
if st.button("Predict"):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale input data
        input_scaled = scaler.transform(input_df)

        # Reduce dimensions using PCA
        input_pca = pca.transform(input_scaled)

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_pca, dtype=torch.float32).to(device)

        # Make predictions
        with torch.no_grad():
            predictions = model(input_tensor).cpu().numpy()

        # Display predictions
        st.markdown("### Predictions")
        predictions_df = pd.DataFrame(predictions, columns=[f"Target {i+1}" for i in range(output_size)])
        st.dataframe(predictions_df)

        # Option to download predictions
        csv = predictions_df.to_csv(index=False)
        st.download_button(label="Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
