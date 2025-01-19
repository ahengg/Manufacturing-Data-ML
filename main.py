import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
scaler = StandardScaler()
pca = PCA(n_components=20)  # Replace with actual PCA components used
model_path = 'multi_output_nn_model.pth'

# Initialize model (adjust input and output sizes as needed)
input_size = 20  # Match PCA output size
output_size = 16  # Number of targets
model = MultiOutputNN(input_size, output_size)
model.load_state_dict(torch.load("multi_output_nn_model.pth", map_location=device))

# Move the model to the detected device
model.to(device)

model.eval()

# Define feature list
features =['timeunit', 'storageCost', 'interestRate', 'compidx0lt2', 'compidx1lt2', 'compidx2lt2', 'compidx3lt2', 'compidx4lt2', 'compidx5lt2', 'compidx6lt2', 'compidx7lt2', 'compidx8lt2', 'compidx9lt2', 'compidx10lt2', 'compidx11lt2', 'compidx12lt2', 'compidx13lt2', 'compidx14lt2', 'compidx15lt2', 'compidx0lt6', 'compidx1lt6', 'compidx2lt6', 'compidx3lt6', 'compidx4lt6', 'compidx5lt6', 'compidx6lt6', 'compidx7lt6', 'compidx8lt6', 'compidx9lt6', 'compidx10lt6', 'compidx11lt6', 'compidx12lt6', 'compidx13lt6', 'compidx14lt6', 'compidx0lt10', 'compidx1lt10', 'compidx2lt10', 'compidx3lt10', 'compidx4lt10', 'compidx5lt10', 'compidx6lt10', 'compidx7lt10', 'compidx8lt10', 'compidx9lt10', 'compidx10lt10', 'compidx11lt10', 'compidx12lt10', 'compidx13lt10', 'compidx14lt10', 'compidx15lt10', 'compidx0lt20', 'compidx1lt20', 'compidx2lt20', 'compidx3lt20', 'compidx4lt20', 'compidx5lt20', 'compidx6lt20', 'compidx7lt20', 'compidx8lt20', 'compidx9lt20', 'compidx10lt20', 'compidx11lt20', 'compidx12lt20', 'compidx13lt20', 'compidx0lt30', 'compidx1lt30', 'compidx2lt30', 'compidx3lt30', 'compidx4lt30', 'compidx5lt30', 'compidx6lt30', 'compidx7lt30', 'compidx8lt30', 'compidx9lt30', 'compidx10lt30', 'compidx11lt30', 'compidx12lt30', 'compidx13lt30', 'compidx14lt30', 'sku1', 'sku2', 'sku9', 'sku10', 'sku11', 'sku12', 'sku13', 'sku14', 'sku15', 'sku16', 'demandseg1', 'demandseg2', 'demandseg3']


# Streamlit app layout
st.title("Neural Network Model Prediction")
st.markdown("Provide input values for the features and get predictions.")

# User input form
st.markdown("### Input Features")
input_data = {feature: st.number_input(f"{feature}", value=0.0) for feature in features}

# Prediction logic
if st.button("Predict"):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale input data
    input_scaled = scaler.fit_transform(input_df)

    # Reduce dimensions using PCA
    input_pca = pca.fit_transform(input_scaled)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_pca, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor).numpy()

    # Display predictions
    st.markdown("### Predictions")
    predictions_df = pd.DataFrame(predictions, columns=[f"Target {i+1}" for i in range(output_size)])
    st.dataframe(predictions_df)

    # Option to download predictions
    csv = predictions_df.to_csv(index=False)
    st.download_button(label="Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
