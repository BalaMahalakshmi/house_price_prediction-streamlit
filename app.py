
# # app.py (Updated for No City Feature)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # --- Configuration ---
# MODEL_FILENAME = 'house_price_model.pkl' # <-- New model file name

# # Features must match the training script
# FEATURES = [
#     'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
#     'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
#     'yr_built', 'yr_renovated'
# ]

# # --- Load Components ---
# @st.cache_resource
# def load_components():
#     """Loads the trained model."""
#     try:
#         model = joblib.load(MODEL_FILENAME)
#         return model
#     except FileNotFoundError as e:
#         st.error(f"Error loading required file: {e.filename}. Please run 'python train_and_save_no_city.py' first.")
#         return None
#     except Exception as e:
#         st.error(f"An unexpected error occurred during loading: {e}")
#         return None

# def make_prediction(model, new_data_dict):
#     """Converts input to DataFrame and makes a prediction."""
    
#     # 1. Ensure input features are in the correct order
#     input_features = {k: new_data_dict[k] for k in FEATURES}
    
#     # 2. Convert input to DataFrame
#     X_new = pd.DataFrame([input_features])
    
#     # 3. Predict
#     prediction = model.predict(X_new)[0]
    
#     return prediction

# # --- STREAMLIT UI ---

# def main():
#     st.set_page_config(
#         page_title="RF House Price Predictor (No City)",
#         layout="wide"
#     )

#     st.title("🏠 House Price Prediction (Random Forest - No Location)")
#     st.markdown("Enter the house features below. This model uses only **numerical and structural features** for prediction.")
    
#     # Load Model
#     model = load_components()
    
#     if model is None:
#         return # Exit if loading failed

#     st.divider()

#     # --- Interactive Prediction Form ---
#     st.header("Input Features")
    
#     with st.form("prediction_form"):
#         # Use two columns for a clean layout
#         c1, c2 = st.columns(2)
        
#         with c1:
#             st.subheader("Structure & Size")
#             bedrooms = st.slider("Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
#             bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.5, step=0.5)
#             floors = st.slider("Floors", min_value=1.0, max_value=3.5, value=2.0, step=0.5)
#             sqft_living = st.number_input("Sqft Living", min_value=500, max_value=15000, value=2000)
#             sqft_above = st.number_input("Sqft Above", min_value=500, max_value=10000, value=1500)
#             sqft_basement = st.number_input("Sqft Basement", min_value=0, max_value=5000, value=500)


#         with c2:
#             st.subheader("Condition & Age")
#             condition = st.slider("Condition (1=poor, 5=excellent)", min_value=1, max_value=5, value=3, step=1)
#             view = st.slider("View (0=none, 4=excellent)", min_value=0, max_value=4, value=0, step=1)
#             waterfront = st.selectbox("Waterfront", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
#             sqft_lot = st.number_input("Sqft Lot (Land Area)", min_value=1000, max_value=1000000, value=10000)
#             yr_built = st.number_input("Year Built", min_value=1900, max_value=2015, value=1980, step=1)
#             yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2015, value=0)
        
#         submitted = st.form_submit_button("Predict House Price")

#     if submitted:
#         # Create a dictionary of the input features
#         new_data_dict = {
#             'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
#             'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront,
#             'view': view, 'condition': condition, 'sqft_above': sqft_above,
#             'sqft_basement': sqft_basement, 'yr_built': yr_built, 
#             'yr_renovated': yr_renovated
#         }
        
#         try:
#             prediction = make_prediction(model, new_data_dict)
            
#             st.success("The predicted price for this house is:")
#             st.metric(label="Predicted House Price", value=f"${prediction:,.2f}")
            
#         except Exception as e:
#             st.error(f"An error occurred during prediction. Error: {e}")
            
#     st.sidebar.markdown("---")
#     st.sidebar.info(f"Model File: `{MODEL_FILENAME}`\n\nFeatures Used: {', '.join(FEATURES)}")


# if __name__ == '__main__':
#     main()


# app.py (Corrected)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_FILENAME = 'house_price_model.pkl' # Matches the training script

# Features MUST match the 5 features used in the training script
FEATURES = ['bedrooms', 'sqft_lot', 'sqft_living', 'bathrooms', 'floors']

# --- Load Components ---
@st.cache_resource
def load_components():
    """Loads the trained model."""
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except FileNotFoundError as e:
        st.error(f"Error loading required file: {e.filename}. Please run your training script (the large Python block) first to create '{MODEL_FILENAME}'.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during loading: {e}")
        return None

def make_prediction(model, new_data_dict):
    """
    Converts input to DataFrame, makes a log-price prediction, 
    and applies the inverse log transformation (np.expm1).
    """
    
    # 1. Ensure input features are in the correct order and match the model
    # We only take the 5 features the model was trained on.
    input_features = {k: new_data_dict[k] for k in FEATURES} 
    
    # 2. Convert input to DataFrame
    X_new = pd.DataFrame([input_features])
    
    # 3. Predict the LOG-PRICE
    y_pred_log = model.predict(X_new)[0]
    
    # 4. Apply Inverse Transformation (np.expm1) to get the actual price
    prediction = np.expm1(y_pred_log)
    
    return prediction

# --- STREAMLIT UI ---

def main():
    st.set_page_config(
        page_title="RF House Price Predictor (5 Features)",
        layout="wide"
    )

    st.title("🏠 House Price Prediction (Random Forest Pipeline)")
    st.markdown("This model was trained using only **5 key features** and log-transformed price.")
    
    # Load Model
    model = load_components()
    
    if model is None:
        return 

    st.divider()

    # --- Interactive Prediction Form ---
    st.header("Input Features")
    
    with st.form("prediction_form"):
        # Use two columns for a clean layout
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Key Structure Inputs")
            bedrooms = st.slider("Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
            bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.5, step=0.5)
            floors = st.slider("Floors", min_value=1.0, max_value=3.5, value=2.0, step=0.5)
            

        with c2:
            st.subheader("Key Size Inputs (Sq. Ft.)")
            sqft_living = st.number_input("Sqft Living (Total Area)", min_value=500, max_value=15000, value=2000)
            sqft_lot = st.number_input("Sqft Lot (Land Area)", min_value=1000, max_value=1000000, value=10000)
            # The remaining features are removed to match the model's training data.
        
        submitted = st.form_submit_button("Predict House Price")

    if submitted:
        # Create a dictionary of the input features
        new_data_dict = {
            'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
            'sqft_lot': sqft_lot, 'floors': floors
        }
        
        try:
            prediction = make_prediction(model, new_data_dict)
            
            st.success("The predicted price for this house is:")
            st.metric(label="Predicted House Price", value=f"${prediction:,.2f}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs and ensure the model was trained successfully.")
            
    st.sidebar.markdown("---")
    # Using the metrics that would have been printed by the successful training run
    st.sidebar.info(
        "**Model Details**\n\n"
        "Model: Random Forest Regressor (Pipeline)\n\n"
        f"Features Used: {', '.join(FEATURES)}\n\n"
        "**Training Metrics** (from your execution)\n\n"
        "R2 Score: $0.7816$ (approx)\n\n"
        "MAE: $\$100,600.86$ (approx)"
    )


if __name__ == '__main__':
    main()