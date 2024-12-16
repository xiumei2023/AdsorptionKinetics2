import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
import streamlit as st
from io import BytesIO

# Streamlit Title
st.title("Kinetics Data Analysis Interface")

# Upload Instructions
st.write("""
**Instructions**:
1. Upload an Excel file with the following format:
    - **First row**: Column names (e.g., `time(min)` and `qt(mg/g)`).
    - **First column**: Time values (`time(min)`).
    - **Second column**: Adsorption data (`qt(mg/g)`).
2. Ensure there are no blank rows or formatting errors.
3. The application will process all sheets in the uploaded file.
""")

# Define pseudo-first-order and pseudo-second-order models
def pseudo_first_order(t, q_e, k1):
    return q_e * (1 - np.exp(-k1 * t))

def pseudo_second_order(t, q_e, k2):
    return (q_e**2 * k2 * t) / (1 + q_e * k2 * t)

# Kinetics Fitting Function
def run_kinetic_fitting(uploaded_file):
    results_list = []
    figure_paths = []  # Store in-memory figures
    combined_data = []  # Combine original data with fitting results

    # Load Excel file
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names

    # Loop through each sheet
    for sheet in sheet_names:
        try:
            # Load and clean data
            data = pd.read_excel(uploaded_file, sheet_name=sheet, skiprows=1, usecols=[0, 1])
            data.columns = ['time(min)', 'qt(mg/g)']
            t_data = data['time(min)'].dropna().values
            q_data = data['qt(mg/g)'].dropna().values

            if len(t_data) == 0 or len(q_data) == 0:
                st.warning(f"No valid data found in sheet '{sheet}'. Skipping...")
                continue

            # Model setup
            first_order_model = Model(pseudo_first_order)
            first_order_params = first_order_model.make_params(q_e=np.max(q_data), k1=0.1)

            second_order_model = Model(pseudo_second_order)
            second_order_params = second_order_model.make_params(q_e=np.max(q_data), k2=0.001)

            # Fit models and create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(t_data, q_data, 'bo', label="Data")

            # Prepare combined DataFrame
            combined_df = pd.DataFrame({"time(min)": t_data, "qt(mg/g)": q_data})

            # Fit pseudo-first-order
            first_order_result = first_order_model.fit(q_data, t=t_data, params=first_order_params)
            ax.plot(t_data, first_order_result.best_fit, 'r-', label="1st Order Fit")
            r_squared_1st = 1 - first_order_result.residual.var() / np.var(q_data)
            results_list.append({
                'Sheet': sheet, 'Model': 'Pseudo First Order',
                'q_e': first_order_result.params['q_e'].value,
                'k': first_order_result.params['k1'].value,
                'R^2': r_squared_1st
            })
            combined_df['Pseudo 1st Order'] = first_order_result.best_fit

            # Fit pseudo-second-order
            second_order_result = second_order_model.fit(q_data, t=t_data, params=second_order_params)
            ax.plot(t_data, second_order_result.best_fit, 'g--', label="2nd Order Fit")
            r_squared_2nd = 1 - second_order_result.residual.var() / np.var(q_data)
            results_list.append({
                'Sheet': sheet, 'Model': 'Pseudo Second Order',
                'q_e': second_order_result.params['q_e'].value,
                'k': second_order_result.params['k2'].value,
                'R^2': r_squared_2nd
            })
            combined_df['Pseudo 2nd Order'] = second_order_result.best_fit

            # Add sheet column for clarity
            combined_df['Sheet'] = sheet
            combined_data.append(combined_df)

            # Finalize figure
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("qt (mg/g)")
            ax.legend()
            ax.set_title(f"Kinetics Fits for {sheet}")

            # Save figure to memory
            fig_io = BytesIO()
            plt.savefig(fig_io, format="png")
            plt.close(fig)
            figure_paths.append(fig_io)

        except Exception as e:
            st.error(f"Error processing sheet '{sheet}': {e}")
            continue

    # Combine all data
    summary_df = pd.DataFrame(results_list)
    combined_export = pd.concat(combined_data, axis=0)
    return summary_df, figure_paths, combined_export

# Streamlit File Upload
uploaded_file = st.file_uploader("Upload Excel File (Kinetics Data)", type=["xlsx"])

# Run Analysis
if uploaded_file:
    st.success("File uploaded successfully.")
    summary_df, figure_paths, combined_export = run_kinetic_fitting(uploaded_file)

    # Display Results Table
    if summary_df is not None and not summary_df.empty:
        st.write("### Fitting Results")
        st.dataframe(summary_df)

        # Provide a Download Option for Fitting Results
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Fitting Results as CSV",
            data=csv_data,
            file_name="kinetic_fitting_results.csv",
            mime="text/csv"
        )

        # Provide Download for Combined Original + Fitting Data
        combined_csv = combined_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Original and Fitting Data",
            data=combined_csv,
            file_name="kinetic_original_and_fitting_data.csv",
            mime="text/csv"
        )

    # Display Figures
    if figure_paths:
        st.write("### Fitting Figures")
        for idx, fig_io in enumerate(figure_paths):
            st.image(fig_io, caption=f"Sheet {idx + 1}", use_container_width=True)
