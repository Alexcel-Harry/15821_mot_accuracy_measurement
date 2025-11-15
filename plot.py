import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# 1. SET YOUR FILE NAME HERE
prefix = "end_to_end_results_edge_11n"
file_name = prefix + ".csv"
# 2. SET THE COLUMN YOU WANT TO PLOT
# (0 is the first column, 1 is the second, etc.)
column_to_plot = 0
# ---

try:
    # Load the CSV file. header=None is the key part.
    df = pd.read_csv(file_name, header=None)

    # --- Inspect the Data ---
    # It's good practice to see what was loaded
    print(f"Successfully loaded {file_name}.")
    print("First 5 rows:")
    print(df.head())
    print("\nData Info (to see columns and types):")
    df.info()

    # --- Check if the column exists and is numeric ---
    if column_to_plot in df.columns and pd.api.types.is_numeric_dtype(df[column_to_plot]):
        
        print(f"\nPlotting distribution for column {column_to_plot}...")
        
        # Set up the plot
        plt.figure(figsize=(10, 6))
        
        # Create the histogram.
        # kde=True adds the smooth density line.
        sns.histplot(df[column_to_plot], bins=30, kde=True)
        
        # Add informative labels
        plt.title(f'Distribution of Column {column_to_plot}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        # Save the plot to a file
        plot_file_name = prefix + '.png'
        plt.savefig(plot_file_name)
        
        print(f"Plot saved as {plot_file_name}")

    else:
        print(f"\nError: Column {column_to_plot} does not exist or is not numeric.")
        print(f"Available numeric columns are: {df.select_dtypes(include=['number']).columns.tolist()}")

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    print("Please make sure the file is in the same directory or provide the full path.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{file_name}' is empty.")
except Exception as e:
    print(f"An error occurred: {e}")