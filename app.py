VERSION="0.0.1"
import os, logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# plt_logger = logging.getLogger("matplotlib")
# plt_logger.setLevel(logging.WARNING)

# logger.info(f"Launching version {VERSION}")

def graph_monthly_sales(product_id, type):
    try:
        data_dir = "data/"

        product_df = pd.read_csv(os.path.join(data_dir, "ttparikh_main_table_Product.csv"))
        barcode_df = pd.read_csv(os.path.join(data_dir, "ttparikh_main_table_Barcode.csv"))
        b_invoice_barcode_df = pd.read_csv(os.path.join(data_dir, "ttparikh_main_table_B_Invoice_Barcode.csv"))
        invoice_df = pd.read_csv(os.path.join(data_dir, "ttparikh_main_table_Invoice.csv"))
        print("Data fetched successfully!")
    except Exception as e:
        print("Error connecting to MySQL:", e)
    finally:
        # Ensure all key columns have the same data type before merging
        product_df["Product_ID"] = product_df["Product_ID"].astype(str)

        barcode_df["Product_ID"] = barcode_df["Product_ID"].astype(str)
        barcode_df["Barcode_ID"] = barcode_df["Barcode_ID"].astype(str)
        barcode_df = barcode_df[barcode_df["Type"] == type]

        b_invoice_barcode_df["Barcode_ID"] = b_invoice_barcode_df["Barcode_ID"].astype(str)
        b_invoice_barcode_df["Invoice_ID"] = b_invoice_barcode_df["Invoice_ID"].astype(str)

        invoice_df["Invoice_ID"] = invoice_df["Invoice_ID"].astype(str)

        # Perform the query using pandas
        # 1. Join Product and Barcode on Product_ID
        merged_df = pd.merge(product_df, barcode_df, on="Product_ID", how="inner", suffixes=('_product', '_barcode'))

        # 2. Left join with B_Invoice_Barcode on Barcode_ID
        merged_df = pd.merge(merged_df, b_invoice_barcode_df, on="Barcode_ID", how="left", suffixes=('', '_b_invoice'))

        # 3. Left join with Invoice on Invoice_ID
        merged_df = pd.merge(merged_df, invoice_df, on="Invoice_ID", how="left", suffixes=('', '_invoice'))

        # 4. Filter where Product_ID = 15 and explicitly create a copy
        result = merged_df[merged_df["Product_ID"] == product_id].copy()

        title = ""
        if not result.empty:
            title = result.iloc[0].get("Name", "No Name Found")
        else:
            title = "No entries found"
        print(title)
        # Get the headers (column names) of the result
        headers = result.columns.tolist()

        # Keep only the specified columns
        filtered_result = result[['Create_Time_invoice', 'Count']]
        
        # Convert 'Create_Time_invoice' to datetime
        result['Create_Time_invoice'] = pd.to_datetime(result['Create_Time_invoice'])

        # Extract month and year from 'Create_Time_invoice' and create a new column 'Month_Year'
        result['Month_Year'] = result['Create_Time_invoice'].dt.to_period('M').astype(str)
        # Extract only the month from 'Create_Time_invoice' and create a new column 'Month'
        
        result['Month'] = result['Create_Time_invoice'].dt.month

        # Group by 'Month_Year' and sum the 'Count', also include the first 'Month' in the group
        monthly_summary = result.groupby('Month_Year').agg({'Count': 'sum', 'Month': 'first'}).reset_index()

        # Keep only 'Month' and 'Count' columns
        monthly_summary = monthly_summary[['Month', 'Count']]
                
        # Convert monthly_summary to a NumPy array
        monthly_summary_array = []
        monthly_summary_array.append(monthly_summary['Month'].values)
        monthly_summary_array.append(monthly_summary['Count'].values)

        X = np.transpose(np.array(monthly_summary_array[:-1]))
        y = np.transpose(np.array(monthly_summary_array[-1:]))
        m = y.size # number of training examples
        #Insert the usual column of 1's into the "X" matrix
        X = np.insert(X,0,1,axis=1)

        #Plot the data to see what it looks like
        plt.figure(figsize=(10,6))
        plt.plot(X[:,1],y[:,0],'rx',markersize=10)
        plt.grid(True) #Always plot.grid true!
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000s')

        # Show the plot
        plt.tight_layout()  # Adjust layout to prevent label overlap
        plt.show()

graph_monthly_sales(product_id="472", type="A")
