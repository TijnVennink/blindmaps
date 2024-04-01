import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Function to read and process CSV files
def process_csv_files(folder_path):
    """
    Read and process CSV files in a folder.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        DataFrame: Original DataFrame with data from all CSV files.
        DataFrame: DataFrame with mean values grouped by depth and condition.
    """
    data = {'depth': [], 'time_to_complete': [], 'avg_distance': [], 'condition': []}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            parts = filename.split("_")
            subject_id = parts[1]
            run_number = parts[3]
            depth = parts[5]
            condition = parts[8].split(".")[0]

            df = pd.read_csv(filepath)
            time_to_complete = df['t'].max()
            avg_distance = df['dist'].mean()

            data['depth'].append(int(depth))
            data['time_to_complete'].append(time_to_complete)
            data['avg_distance'].append(avg_distance)
            data['condition'].append(condition)

    df = pd.DataFrame(data)
    # Group by depth and condition, and calculate the mean
    df_mean = df.groupby(['depth', 'condition']).mean().reset_index()
    return df, df_mean

# Function to split DataFrame into a certain number of parts
def split_dataframe(df, num_parts, value):
    """
    Split DataFrame into multiple parts based on a certain column value.

    Parameters:
        df (DataFrame): DataFrame to split.
        num_parts (int): Number of parts to split the DataFrame into.
        value (str): Column name based on which DataFrame is to be split.

    Returns:
        list: List of DataFrames, each representing a part.
    """
    parts = []
    data = df.sort_values(by=f'{value}', ascending=True)
    for depth in data['depth'].unique():
        df_depth = data[data['depth'] == depth]
        total_rows = len(df_depth)
        rows_per_part = total_rows // num_parts
        remaining_rows = total_rows % num_parts
        
        sub_parts = []
        start = 0
        for i in range(num_parts):
            end = start + rows_per_part
            if i < remaining_rows:
                end += 1
            sub_parts.append(df_depth.iloc[start:end])
            start = end
        parts.append(sub_parts)
    
    return parts

# Function to plot graphs
def plot_graphs(df, df_mean):
    """
    Plot various graphs based on provided DataFrame and its mean.

    Parameters:
        df (DataFrame): Original DataFrame with data from all CSV files.
        df_mean (DataFrame): DataFrame with mean values grouped by depth and condition.
    """
    plt.figure(figsize=(12, 12))
    
    # Seperate normal and perturbed data
    perturbed_data = df_mean[df_mean['condition'] == 'perturbed']
    normal_data = df_mean[df_mean['condition'] == 'normal']
    
    # Get maximum and minimum values of time_to_complete for both conditions
    min_time = min(normal_data['time_to_complete'].min(), perturbed_data['time_to_complete'].min())
    max_time = max(normal_data['time_to_complete'].max(), perturbed_data['time_to_complete'].max())    

    # Plot time to complete vs depth for normal data
    plot1 = plt.subplot(2, 2, 1)
    bar_width = 20  # Adjust the width of the bars
    segmentation = 1 if len(df) < 16 else 3 # please only use odd segmentation :D
    offset = bar_width / (segmentation)
    plot1.bar(normal_data['depth'], normal_data['time_to_complete'], color='blue', width=bar_width, alpha=0.7) # Make slightly opaque
    split_normal_time = split_dataframe(df[df['condition'] == 'normal'], segmentation, 'time_to_complete')
    for i, part in enumerate(split_normal_time):
        for j, sub_part in enumerate(part):
            plot1.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))) ,sub_part['time_to_complete'].mean(), color='blue', width=bar_width / segmentation - 3)
            min_time = min(min_time, sub_part['time_to_complete'].mean().min())
            max_time = max(max_time, sub_part['time_to_complete'].mean().max())
    plot1.set_title('Time to Complete vs Depth (Normal)')
    plot1.set_xlabel('Depth')
    plot1.set_ylabel('Time to Complete')
    plot1.set_xticks(normal_data['depth'].unique())  # Show only unique depth values on x-axis

    # Plot time to complete vs depth for perturbed data
    plot2 = plt.subplot(2, 2, 2)
    plot2.bar(perturbed_data['depth'], perturbed_data['time_to_complete'], color='orange', width=bar_width, alpha=0.7) # Make slightly opaque
    split_perturbed_time = split_dataframe(df[df['condition'] == 'perturbed'], segmentation, 'time_to_complete')
    for i, part in enumerate(split_perturbed_time):
        for j, sub_part in enumerate(part):
            plot2.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))), sub_part['time_to_complete'].mean(), color='orange', width=bar_width / segmentation - 3)    
            min_time = min(min_time, sub_part['time_to_complete'].mean().min())
            max_time = max(max_time, sub_part['time_to_complete'].mean().max())
    plot2.set_title('Time to Complete vs Depth (Perturbed)')
    plot2.set_xlabel('Depth')
    plot2.set_ylabel('Time to Complete')
    plot2.set_xticks(perturbed_data['depth'].unique())  # Show only unique depth values on x-axis
    
    plot1.set_ylim(min_time, max_time+500)  # Set y-limits
    plot2.set_ylim(min_time, max_time+500)  # Set y-limits
    
     # Get maximum and minimum values of avg_distance for both conditions
    min_avg_distance = min(normal_data['avg_distance'].min(), perturbed_data['avg_distance'].min())
    max_avg_distance = max(normal_data['avg_distance'].max(), perturbed_data['avg_distance'].max())    

    # Plot average distance vs depth for normal data
    plot3 = plt.subplot(2, 2, 3)
    plot3.bar(normal_data['depth'], normal_data['avg_distance'], color='blue', width=bar_width, alpha=0.7) # Make slightly opaque
    split_normal_distance = split_dataframe(df[df['condition'] == 'normal'], segmentation, 'avg_distance')
    for i, part in enumerate(split_normal_distance):
        for j, sub_part in enumerate(part):
            plot3.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))) ,sub_part['avg_distance'].mean(), color='blue', width=bar_width / segmentation - 3)
            min_avg_distance = min(min_avg_distance, sub_part['avg_distance'].mean().min())
            max_avg_distance = max(max_avg_distance, sub_part['avg_distance'].mean().max())
    plot3.set_title('Average Distance to Desired Path vs Depth (Normal)')
    plot3.set_xlabel('Depth')
    plot3.set_ylabel('Average Distance')
    plot3.set_xticks(normal_data['depth'].unique())  # Show only unique depth values on x-axis
    


    # Plot average distance vs depth for perturbed data
    plot4 = plt.subplot(2, 2, 4)
    plot4.bar(perturbed_data['depth'], perturbed_data['avg_distance'], color='orange', width=bar_width, alpha=0.7) # Make slightly opaque
    split_perturbed_distance = split_dataframe(df[df['condition'] == 'perturbed'], segmentation, 'avg_distance')
    for i, part in enumerate(split_perturbed_distance):
        for j, sub_part in enumerate(part):
            plt.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))), sub_part['avg_distance'].mean(), color='orange', width=bar_width / segmentation - 3)
            min_avg_distance = min(min_avg_distance, sub_part['avg_distance'].mean().min())
            max_avg_distance = max(max_avg_distance, sub_part['avg_distance'].mean().max())
    plot4.set_title('Average Distance to Desired Path vs Depth (Perturbed)')
    plot4.set_xlabel('Depth')
    plot4.set_ylabel('Average Distance')
    plot4.set_xticks(perturbed_data['depth'].unique())  # Show only unique depth values on x-axis
    
    plot3.set_ylim(min_avg_distance, max_avg_distance+5)  # Set y-limits
    plot4.set_ylim(min_avg_distance, max_avg_distance+5)  # Set y-limits

    plt.tight_layout()
    plt.show()
    
    df_sorted_time = df.sort_values(by='time_to_complete', ascending=True)
    plt.plot(df_sorted_time[df_sorted_time['condition'] == 'normal']['time_to_complete'], df_sorted_time[df_sorted_time['condition'] == 'normal']['avg_distance'], '-', color='blue', label='Normal')
    plt.plot(df_sorted_time[df_sorted_time['condition'] == 'perturbed']['time_to_complete'], df_sorted_time[df_sorted_time['condition'] == 'perturbed']['avg_distance'], '-', color='orange', label='Perturbed')
    plt.xlabel('Time to complete')
    plt.ylabel('Average distance from desired path')
    
    # Step 3: Fit Polynomial Regression Model
    degree = 3 # Define the degree of the polynomial
    coefficients_normal = np.polyfit(df_sorted_time[df_sorted_time['condition'] == 'normal']['time_to_complete'], df_sorted_time[df_sorted_time['condition'] == 'normal']['avg_distance'], degree)
    coefficients_perturbed = np.polyfit(df_sorted_time[df_sorted_time['condition'] == 'perturbed']['time_to_complete'], df_sorted_time[df_sorted_time['condition'] == 'perturbed']['avg_distance'], degree)

    # Generate values for the regression curve
    x_values_normal = np.linspace(df_sorted_time[df_sorted_time['condition'] == 'normal']['time_to_complete'].min(), df_sorted_time[df_sorted_time['condition'] == 'normal']['time_to_complete'].max(), 10)
    y_values_normal = np.polyval(coefficients_normal, x_values_normal)
    plt.plot(x_values_normal, y_values_normal, '--', color='blue', label=f'Normal {degree}nd degree polynomial regression')
    
    # Generate values for the regression curve
    x_values_perturbed = np.linspace(df_sorted_time[df_sorted_time['condition'] == 'perturbed']['time_to_complete'].min(), df_sorted_time[df_sorted_time['condition'] == 'perturbed']['time_to_complete'].max(), 10)
    y_values_perturbed = np.polyval(coefficients_perturbed, x_values_perturbed)
    plt.plot(x_values_perturbed, y_values_perturbed, '--', color='orange', label=f'Perturbed {degree}nd degree polynomial regression')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function
def main(folder_path):
    """
    Main function to process CSV files and plot graphs.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
    """
    # Process CSV files
    df, df_mean = process_csv_files(folder_path)

    # Sort DataFrames by depth in ascending order
    df_mean.sort_values(by='depth', ascending=True, inplace=True)
    df.sort_values(by='depth', ascending=True, inplace=True)

    # Plot graphs
    plot_graphs(df, df_mean)

if __name__ == "__main__":
    folder_path = f"data_recordings"
    main(folder_path)