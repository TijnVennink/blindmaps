import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

# Function to read and process CSV files
def process_csv_files(folder_path):
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
    plt.figure(figsize=(12, 12))

    # Plot time to complete vs depth for normal data
    plt.subplot(2, 2, 1)
    normal_data = df_mean[df_mean['condition'] == 'normal']
    bar_width = 20  # Adjust the width of the bars
    segmentation = 3 # please only use odd segmentation :D
    offset = bar_width / (segmentation)
    plt.bar(normal_data['depth'], normal_data['time_to_complete'], color='blue', width=bar_width, alpha=0.7) # Make slightly opaque
    split_normal_time = split_dataframe(df[df['condition'] == 'normal'], segmentation, 'time_to_complete')
    for i, part in enumerate(split_normal_time):
        for j, sub_part in enumerate(part):
            plt.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))) ,sub_part['time_to_complete'].mean(), color='blue', width=bar_width / segmentation - 3)
    plt.title('Time to Complete vs Depth (Normal)')
    plt.xlabel('Depth')
    plt.ylabel('Time to Complete')
    plt.xticks(normal_data['depth'].unique())  # Show only unique depth values on x-axis

    # Plot time to complete vs depth for perturbed data
    plt.subplot(2, 2, 2)
    perturbed_data = df_mean[df_mean['condition'] == 'perturbed']
    plt.bar(perturbed_data['depth'], perturbed_data['time_to_complete'], color='orange', width=bar_width, alpha=0.7) # Make slightly opaque
    split_perturbed_time = split_dataframe(df[df['condition'] == 'perturbed'], segmentation, 'time_to_complete')
    for i, part in enumerate(split_perturbed_time):
        for j, sub_part in enumerate(part):
            plt.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))), sub_part['time_to_complete'].mean(), color='orange', width=bar_width / segmentation - 3)    
    plt.title('Time to Complete vs Depth (Perturbed)')
    plt.xlabel('Depth')
    plt.ylabel('Time to Complete')
    plt.xticks(perturbed_data['depth'].unique())  # Show only unique depth values on x-axis

    # Plot average distance vs depth for normal data
    plt.subplot(2, 2, 3)
    plt.bar(normal_data['depth'], normal_data['avg_distance'], color='blue', width=bar_width, alpha=0.7) # Make slightly opaque
    split_normal_distance = split_dataframe(df[df['condition'] == 'normal'], segmentation, 'avg_distance')
    for i, part in enumerate(split_normal_distance):
        for j, sub_part in enumerate(part):
            plt.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))) ,sub_part['avg_distance'].mean(), color='blue', width=bar_width / segmentation - 3)
    plt.title('Average Distance to Desired Path vs Depth (Normal)')
    plt.xlabel('Depth')
    plt.ylabel('Average Distance')
    plt.xticks(normal_data['depth'].unique())  # Show only unique depth values on x-axis

    # Plot average distance vs depth for perturbed data
    plt.subplot(2, 2, 4)
    plt.bar(perturbed_data['depth'], perturbed_data['avg_distance'], color='orange', width=bar_width, alpha=0.7) # Make slightly opaque
    split_perturbed_distance = split_dataframe(df[df['condition'] == 'perturbed'], segmentation, 'avg_distance')
    for i, part in enumerate(split_perturbed_distance):
        for j, sub_part in enumerate(part):
            plt.bar(sub_part['depth'] + offset * (((segmentation - 1) / 2) * (j - ((segmentation - 1) / 2))), sub_part['avg_distance'].mean(), color='orange', width=bar_width / segmentation - 3)
    plt.title('Average Distance to Desired Path vs Depth (Perturbed)')
    plt.xlabel('Depth')
    plt.ylabel('Average Distance')
    plt.xticks(perturbed_data['depth'].unique())  # Show only unique depth values on x-axis

    plt.tight_layout()
    plt.show()
    
    df_sorted_time = df.sort_values(by='time_to_complete', ascending=True)
    plt.plot(df_sorted_time[df_sorted_time['condition'] == 'normal']['time_to_complete'], df_sorted_time[df_sorted_time['condition'] == 'normal']['avg_distance'], '-', color='blue', label='Normal')
    plt.plot(df_sorted_time[df_sorted_time['condition'] == 'perturbed']['time_to_complete'], df_sorted_time[df_sorted_time['condition'] == 'perturbed']['avg_distance'], '-', color='orange', label='Perturbed')
    plt.xlabel('Time to complete')
    plt.ylabel('Average distance from desired path')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function
def main(folder_path):
    df, df_mean = process_csv_files(folder_path)
    df_mean.sort_values(by='depth', ascending=True, inplace=True)
    df.sort_values(by='depth', ascending=True, inplace=True)
    plot_graphs(df, df_mean)

if __name__ == "__main__":
    folder_path = f"data_recordings"
    main(folder_path)