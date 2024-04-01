# BlindMaps Project

This project simulates a virtual haptic device interacting with heightmap environments. The goal of this project is to see the impact of the depth of a generated path on the time required to complete the task and the deviation from the path, depending on if perturbations are present or not. It includes Python scripts for generating heightmaps, simulating haptic interactions, saving simulation data, and data analysis.

## Files

### 1. blindmaps.py

This Python script contains the main functionality of the BlindMaps project. It includes classes and functions for:

- Generating heightmaps based on given coordinates.
- Creating perturbation areas along the path.
- Computing gradients and forces for haptic interactions.
- Running the simulation and recording data.
- Connecting and interacting with a physical haptic device (if available).

### 2. data_analysis.py

This python script will create plots based on the data obtained from the blindmaps.py script. It contains plotting for:

- Bar plots for the average deviation to the path and the time to complete the task, depending on the depth of the path but also on the presence or not of perturbations.
- A plot of the average deviation from the path with respect to the time to complete the task. Along with polynomial regression of the 3nd degree.

### 3. handle.png and robot.png

This are images required to run the simulation.

### 4. pantograph.py/pshape.py/pyhapi.py

These are dependencies required to be able to use the haptic device, but also aloowing certain visual representations.

## How to Use

1. **Python Environment Setup**: Make sure you have Python installed along with necessary packages.

2. **Connect Haptic Device (Optional)**: If you have a physical haptic device, connect it to your computer and ensure it's compatible. The script will attempt to detect and connect to the device automatically. In order to connect the haptic device you must have the arduino IDE installed.

3. **Run the Script**: Execute the `blindmaps.py` script. It will simulate the virtual haptic device interacting with heightmap environments.

4. **Complete the experiment**: In the python terminal you will be asked to reset the haptic device position before starting every trial. Once this is done, you may press enter in the terminal and a new window will open. Using your haptic device or your mouse, place the orange square in the circle appearing on screen and press 'p' on your keyboard to start the experiment. The goal is to try and follow the path as close as possible and reach the end as fast as possible. Once you reach the goal the window will close automatically and you'll be asked to reset the haptic device postion.

    - **Additionnal Interactions**: Use keyboard commands to control the simulation. Press 'm' to toggle mouse visibility, 'q' to quit, 'd' to toggle debug mode, 'r' to toggle robot visualization, 'h' to toggle heightmap display.

5. **Data Recording**: Simulation data will be recorded and saved to CSV files in the `data_recordings` folder. Each CSV file contains timestamped positions and distances during the simulation.

6. **Visualization**: After running the simulation, a visualization of the recorded data will be displayed. Each plot corresponds to a different environment, showing the path and haptic interactions.

7. **Data Analysis**: Once data has been recorded, you may execute the `data_analysis.py` that will display multiple plots to help analyse the data obtained.

## Dependencies

### Packages
- Python 3.x
- pygame
- serial
- numpy
- pandas
- matplotlib
- os

### Files
- pantograph.py
- pshape.py 
- pyhapi.py
- handle.png
- robot.png

### Haptic Device Support (optionnal)
- arduino IDE

## Additional Notes

- This project includes support for connecting and interacting with a physical haptic device if available.

