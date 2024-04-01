# -*- coding: utf-8 -*-
import time
import numpy as np
import pygame
import serial
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

from serial.tools import list_ports
from pantograph import Pantograph
from pshape import PShape
from pyhapi import Board, Device, Mechanisms

# Functions for heightmap generation and gradient calculations

class PerturbationArea(pygame.Rect):
    """
    A class to represent a perturbation area in a pygame environment.

    Attributes:
        mode (int): The mode determining the type of perturbation behavior.
    """

    def __init__(self, x, y, height, width):
        """
        Initializes the PerturbationArea object.

        Args:
            x (int): The x-coordinate of the top-left corner of the perturbation area.
            y (int): The y-coordinate of the top-left corner of the perturbation area.
            height (int): The height of the perturbation area.
            width (int): The width of the perturbation area.
        """
        super().__init__(x, y, width, height)
        
    def apply_force(self, haptic, t):
        """
        Computes the force applied during interaction with the haptic object.

        Args:
            haptic (pygame.Rect): The rectangle representing the haptic object.
            t (float): The current time.

        Returns:
            numpy.ndarray: The force applied during the interaction.
        """
        F = np.array([0, 0])
        if haptic.colliderect(self):
            amplitude = 2
            frequency = 0.1
            phase = 0
            F[0] += amplitude * np.cos(frequency * t + phase)
            F[1] += amplitude * np.sin(frequency * t + phase)
        return F

    def draw(self, screen):
        """
        Draws the perturbation area on the screen.

        Args:
            screen (pygame.Surface): The surface representing the screen.
        """
        pygame.draw.rect(screen, (255, 255, 255, 128), self, width=4)


def gaussian(x, y, mu, depth=100, sigma=100):
    """
    Compute the value of a 2D Gaussian function at given coordinates.

    Parameters:
        x (float or ndarray): x-coordinate(s) of the point(s) where to evaluate the Gaussian.
        y (float or ndarray): y-coordinate(s) of the point(s) where to evaluate the Gaussian.
        mu (tuple): Mean of the Gaussian function in the form (mu_x, mu_y).
        sigma (float, optional): Standard deviation of the Gaussian. Defaults to 100.

    Returns:
        float or ndarray: Value(s) of the Gaussian function at the given coordinates.
    """
    return depth * -np.exp(-10 * ((x - mu[0])**2/(sigma**2) + (y - mu[1])**2/(sigma**2)))

def generate_heightmaps(coordinates, depth, step_size=1):
    """
    Generate a detailed and lower-resolution heightmap based on given coordinates.

    Parameters:
        step_size (int): Step size for generating the grid.
        coordinates (list): List of coordinate tuples defining the path.

    Returns:
        tuple: A tuple containing detailed heightmap and lower resolution heightmap.
    """
    # Generate grid
    x = np.arange(0, 600, step_size)
    y = np.arange(0, 400, step_size)
    X, Y = np.meshgrid(x, y)

    heightmap = np.ones((600,400))

    # Compute heightmap
    for i in range(0, len(coordinates) - 1):
        dif = np.array(coordinates[i]) - np.array(coordinates[i+1])
        grad = dif / np.linalg.norm(dif)
        for j in range(int(np.linalg.norm(dif)/10) - 1):
            heightmap += gaussian(X.T, Y.T, coordinates[i] + j * - grad * 10, depth)
    
    heightmap += gaussian(X.T, Y.T, coordinates[-1], depth=150, sigma=250)
    heightmap += gaussian(X.T, Y.T, coordinates[0], depth=150, sigma=250)

    lower_res_heightmap = lower_resolution(heightmap, 10)
    
    return heightmap, lower_res_heightmap

def generate_perturbations_area(coordinates):
    """
    Generate perturbation areas along the given path coordinates.

    Parameters:
        coordinates (list): List of coordinate tuples defining the path.

    Returns:
        list: List of PerturbationArea objects representing the generated perturbation areas.
    """
    areas = []  # Initialize list to store generated perturbation areas
    nrb_perturbations = 1  # Number of random non-roadblock perturbations
    coord_list = []

    for i in range(1, len(coordinates) - 2):
        dif = np.array(coordinates[i]) - np.array(coordinates[i+1])  # Vector between consecutive coordinates
        grad = dif / np.linalg.norm(dif)  # Gradient representing the direction of the path
        for j in range(int(np.linalg.norm(dif)/10) - 1):  # Iterate over intervals along the path
                coord_list.append(coordinates[i] + j * -grad * 10)  # Compute coordinates
    random_perturbation_idx = random.randint(0, len(coord_list))
    areas.append(PerturbationArea(coord_list[random_perturbation_idx][0] - 50, coord_list[random_perturbation_idx][1] - 50, 100, 100))  # Create and append PerturbationArea object
    
    return areas
                
def lower_resolution(array, factor):
    """
    Lower the resolution of a 2D array by averaging values within blocks.

    Parameters:
        array (ndarray): The input 2D array.
        factor (int): The factor by which to reduce the resolution.

    Returns:
        ndarray: Lower resolution 2D array.
    """
    # Determine the new shape of the array
    new_shape = (array.shape[0] // factor, array.shape[1] // factor)
    
    # Reshape the array into blocks of size (factor, factor)
    reshaped_array = array[:new_shape[0]*factor, :new_shape[1]*factor].reshape(new_shape[0], factor, new_shape[1], factor)
    
    # Take the average within each block to reduce resolution
    lower_res_array = reshaped_array.mean(axis=(1, 3))
    
    return lower_res_array

def compute_gradient_at_position(pos, heightmap, step_size=1):
    """
    Compute the gradient at a given position on a heightmap.

    Parameters:
        pos (tuple): Position coordinates (x, y) where gradient needs to be computed.
        heightmap (ndarray): 2D array representing the heightmap.
        step_size (int): Step size used in generating the heightmap.

    Returns:
        ndarray: Array containing the gradient components (dx, dy) at the given position.
    """
    # position to heightmap grid indices
    ix, iy = int(pos[0] // step_size), int(pos[1] // step_size)
    ix = max(min(ix, heightmap.shape[0] - 2), 1)
    iy = max(min(iy, heightmap.shape[1] - 2), 1)
    # calculate gradient components
    dx = (heightmap[ix + 1, iy] - heightmap[ix - 1, iy]) / (2 * step_size)
    dy = (heightmap[ix, iy + 1] - heightmap[ix, iy - 1]) / (2 * step_size)
    # print(f'gradient at current position(dx, dy): ({dx:.3f}, {dy:.3f})')
    return np.array([dx, dy])

def generate_random_heighmaps():
    """
    Generate random heightmaps for different environments.

    Returns:
        list: A list containing tuples for each environment. Each tuple contains:
              - int: Depth of the environment.
              - heightmap: The generated heightmap for the environment.
              - lower_res_heightmap: A lower resolution version of the heightmap.
              - list: Coordinates defining the boundaries of the environment.
    """
    coordinates_0 = [(50,50),(300,50),(300,200),(100,200),(100,350),(500,350),(500,100)]
    coordinates_1 = [(100, 100), (100, 300), (300, 300), (300, 100), (500, 100), (500, 300)]
    coordinates_2 = [(100,100),(500,100),(500,350),(100,350),(100,200),(250,200)]

    coord_list = [coordinates_0, coordinates_1, coordinates_2]

    depth_list = [110, 80, 50]  # to change
    environments = []
    depths = []
    randomness_depth = random.randint(0, 2)
    randomness_coord = random.randint(0, 2)
    for i in range(3):
        depth = depth_list[(i + randomness_depth) % len(depth_list)]
        depths.append(depth)
        coord = coord_list[(i + randomness_coord) % len(coord_list)]
        heightmap, lower_res_heightmap = generate_heightmaps(coord, depth)
        areas = generate_perturbations_area(coord)
        environments.append((depth, heightmap, lower_res_heightmap, coord, (i + randomness_coord) % len(coord_list), areas))
    
    return environments, depths

def closest_point_on_line_segment(p1, p2, p):
    """
    Calculate the closest point to p on the line segment defined by points p1 and p2.

    Parameters:
    p1 (tuple): The start point of the segment.
    p2 (tuple): The end point of the segment.
    p (tuple): The point to find the closest point to.

    Returns:
    tuple: The closest point on the segment to point p.
    """
    line_vec = np.array(p2) - np.array(p1)
    point_vec = np.array(p) - np.array(p1)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    p_vec_scaled = np.dot(point_vec, line_unitvec)
    if p_vec_scaled < 0:
        return p1
    elif p_vec_scaled > line_len:
        return p2
    else:
        return np.array(p1) + line_unitvec * p_vec_scaled


def distance_to_track(xh, coordinates):
    """
    Calculates the minimum distance from a point to a track defined by a list of coordinates.

    Parameters:
    xh (tuple): The point (x, y).
    coordinates (list): The list of coordinates defining the track.

    Returns:
    float: The minimum distance from the point to the track.
    """
    min_distance = np.inf
    for i in range(len(coordinates) - 1):
        closest_point = closest_point_on_line_segment(coordinates[i], coordinates[i+1], xh)
        distance = np.linalg.norm(np.array(xh) - np.array(closest_point))
        if distance < min_distance:
            min_distance = distance
    return min_distance


def save_to_csv(subject_id, run_i, state, depth, coordinates_index, perturbed=False):
    """
    Save simulation data to CSV file.

    Parameters:
        subject_id (int): Identifier for the subject.
        run_i (int): Identifier for the run.
        state (list): List containing simulation state data.
        depth (int): Depth of the environment.
        coordinates_index (int): Index representing the coordinates of the environment.
        perturbed (bool, optional): Whether the environment was perturbed. Defaults to False.
    """
    # Prepare the data for CSV
    # Assuming state is a list of [t, xh[0], xh[1]] for each timestep
    df = pd.DataFrame(state, columns=['t', 'xh[0]', 'xh[1]', 'dist'])

    # Determine filename
    perturbation_status = "perturbed" if perturbed else "normal"
    file_name = f'subject_{subject_id}_run_{run_i}_dept_{depth}_coordinates_{coordinates_index}_{perturbation_status}.csv'
    path = folder + '/' + file_name

    # Save to CSV
    df.to_csv(path , index=False)
    print(f'Data saved to {file_name} inside folder {folder}')


def main(environment, perturbations=False):
    """
    Main function to initialize and run the virtual haptic device simulation.

    Args:
        environment (tuple): A tuple containing the environment information including:
            - depth (int): Depth of the environment.
            - heightmap (list of lists): 2D list representing the heightmap of the terrain.
            - lower_res_heightmap (list of lists): 2D list representing a lower resolution version of the heightmap.
            - coordinates (list of tuples): List of coordinate points.

    Returns:
        None
    """
    ################################ Pygame Initialisation ################################
    
    # General Pygame Initialization
    pygame.init()
    window = pygame.display.set_mode((1200, 400))
    pygame.display.set_caption('Virtual Haptic Device')
    clock = pygame.time.Clock()

    # Setting up surfaces
    screenHaptics = pygame.Surface((600, 400))
    screenVR = pygame.Surface((600, 400))

    # Setting up icon
    icon = pygame.image.load('robot.png')
    pygame.display.set_icon(icon)

    # Setting up text for debugging
    font = pygame.font.Font('freesansbold.ttf', 18)
    text = font.render('Virtual Haptic Device', True, (0, 0, 0), (255, 255, 255))
    textRect = text.get_rect()
    textRect.topleft = (10, 10)

    # Some Constants
    cWhite = (255, 255, 255)
    cOrange = (255, 100, 0)
    cLightblue = (0,176,240)

    # Pseudo-haptics dynamic parameters
    k = 0.5  # Stiffness between cursor and haptic display
    b = 0.8  # Viscous of the pseudohaptic display

    ##define sprites
    hhandle = pygame.image.load('handle.png')
    haptic  = pygame.Rect(*screenHaptics.get_rect().center, 0, 0).inflate(30, 30)
    cursor  = pygame.Rect(0, 0, 5, 5)
    colorHaptic = cOrange ##color of the wall

    xh = np.array(haptic.center)

    ##Set the old value to 0 to avoid jumps at init
    xhold = 0

    ################################ Main Loop ################################
    
    # Main loop initialisation
    run = True
    robotToggle = True
    heightmapToggle = False
    recordingToggle = False
    debugToggle = False
    depth, heightmap, lower_res_heightmap, coordinates, coordinates_index, areas = environment
    state = [] # State vector
    endpoint  = pygame.Rect(coordinates[-1][0] - 1, coordinates[-1][1] - 1, 1, 1)
    t_start = None
    # conversion from meters to pixels
    window_scale = 3
    dt = 0.01  # Integration step time
    FPS = int(1 / dt)  # Refresh rate
    
    while run:
        #########Process events  (Mouse, Keyboard etc...)#########
        for event in pygame.event.get():
            ##If the window is close then quit 
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYUP:
                if event.key == ord('m'):   ##Change the visibility of the mouse
                    pygame.mouse.set_visible(not pygame.mouse.get_visible())  
                if event.key == ord('q'):   ##Force to quit
                    run = False            
                if event.key == ord('d'):
                    debugToggle = not debugToggle
                if event.key == ord('r'):
                    robotToggle = not robotToggle
                ##Toggle the wall or the height map
                if event.key == ord('h'):
                    heightmapToggle = not heightmapToggle
                    
                if event.key == ord('p'):
                    recordingToggle = not recordingToggle

        ######### Read position (Haply and/or Mouse)  #########
        if haptic.colliderect(endpoint) and recordingToggle:
            if port:
                fe = np.zeros(2)
                ##Update the forces of the device
                device.set_device_torques(fe)
                device.device_write_torques()
                #pause for 1 millisecond
                time.sleep(0.001)
            run = False


        ##Get endpoint position xh
        if port and haplyBoard.data_available():    ##If Haply is present
            #Waiting for the device to be available
            #########Read the motorangles from the board#########
            device.device_read_data()
            motorAngle = device.get_device_angles()
            
            #########Convert it into position#########
            device_position = device.get_device_position(motorAngle)
            xh = np.array(device_position)*1e3*window_scale
            xh[0] = np.round(-xh[0]+300)
            xh[1] = np.round(xh[1]-60)
            xm = xh     ##Mouse position is not used
            
        else:
            ##Compute distances and forces between blocks
            xh = np.clip(np.array(haptic.center),0,599)
            xh = np.round(xh)
            
            ##Get mouse position
            cursor.center = pygame.mouse.get_pos()
            xm = np.clip(np.array(cursor.center),0,599)
                
        
        
        ################################ Compute forces ################################
        
        fe = np.zeros(2)  ##Environment force is set to 0 initially.

        # Damping
        b_virtual = 5  # Virtual damping coefficient
        velocity_estimate = (np.array(xh) - np.array(xhold)) / FPS  # Estimate velocity
        damping_force = -b_virtual * velocity_estimate  # Calculate the force applied on the device by the damper based on velocity estimations
        fe += damping_force
        
        # Virtual Shapes
        k_field = 0.25 #field stiffness coefficient

        # calculate(based on gradient and add force)
        fe_shape = k_field * compute_gradient_at_position(xh, heightmap)
        fe += fe_shape
        
        if perturbations:
            for p_area in areas:
                fe += p_area.apply_force(haptic, pygame.time.get_ticks())
                
        ################################ Record data and send force to Haptic device ################################

        if recordingToggle:
            if t_start is None:
                t_start = pygame.time.get_ticks()
            # log states for analysis
            t = (pygame.time.get_ticks() - t_start)

            # Calculate the distance and append to state
            distance = distance_to_track(xh, coordinates) #distance in pixels!
            state.append([t, xh[0], xh[1], distance]) #append to state!
        
        ##Update old samples for velocity computation
        xhold = xh   
        
        ######### Send forces to the device #########
        if port:
            fe[1] = -fe[1]  ##Flips the force on the Y=axis 
            
            ##Update the forces of the device
            device.set_device_torques(fe)
            device.device_write_torques()
            #pause for 1 millisecond
            time.sleep(0.001)
        else:
            ######### Update the positions according to the forces ########
            ##Compute simulation (here there is no inertia)
            ##If the haply is connected xm=xh and dxh = 0
            dxh = (k/b*(xm-xh)/window_scale -fe/b)    ####replace with the valid expression that takes all the forces into account
            dxh = dxh*window_scale
            xh = np.round(xh+dxh)             ##update new positon of the end effector
            
        haptic.center = xh 
        
        ################################ Graphical output ################################
        
        ##Render the haptic surface
        screenHaptics.fill(cWhite)
        
        ##Change color based on effort
        colorMaster = (255,\
            255-np.clip(np.linalg.norm(k*(xm-xh)/window_scale)*15,0,255),\
            255-np.clip(np.linalg.norm(k*(xm-xh)/window_scale)*15,0,255)) #if collide else (255, 255, 255)

            
        pygame.draw.rect(screenHaptics, colorMaster, haptic,border_radius=4)
        

        ######### Robot visualization ###################
        # Update individual link position
        if robotToggle:
            robot.createPantograph(screenHaptics,xh)
            
        # Hand visualisation
        screenHaptics.blit(hhandle,(haptic.topleft[0],haptic.topleft[1]))
        pygame.draw.line(screenHaptics, (0, 0, 0), (haptic.center),(haptic.center+2*k*(xm-xh)))
        
        # Render the VR surface
        screenVR.fill(cLightblue)
        
        # Define colors for terrain
        color_min = (0, 0, 255)  # Dark blue
        color_max = (0, 255, 0)  # Green
        
        # Normalize the heightmap values to the range [0, 1]
        min_height = min(map(min, lower_res_heightmap))
        max_height = max(map(max, lower_res_heightmap))
        normalized_heightmap = [[(height - min_height) / (max_height - min_height) for height in row] for row in lower_res_heightmap]

        # Function to convert a normalized height to a color
        def height_to_color(height):
            r = int((1 - height) * color_min[0] + height * color_max[0])
            g = int((1 - height) * color_min[1] + height * color_max[1])
            b = int((1 - height) * color_min[2] + height * color_max[2])
            return r, g, b
        
        # Draw the heightmap
        if heightmapToggle:
            for y, row in enumerate(normalized_heightmap):
                for x, height in enumerate(row):
                    color = height_to_color(height)
                    pygame.draw.rect(screenVR, color, (y*10 , x*10 , 10, 10))
            
            if perturbations:
                for p_area in areas:
                    p_area.draw(screenVR)

        # Draw coordinate points
        def draw_point(coord, color=(255, 0, 0), radius=50, width=3):
            pygame.draw.circle(screenVR, color, coord, radius, width)
        
        # Draw starting circle        
        if not recordingToggle:
            draw_point(coordinates[0])
        
        # Draw haptic device on VR screen
        pygame.draw.rect(screenVR, colorHaptic, haptic, border_radius=8)
        
        # Draw goal
        #pygame.draw.rect(screenVR, (255, 0, 255), endpoint.inflate(10,10))
        
        # Fuse it back together
        window.blit(screenHaptics, (0,0))
        window.blit(screenVR, (600,0))

        # Print status in  overlay
        if debugToggle: 
            text = font.render("FPS = " + str(round(clock.get_fps())) + \
                                "  xm = " + str(np.round(10*xm)/10) +\
                                "  xh = " + str(np.round(10*xh)/10) +\
                                "  fe = " + str(np.round(10*fe)/10) \
                                , True, (0, 0, 0), (255, 255, 255))
            window.blit(text, textRect)


        pygame.display.flip()    
        # Slow down the loop to match FPS
        clock.tick(FPS)

    pygame.display.quit()
    pygame.quit()

    
    
    state = np.array(state)
    coordinates = np.array(coordinates)
    return state, coordinates, depth, coordinates_index


if __name__ == "__main__":
    # USB serial microcontroller program id data:
    def serial_ports():
        """ Lists serial port names """
        ports = list(serial.tools.list_ports.comports())

        result = []
        for p in ports:
            try:
                port = p.device
                s = serial.Serial(port)
                s.close()
                if p.description[0:12] == "Arduino Zero":
                    result.append(port)
                    print(p.description[0:12])
            except (OSError, serial.SerialException):
                pass
        return result


    # Detect and Connect Physical device
    port = None  # Default to no port
    haplyBoard = None
    device = None
    pantograph = None
    robot = PShape
    CW = 0
    CCW = 1

    # Open the connection with the arduino board
    port = serial_ports()  ##port contains the communication port or False if no device

    if port:
        print("Board found on port %s" % port[0])
        haplyBoard = Board("test", port[0], 0)
        device = Device(5, haplyBoard)
        pantograph = Pantograph()
        device.set_mechanism(pantograph)

        device.add_actuator(1, CCW, 2)
        device.add_actuator(2, CW, 1)

        device.add_encoder(1, CCW, 241, 10752, 2)
        device.add_encoder(2, CW, -61, 10752, 1)

        device.device_set_parameters()
    else:
        print("No compatible device found. Running virtual environnement...")

    print(f'Insert participant ID please:')
    subject_id = input()
    environments, depths_list = generate_random_heighmaps()
    data = list()
    run_i = 0
    folder = f'data_recordings'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully.")
    else:
        print(f"Folder '{folder}' already exists.")

    # Run environments without perturbations
    for environment in environments:
        input('Please reset the position. Ready?')
        recorded = False
        while not recorded:
            state, coordinates, depth, coordinates_index = main(environment)
            if port:
                fe = np.zeros(2)
                # Update the forces of the device
                device.set_device_torques(fe)
                device.device_write_torques()
                # Pause for 1 millisecond
                time.sleep(0.001)
        
            # save_to_csv(folder, subject_id, run_i, state, coordinates)
            if len(state) != 0:
                data.append((state, coordinates))
                recorded = True
                run_i += 1
                save_to_csv(subject_id, run_i, state, depth, coordinates_index, False)
            else:
                print('nothing recorded(didnt press p), please try again')
                print('lets try again!')
                continue

    # Run environments with perturbations
    for environment in environments:
        input('Please reset the position. Ready?')
        recorded = False
        while not recorded:
            state, coordinates, depth, coordinates_index = main(environment, perturbations=True)
            if port:
                fe = np.zeros(2)
                # Update the forces of the device
                device.set_device_torques(fe)
                device.device_write_torques()
                # Pause for 1 millisecond
                time.sleep(0.001)
            
            # save_to_csv(folder, subject_id, run_i, state, coordinates)
            if len(state) != 0:
                data.append((state, coordinates))
                recorded = True
                run_i += 1
                save_to_csv(subject_id, run_i, state, depth, coordinates_index, True)
            else:
                print('nothing recorded(didnt press p), please try again')
                print('lets try again!')
                continue
        
    # Create a figure and 6 subplots arranged in a 3 column, 2 row format
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # Plot data on each subplot
    for d in range(len(data)):
        for i in range(len(data[d][1]) - 1):
            axs[d//3, d%3].plot(data[d][1][i:i + 2, 0], -data[d][1][i:i + 2, 1], "red", label="Segment" if i == 0 else None)
        if data[d][0].size != 0:
            axs[d//3, d%3].plot(data[d][0][:, 1], -data[d][0][:, 2], "lime", label="CONTROLLER")
        else:
            axs[d//3, d%3].plot(300, 200, "lime", label="CONTROLLER")
        axs[d//3, d%3].axis('equal')
        axs[d//3, d%3].set_xlabel("x")
        axs[d//3, d%3].set_ylabel("y [m")
        axs[d//3, d%3].legend()
        axs[d//3, d%3].set_title(f'Plot {d + 1}, Depth: {depths_list[d%3]}')
    
    # Adjust layout
    plt.tight_layout()

    # Save plot as .png image
    image_path= folder + "/" + subject_id
    plt.savefig(f'{image_path}.png')

    # Show plot
    plt.show()
