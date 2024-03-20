# -*- coding: utf-8 -*-
import time
import numpy as np
import pygame
import serial
import random

from serial.tools import list_ports
from pantograph import Pantograph
from pshape import PShape
from pyhapi import Board, Device, Mechanisms

##################### Init Simulated haptic device #####################

#### Functions for heighmap generation and gradient calculations ####

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
    for i in range(len(coordinates) - 1):
        dif = np.array(coordinates[i]) - np.array(coordinates[i+1])
        grad = dif / np.linalg.norm(dif)
        for j in range(int(np.linalg.norm(dif)/10) - 1):
            heightmap += gaussian(X.T, Y.T, coordinates[i] + j * - grad * 10, depth)

    lower_res_heightmap = lower_resolution(heightmap, 10)
    
    return heightmap, lower_res_heightmap

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
    reshaped_array = array[:new_shape[0]*factor, :new_shape[1]*factor].reshape(
        new_shape[0], factor, new_shape[1], factor)
    
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
    coordinates_1 = [(100,100),(400,100),(400,200),(200,200),(200,300),(500,300)]
    coordinates_2 = [(100,100),(400,100),(400,200),(200,200),(200,300),(500,300)]  # placeholder
    coordinates_3 = [(100,100),(400,100),(400,200),(200,200),(200,300),(500,300)]  # placeholder

    coord_list = [coordinates_1, coordinates_2, coordinates_3]

    depth_list = [100, 50, 20]  # to change

    environments = []

    randomness = random.randint(0, 2)
    for i in range(3):
        depth = depth_list[(i + randomness) % len(depth_list)]
        heightmap, lower_res_heightmap = generate_heightmaps(coord_list[i], depth)
        environments.append((depth, heightmap, lower_res_heightmap, coord_list[i]))

    return environments


def main(environment):
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
    ##################### General Pygame Init #####################
    ##initialize pygame window
    pygame.init()
    window = pygame.display.set_mode((1200, 400))   ##twice 600x400 for haptic and VR
    pygame.display.set_caption('Virtual Haptic Device')

    screenHaptics = pygame.Surface((600,400))
    screenVR = pygame.Surface((600,400))

    ##add nice icon from https://www.flaticon.com/authors/vectors-market
    icon = pygame.image.load('robot.png')
    pygame.display.set_icon(icon)

    ##add text on top to debugToggle the timing and forces
    font = pygame.font.Font('freesansbold.ttf', 18)

    pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' toggles it
    
    ##set up the on-screen debugToggle
    text = font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
    textRect = text.get_rect()
    textRect.topleft = (10, 10)


    xc,yc = screenVR.get_rect().center ##center of the screen


    ##initialize "real-time" clock
    clock = pygame.time.Clock()
    FPS = 100   #in Hertz

    ##define some colors
    cWhite = (255,255,255)
    cDarkblue = (36,90,190)
    cLightblue = (0,176,240)
    cRed = (255,0,0)
    cOrange = (255,100,0)
    cYellow = (255,255,0)

    ####Pseudo-haptics dynamic parameters, k/b needs to be <1
    k = .5      ##Stiffness between cursor and haptic display
    b = .8       ##Viscous of the pseudohaptic display


    ##################### Define sprites #####################

    ##define sprites
    hhandle = pygame.image.load('handle.png')
    haptic  = pygame.Rect(*screenHaptics.get_rect().center, 0, 0).inflate(48, 48)
    cursor  = pygame.Rect(0, 0, 5, 5)
    colorHaptic = cOrange ##color of the wall

    xh = np.array(haptic.center)

    ##Set the old value to 0 to avoid jumps at init
    xhold = 0
    xmold = 0

    ##################### Init Virtual env. #####################


    ##################### Detect and Connect Physical device #####################
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


    CW = 0
    CCW = 1

    haplyBoard = Board
    device = Device
    SimpleActuatorMech = Mechanisms
    pantograph = Pantograph
    robot = PShape
    

    #########Open the connection with the arduino board#########
    port = serial_ports()   ##port contains the communication port or False if no device
    if port:
        print("Board found on port %s"%port[0])
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
        #sys.exit(1)
        

    # conversion from meters to pixels
    window_scale = 3

    ##################### Main Loop #####################
    ##Run the main loop
    run = True
    ongoingCollision = False
    fieldToggle = True
    robotToggle = True
    heightmapToggle = True

    debugToggle = False

    depth, heightmap, lower_res_heightmap, coordinates = environment
    
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
                '''*********** Student can add more ***********'''
                ##Toggle the wall or the height map
                if event.key == ord('h'):
                    heightmapToggle = not heightmapToggle

                '''*********** !Student can add more ***********'''

        ######### Read position (Haply and/or Mouse)  #########
        

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
        
        '''*********** Student should fill in ***********'''
        
        fe = np.zeros(2)  ##Environment force is set to 0 initially.
        ##Replace 
        
        ######### Compute forces ########
        
        # Step 1: Elastic Impedance
        center = [xc, yc]  # Center coordinates
        k_virtual = 0.05  # Virtual stiffness coefficient
        spring_force = -k_virtual * (np.array(center) - np.array(xh))  # Calculate the force applied on the device by the spring
        #fe += spring_force

        # Step 2: Damping and Masses
        b_virtual = 5  # Virtual damping coefficient
        velocity_estimate = (np.array(xh) - np.array(xhold)) / FPS  # Estimate velocity
        damping_force = -b_virtual * velocity_estimate  # Calculate the force applied on the device by the damper based on velocity estimations
        fe += damping_force
        
        # Step 3: Virtual Shapes
        k_field = 0.25 #field stiffness coefficient

        # calculate(based on gradient and add force)
        fe_shape = k_field * compute_gradient_at_position(xh, heightmap)
        fe += fe_shape
        '''*********** !Student should fill in ***********'''
        ##Update old samples for velocity computation
        xhold = xh
        xmold = xm    
        
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
        
            ######### Graphical output #########
        ##Render the haptic surface
        screenHaptics.fill(cWhite)
        
        ##Change color based on effort
        colorMaster = (255,\
            255-np.clip(np.linalg.norm(k*(xm-xh)/window_scale)*15,0,255),\
            255-np.clip(np.linalg.norm(k*(xm-xh)/window_scale)*15,0,255)) #if collide else (255, 255, 255)

            
        pygame.draw.rect(screenHaptics, colorMaster, haptic,border_radius=4)
        

        ######### Robot visualization ###################
        # update individual link position
        if robotToggle:
            robot.createPantograph(screenHaptics,xh)
            
        
        ### Hand visualisation
        screenHaptics.blit(hhandle,(haptic.topleft[0],haptic.topleft[1]))
        pygame.draw.line(screenHaptics, (0, 0, 0), (haptic.center),(haptic.center+2*k*(xm-xh)))
        
        
        ##Render the VR surface
        screenVR.fill(cLightblue)
        '''*********** Student should fill in ***********'''
        ### here goes the visualisation of the VR sceen. 
        
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
        for y, row in enumerate(normalized_heightmap):
            for x, height in enumerate(row):
                color = height_to_color(height)
                pygame.draw.rect(screenVR, color, (y*10 , x*10 , 10, 10))

        # Draw coordinate points
        def draw_points(coordinates, color=(255, 0, 0), radius=5):
            for coord in coordinates:
                pygame.draw.circle(screenVR, color, coord, radius)
        
        draw_points(coordinates)
        
        
        ### Use pygame.draw.rect(screenVR, color, rectangle) to render rectangles. 
        pygame.draw.rect(screenVR, colorHaptic, haptic, border_radius=8)
        
        
        '''*********** !Student should fill in ***********'''


        ##Fuse it back together
        window.blit(screenHaptics, (0,0))
        window.blit(screenVR, (600,0))

        ##Print status in  overlay
        if debugToggle: 
            
            text = font.render("FPS = " + str(round(clock.get_fps())) + \
                                "  xm = " + str(np.round(10*xm)/10) +\
                                "  xh = " + str(np.round(10*xh)/10) +\
                                "  fe = " + str(np.round(10*fe)/10) \
                                , True, (0, 0, 0), (255, 255, 255))
            window.blit(text, textRect)


        pygame.display.flip()    
        ##Slow down the loop to match FPS
        clock.tick(FPS)

    pygame.display.quit()
    pygame.quit()




if __name__ == "__main__":
    
    environments = generate_random_heighmaps()
    
    for environment in environments:
        main(environment)
    