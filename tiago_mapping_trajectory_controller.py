import numpy as np
from controller import Robot, GPS, Compass, Display, Supervisor, Lidar
import math
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy import signal

# Initialize variables
TIME_STEP = 32
MAX_SPEED = 12.0  # Maximum speed of the robot
WHEEL_RADIUS = 0.0985  # Radius of the robot's wheels
WHEEL_BASE = 0.408  # Distance between the wheels
MAP_SIZE_X = 500  # Width of the map
MAP_SIZE_Y = 650  # Height of the map
MIN_SPEED = 2.0  # Minimum speed of the robot
SHIELDED_READINGS = 80  # Number of shielded LIDAR readings to ignore
ROBOT_RADIUS = 0.265  # Radius of the robot's base
KERNEL_SIZE = 25  # Kernel size for obstacle inflation

# Create the Robot instance
robot = Supervisor()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Waypoints for the robot to navigate
WP = [(0.5, 0.0), (0.4, -3.0), (-1.7, -3.0), (-1.55, 0.1), (0.0, 0.1), (0, 0), (0.0, 0.1), (-1.55, 0.1), (-1.7, -3.0),
      (0.4, -3.0), (0.5, 0.0), (0, 0)]
index = 0
forward = True

# Initialize devices
gps = robot.getDevice('gps')
gps.enable(timestep)

compass = robot.getDevice('compass')
compass.enable(timestep)

lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

display = robot.getDevice('display')

leftMotor = robot.getDevice('wheel_left_joint')
rightMotor = robot.getDevice('wheel_right_joint')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize the map
prob_map = np.zeros((MAP_SIZE_X, MAP_SIZE_Y))
obstacle_map = np.zeros((MAP_SIZE_X, MAP_SIZE_Y))
trajectory_map = np.zeros((MAP_SIZE_X, MAP_SIZE_Y))
map = np.zeros((MAP_SIZE_X, MAP_SIZE_Y))


# Convert world coordinates to map coordinates
def world2map(xw, yw, map_size_x=MAP_SIZE_X, map_size_y=MAP_SIZE_Y, world_size_x=5.0, world_size_y=6.5, center_x=0.0,
              center_y=-0.75):
    px = int((xw - center_x + world_size_x / 2) * map_size_x / world_size_x)
    py = int((yw - center_y + world_size_y / 2) * map_size_y / world_size_y)
    px = max(0, min(map_size_x - 1, px))
    py = map_size_y - 1 - max(0, min(map_size_y - 1, py))  # Flip Y coordinate
    return px, py


# Normalize an angle to the range [-pi, pi]
def normalize_angle(angle):
    return ((angle + math.pi) % (2 * math.pi)) - math.pi


# Compute the difference between two angles
def angle_difference(angle1, angle2):
    return normalize_angle(angle1 - angle2)


# Compute the configuration space by inflating obstacles
def compute_cspace(obstacle_map, kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    cmap = signal.convolve2d(obstacle_map, kernel, mode='same')
    cspace = cmap > 0.9  # Threshold the convolved map
    return cspace


# Main loop
while robot.step(timestep) != -1:
    xw = gps.getValues()[0]
    yw = gps.getValues()[1]
    compass_values = compass.getValues()
    theta = np.arctan2(compass_values[0], compass_values[1])

    if robot.getTime() < timestep / 1000.0:  # Check if it's the first step
        print(f"Initial Position: x = {xw:.4f}, y = {yw:.4f}, theta = {theta:.4f}")

    # Check if the robot should move forward or backward through waypoints
    if forward and index == len(WP):
        forward = False
        index -= 2
    elif not forward and index < 0:
        forward = True
        index += 2

    target = WP[index]

    # Calculate distance and heading errors
    rho = np.sqrt((xw - target[0]) ** 2 + (yw - target[1]) ** 2)
    alpha = angle_difference(np.arctan2(target[1] - yw, target[0] - xw), theta)

    if rho < 0.1:  # Switch to the next waypoint if close enough
        index = index + 1 if forward else index - 1

    print(f"Position: x = {xw:.4f}, y = {yw:.4f}")
    print(f"Distance Error: {rho}")
    print(f"Heading Error: {alpha}")
    print(f"Compass readings: X = {compass_values[0]:.4f}, Y = {compass_values[1]:.4f}, Z = {compass_values[2]:.4f}")
    print(f"Orientation (theta) = {theta:.4f} radians = {np.degrees(theta):.2f} degrees")

    ranges = np.array(lidar.getRangeImage())
    ranges[ranges == np.inf] = 0
    num_points = len(ranges)

    if num_points > 0:
        # Drop the first 80 and last 80 readings
        ranges = ranges[SHIELDED_READINGS:-SHIELDED_READINGS]
        num_points = len(ranges)

        # Recalculate the angles for the reduced field of view
        fov = 4.18879  # Original field of view in radians
        angles = np.linspace(-fov / 2, fov / 2, 667)  # Original angles
        angles = angles[SHIELDED_READINGS:-SHIELDED_READINGS]  # Drop the first and last 80 readings

        # Transform points to world coordinates
        w_T_r = np.array([
            [np.cos(theta), -np.sin(theta), xw],
            [np.sin(theta), np.cos(theta), yw],
            [0, 0, 1]
        ])

        # Calculate the coordinates of the LIDAR points in the robot's frame
        X_r = np.vstack((ranges * np.cos(angles), -ranges * np.sin(angles), np.ones(num_points)))
        D = w_T_r @ X_r

        x_w = D[0, :]
        y_w = D[1, :]

        # Update the obstacle map and display the points
        for i in range(num_points):
            px, py = world2map(x_w[i], y_w[i])
            if trajectory_map[px, py] == 0:
                obstacle_map[px, py] = min(obstacle_map[px, py] + 0.01, 1.0)  # Reduced increment to 0.01
                map[px, py] = min(map[px, py] + 0.01, 1.0)  # Update the map
                v = int(map[px, py] * 255)
                color = (v * 256 ** 2 + v * 256 + v)
                display.setColor(color)
                display.drawPixel(px, py)

    # Update the trajectory map with the robot's current position
    px, py = world2map(xw, yw)
    trajectory_map[px, py] = 1
    display.setColor(0xFF0000)  # Red for the robot's path
    display.drawPixel(px, py)

    # Adjusting control gains for more precise turns
    p1 = 5.0  # Increased gain for more aggressive turning
    p2 = 1.0  # Slightly reduced gain for linear movement

    leftSpeed = -alpha * p1 + rho * p2
    rightSpeed = alpha * p1 + rho * p2

    # Ensure minimum speeds
    if abs(leftSpeed) < MIN_SPEED:
        leftSpeed = np.sign(leftSpeed) * MIN_SPEED
    if abs(rightSpeed) < MIN_SPEED:
        rightSpeed = np.sign(rightSpeed) * MIN_SPEED

    # Limit speeds to maximum allowed
    leftSpeed = max(min(leftSpeed, MAX_SPEED), -MAX_SPEED)
    rightSpeed = max(min(rightSpeed, MAX_SPEED), -MAX_SPEED)

    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

    # Check if the robot has reached the final waypoint to compute the C-space
    if index == len(WP) - 1 and rho < 0.1:
        cspace_map = compute_cspace(obstacle_map, KERNEL_SIZE)

        # Plot the C-space map with appropriate colors
        cspace_colormap = np.zeros((MAP_SIZE_X, MAP_SIZE_Y, 3), dtype=np.uint8)
        for i in range(MAP_SIZE_X):
            for j in range(MAP_SIZE_Y):
                if cspace_map[i, j]:
                    cspace_colormap[i, j] = [255, 255, 0]  # Yellow for inflated obstacles
                elif obstacle_map[i, j] > 0:
                    cspace_colormap[i, j] = [255, 255, 255]  # White for obstacles
                else:
                    cspace_colormap[i, j] = [0, 0, 255]  # Blue for free space

        plt.imshow(cspace_colormap)
        plt.title('Configuration Space Map')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.savefig('../../cspace_map.png', bbox_inches='tight', pad_inches=0)
        plt.show()
        break  # Stop the robot after computing the C-space
