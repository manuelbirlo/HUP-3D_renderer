import numpy as np

class CameraSphere:

    def __init__(self, sphere_radius, circle_radius):
        self.sphere_radius = sphere_radius
        self.circle_radius = circle_radius
        self.max_latitude_floors = int(np.ceil(np.pi / (2 * np.arcsin(self.circle_radius / self.sphere_radius))))

    def spherical_to_camera_view(self, camera_locations):
        """
        Convert spherical coordinates to camera view roll, pitch, and yaw angles.
        :param sensor_coordinates: List of (theta, phi) pairs representing spherical coordinates.
        :return: List of (roll, pitch, yaw) pairs representing camera view angles.
        """
        camera_view_angles = []

        for theta, phi in camera_locations:
            # Convert spherical coordinates to Cartesian coordinates
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

        # Convert Cartesian coordinates to camera view roll, pitch, and yaw angles
            roll = np.arctan2(y, x)
            pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
            yaw = -np.arctan2(np.sin(roll) * z, np.cos(roll) * x - np.sin(roll) * y)

            # Convert angles to degrees and ensure they are in the range [0, 360)
            roll_deg = np.degrees(roll) % 360
            pitch_deg = np.degrees(pitch) % 360
            yaw_deg = np.degrees(yaw) % 360

            # Append the camera view angles to the result list
            camera_view_angles.append((roll_deg, pitch_deg, yaw_deg))

        return camera_view_angles

    def generate_spherical_camera_locations(self):
        camera_locations_dict = {}

        # Add the 'north pole' sensor location at the top of the sphere (theta = 0 , phi = 0)
        camera_locations_dict[0] = [(0, 0)]

        for i in range(1, self.max_latitude_floors):  # Start from 1 since the north pole is already added
            theta = i * np.pi / self.max_latitude_floors

            # Calculate the number of circles that can fit on the current latitude floor
            num_circles_per_floor = int(np.floor(2 * np.pi * self.sphere_radius * np.sin(theta) / (2 * self.circle_radius)))

            phi_values = np.linspace(0, 2 * np.pi, num_circles_per_floor, endpoint=False)

            # Add the sensor locations to the dictionary under the current latitude floor key
            camera_locations_dict[i] = [(theta, phi) for phi in phi_values]

        # Add the 'south pole' sensor location at the bottom of the sphere (theta = pi , phi = 0)
        camera_locations_dict[self.max_latitude_floors] = [(np.pi, 0)]

        return camera_locations_dict

    def generate_blender_camera_view_angles(self):

        camera_view_angles_per_floor = {}

        camera_locations_dict = self.generate_spherical_camera_locations()

        for floor, camera_locations in camera_locations_dict.items():
            camera_view_angles_of_current_floor = self.spherical_to_camera_view(camera_locations)

            blender_converted_camera_angles = []

            # Consider latitude floors above the sphere's equator (+ 1 due to the added latitude floor 0)
            if floor <= self.max_latitude_floors / 2 + 1:
                for camera_angles in camera_view_angles_of_current_floor:
                    roll, pitch, yaw = camera_angles
                    converted_angles = (np.radians(360.0 - pitch), np.radians(360.0 - roll), np.radians(0))
                    blender_converted_camera_angles.append(converted_angles)
            else:
                for camera_angles in camera_view_angles_of_current_floor:
                    roll, pitch, yaw = camera_angles
                    converted_angles = (np.radians(-pitch), np.radians(360.0 - roll), np.radians(0))
                    blender_converted_camera_angles.append(converted_angles)

            camera_view_angles_per_floor[floor] = blender_converted_camera_angles
        
        return camera_view_angles_per_floor

            

    
