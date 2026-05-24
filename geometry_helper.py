import numpy as np
from typing import List, Tuple

import pybullet

def dist(
        u: np.ndarray,
        v: np.ndarray
) -> float:
    """
    Calculates the Euclidean distance between two points.

    :param u: First point given by a vector.
    :param v: Second point given by a vector.
    :return: If the distance between u and v is too small, return 0, otherwise, return the distance between u and v.
    """
    uv_path_len = np.linalg.norm(u - v)

    if uv_path_len < 1e-9:
        return 0.0

    return float(uv_path_len)

def quaternion_to_rotation_matrix(
        quaternion: np.ndarray
) -> np.ndarray:
    """
    Calculates the rotation matrix for given quaternion.

    :param quaternion: Orientation (quaternion [x, y, z, w]).
    :return: Rotation matrix (3x3)
    """
    x, y, z, w = quaternion

    # Normalize quaternion
    n = x ** 2 + y ** 2 + z ** 2 + w ** 2
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/np.sqrt(n), y/np.sqrt(n), z/np.sqrt(n), w/np.sqrt(n)

    # Formula quaternion Q = [x, y, z, w] to rotation matrix R(Q)
    rotation_matrix = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])

    return rotation_matrix

def point_to_box_distance(
        point: np.ndarray,
        box_position: np.ndarray,
        box_orientation: np.ndarray,
        box_extents: np.ndarray
) -> float:
    """
    Calculates the distance between a point and primitive box shape.

    :param point: 3D coordinates of a point.
    :param box_position: 3D position of box.
    :param box_orientation: Orientation of box given in quaternion.
    :param box_extents: Box dimensions given in extents.
    :return: Distance between point and primitive box.
    """
    # Get rotation matrix for quaternion
    rotation_matrix = quaternion_to_rotation_matrix(box_orientation)

    # Transform coordinates of point into box frame
    local_point = rotation_matrix.T @ (point - box_position)

    # Calculate distances from box axes
    distances = np.abs(local_point) - box_extents
    x, y, z = distances

    # Calculate final distance
    return float(np.linalg.norm(np.maximum(distances, 0)) + min(max(x, max(y, z)), 0))

def point_to_sphere_distance(
        point: np.ndarray,
        sphere_position: np.ndarray,
        sphere_radius: float
) -> float:
    """
    Calculates the distance between a point and primitive sphere shape.

    :param point: 3D coordinates of a point.
    :param sphere_position: 3D position of sphere.
    :param sphere_radius: Radius of sphere.
    :return: Distance between point and primitive sphere.
    """

    return dist(point, sphere_position) - sphere_radius

def point_to_cylinder_distance(
        point: np.ndarray,
        cylinder_position: np.ndarray,
        cylinder_orientation: np.ndarray,
        cylinder_radius: float,
        cylinder_height: float
) -> float:
    """
    Calculates the distance between a point and primitive cylinder shape.

    :param point: 3D coordinates of a point.
    :param cylinder_position: 3D position of cylinder.
    :param cylinder_orientation: Orientation of cylinder given in quaternion.
    :param cylinder_radius: Radius of cylinder.
    :param cylinder_height: Height of cylinder.
    :return: Distance between point and primitive cylinder.
    """

    # Get rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(cylinder_orientation)

    # Transform coordinates of point into cylinder frame
    local_point = rotation_matrix.T @ (point - cylinder_position)
    x, y, z = local_point

    # Calculate distances for formula
    radial_distance = np.sqrt(x**2 + y**2) - cylinder_radius
    vertical_distance = abs(z) - cylinder_height / 2

    return float(min(max(radial_distance, vertical_distance), 0) + np.sqrt(max(radial_distance, 0) ** 2 + max(vertical_distance, 0) ** 2))

def point_to_object_distance(
        point: np.ndarray,
        geometrical_shape: int,
        position: np.ndarray,
        orientation: np.ndarray,
        dimensions: List[float]
) -> float:
    """
    Calculates distance between a point and an object. If object is mesh, it approximated as sphere and is treated as such.
    The distance is signed, meaning if the point is inside of object, the function returns negative value.

    :param point: 3D coordinates of point.
    :param geometrical_shape: Geometry of object.
    :param position: 3D position of object [x, y, z].
    :param orientation: Orientation of object given in quaternion [x, y, z, w].
    :param dimensions: Dimensions of object depending on its geometry.
    :return: Distance which is positive if the point is outside the object, negative if inside, and 0 if on.
    """

    if geometrical_shape == pybullet.GEOM_BOX:
        return point_to_box_distance(
            point=point,
            box_position=position,
            box_orientation=orientation,
            box_extents=np.array(dimensions, dtype=float)
        )
    elif geometrical_shape == pybullet.GEOM_CYLINDER:
        return point_to_cylinder_distance(
            point=point,
            cylinder_position=position,
            cylinder_orientation=orientation,
            cylinder_height=dimensions[0],
            cylinder_radius=dimensions[1]
        )
    else:
        # For sphere or other object that is approximated by sphere (mesh)
        return point_to_sphere_distance(
            point=point,
            sphere_position=position,
            sphere_radius=dimensions[0]
        )

def distance_based_push(
        point: np.ndarray,
        object_position: np.ndarray,
        object_orientation: np.ndarray,
        object_shape: int,
        object_dimensions: List[float],
        displacement: float = 1e-4
) -> np.ndarray:
    """
    Used to calculate normalized vector that is used to push position of given point based on gradient of our signed
    distance function. We used finite-difference gradient estimation of this function.

    :param point: Point to be moved.
    :param object_position: Object position for SDF.
    :param object_orientation: Object orientation for SDF.
    :param object_shape: Object shape for SDF.
    :param object_dimensions: Object dimensions for SDF.
    :param displacement: Displacement (tiny positive) value for gradient estimation.
    :return: Normalized vector that is used to push position of given point.
    """

    # Create displacement vectors
    x_displacement = np.array([displacement, 0, 0], float)
    y_displacement = np.array([0, displacement, 0], float)
    z_displacement = np.array([0, 0, displacement], float)

    # Use them to estimate gradient vector
    gradient = np.array([
        point_to_object_distance(
            point=point + x_displacement,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        ) - point_to_object_distance(
            point=point - x_displacement,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        ),
        point_to_object_distance(
            point=point + y_displacement,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        ) - point_to_object_distance(
            point=point - y_displacement,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        ),
        point_to_object_distance(
            point=point + z_displacement,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        ) - point_to_object_distance(
            point=point - z_displacement,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        ),
    ], float)

    # Calculate magnitude of gradient and check if it is not too small
    gradient_magnitude = np.linalg.norm(gradient)
    if gradient_magnitude < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    # Normalize gradient and return
    return gradient / gradient_magnitude

def project_point_outside_object(
        point: np.ndarray,
        margin: float,
        object_position: np.ndarray,
        object_orientation: np.ndarray,
        object_shape: int,
        object_dimensions: List[float],
        max_n_iterations: int = 30,
) -> np.ndarray:
    """
    Projects point outside the given object if it is inside meaning distance of the point from the object must be positive.

    Point is pushed outside the object by iteratively pushing it in the direction of gradient of our signed distance function.
    As the function returns negative or 0 values if the point is inside object, moving it in direction of gradient, meaning
    away from object, distance will eventually get into positive numbers, meaning outside the object.

    :param point: 3D coordinates of point.
    :param margin: Safety margin to pad the distance between point and object.
    :param object_position: Object position [x, y, z].
    :param object_orientation: Object orientation given in quaternion [x, y, z, w].
    :param object_shape: Object shape.
    :param object_dimensions: Object dimensions based on shape.
    :param max_n_iterations: Maximum number of iterations/tries to project point outside the object.
    :return: New 3D coordinates of projected point.
    """

    # If no obstacle
    if not object_shape:
        return point

    for _ in range(max_n_iterations):
        # Get distance between point and object
        distance_to_object = point_to_object_distance(
            point=point,
            geometrical_shape=object_shape,
            position=object_position,
            orientation=object_orientation,
            dimensions=object_dimensions
        )

        # If it is above margin, it is safely outside the object
        if distance_to_object >= margin:
            return point

        # Get push in direction of gradient
        normal_push = distance_based_push(
            point=point,
            object_position=object_position,
            object_orientation=object_orientation,
            object_shape=object_shape,
            object_dimensions=object_dimensions,
        )

        # Project point to new position and try again if it is safely outside
        point = point + (margin - distance_to_object + 1e-4) * normal_push

    return point

def perpendicular_basis(
        direction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Used to calculate perpendicular basis of plane orthogonal to given direction.

    :param direction: Vector representing a direction.
    :return: Orthogonal basis.
    """

    # Create helper vector (same as z-axis) to get perpendiculars and check if it is not parallel to given direction
    axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, direction)) > 0.9:
        # If z-axis is too similar, pick y-axis
        axis = np.array([0.0, 1.0, 0.0])

    # Get perpendicular unit vector u to given direction
    u = np.cross(direction, axis)
    u /= (np.linalg.norm(u) + 1e-9)
    # Get another unit vector v perpendicular to both given direction and vector u
    v = np.cross(direction, u)
    v /= (np.linalg.norm(v) + 1e-9)

    return u, v