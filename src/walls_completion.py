import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from alphashape import alphashape
import shapely.geometry as sg


ALPHA_INIT = 2  # The bigger alpha is, the tighter the contour,
# but the more likely we can have multiple polygons in the same layer,
# which we don't want in v1
SPHERE_RADIUS_SCALING_FACTOR = 1.5


def get_best_planes(point_cloud: np.ndarray):
    # We're going to compute the silhouette score
    # for each number of clusters between 2 and 10
    planes_for_n = []
    silhouette_scores = []

    for n_clusters in range(2, 10):
        z_coords = point_cloud[:, 2].reshape(-1, 1)
        # Convert n_clusters to an integer
        n_clusters = int(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            z_coords
        )

        # Now print the points in each cluster
        planes = [[] for _ in range(n_clusters)]

        for i in range(len(kmeans.labels_)):
            planes[kmeans.labels_[i]].append(point_cloud[i])

        planes_arrs = [np.array(plane) for plane in planes]

        silhouette_scores.append(silhouette_score(z_coords, kmeans.labels_))
        planes_for_n.append(planes_arrs)

    # Now we want to find the number of clusters that maximizes the silhouette score
    best_silhouette = np.argmax(silhouette_scores)
    best_planes = planes_for_n[best_silhouette]

    # Sort the best_planes by the median z coordinate of each plane
    best_planes = sorted(best_planes, key=lambda x: np.median(x[:, 2]))

    best_planes[0] = np.array(
        [p for p in best_planes[0] if p[2] == np.median(best_planes[0][:, 2])]
    )

    return best_planes


def compute_contours(planes):
    planes_2d = [plane[:, :2] for plane in planes]
    alpha_shapes = []

    for plane_2d in planes_2d:
        found_one_shape = False

        alpha = ALPHA_INIT

        while not found_one_shape:
            if alpha < 0.01:
                # Then there's clearly a problem with the layer. Stop and print it
                print("Alpha value too small, skipping this layer...")
                exit(1)

            alpha_shape = alphashape(plane_2d, alpha)
            if alpha_shape.geom_type == "Polygon":
                found_one_shape = True
                break
            else:  # We have a MultiPolygon, this should not happen in this version
                # we probably took a too big alpha value. Let's try with a smaller one.
                print("MultiPolygon found, trying with a smaller alpha value...")
                alpha /= 2

        alpha_shapes.append(alpha_shape)

    contour_planes_2d = [
        np.array(list((zip(a_s.exterior.xy[0], a_s.exterior.xy[1]))))
        for a_s in alpha_shapes
    ]
    contour_planes_3d = [
        np.array(
            [
                plane[np.argmin(np.linalg.norm(plane_2d - point, axis=1))]
                for point in contour_plane_2d
            ]
        )
        for plane, plane_2d, contour_plane_2d in zip(
            planes, planes_2d, contour_planes_2d
        )
    ]

    return contour_planes_3d


def link_contours(
    point_cloud,
    planes,
    contour_planes_3d,
):
    # Since the building footprint was generated with a very regular grid, we can
    # assume that the distance between points in the contour is the same as
    # the distance between points in the lower plane contour
    # (if it was accurately computed).
    average_point_distance = np.mean(
        np.linalg.norm(contour_planes_3d[0][1:] - contour_planes_3d[0][:-1], axis=1)
    )

    # Now because we should a square mesh, the sphere radius should be this average
    # point distance divided by the square root of 2 divided by 2.
    sphere_radius = average_point_distance * 2 / np.sqrt(2)
    # COULD BE DIFFERENT. 30cm? A STATIC VAR? ALGORITHM HYPER-PARAMETER?

    # As per code in the notebook above, we get better results by
    # adjusting the sphere radius by a factor of 1.5
    sphere_radius_adjusted = sphere_radius * SPHERE_RADIUS_SCALING_FACTOR

    final_points = list(point_cloud.copy())

    for plane_idx, plane_contour in enumerate(contour_planes_3d):
        for point in plane_contour:
            for plane_above_idx, plane_above_contour in enumerate(
                contour_planes_3d[plane_idx + 1 :]
            ):
                has_found_point_in_plane_above = False
                plane_above = planes[plane_above_idx + plane_idx + 1]
                for potential_point_above in plane_above:
                    if (
                        np.linalg.norm(potential_point_above[:2] - point[:2])
                        < sphere_radius_adjusted
                    ):
                        # Then we found a point in the plane above that is within
                        # the sphere. So we complete the structure by going up to
                        # the plane above, from the point in the lower plane, by
                        # adding new points every average_point_distance
                        # in the z direction.
                        # We do this until we reach the height of point2.
                        for z in np.arange(
                            point[2], potential_point_above[2], average_point_distance
                        )[1:]:
                            final_points.append(np.array([point[0], point[1], z]))
                        has_found_point_in_plane_above = True
                        break
                if has_found_point_in_plane_above:
                    break

    final_points_arr = np.array(final_points)

    return final_points_arr


def complete_structures(point_cloud: np.ndarray):
    print("Computing best planes...")
    best_planes = get_best_planes(point_cloud)
    print(f"Determined there are {len(best_planes)} main planes. Computing contours...")
    contour_planes_3d = compute_contours(best_planes)
    print("Contours computed. Linking planes...")
    final_points = link_contours(point_cloud, best_planes, contour_planes_3d)
    print("Structure completed!")
    return final_points
