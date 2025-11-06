import numpy as np
import trimesh
from scipy.spatial import cKDTree


def normalize_mesh1(vertices):

    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    scale = np.max(np.linalg.norm(vertices, axis=1))
    return vertices / scale, scale

def normalize_mesh2(vertices):

    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    return vertices

def compute_accuracy(gt_mesh, pred_mesh, num_samples=5000):

    points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)


    pred_vertices = np.asarray(pred_mesh.vertices)
    pred_tree = cKDTree(pred_vertices)

    distances, _ = pred_tree.query(points)

    return np.mean(np.abs(distances))


def compute_completion(gt_mesh, pred_mesh, num_samples=5000):

    points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)


    gt_vertices = np.asarray(gt_mesh.vertices)
    gt_tree = cKDTree(gt_vertices)


    distances, _ = gt_tree.query(points)

    return np.mean(np.abs(distances))


def compute_f_score(gt_mesh, pred_mesh, num_samples=5000, threshold=0.05):


    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)


    gt_tree = cKDTree(np.asarray(gt_mesh.vertices))
    pred_tree = cKDTree(np.asarray(pred_mesh.vertices))


    gt_distances, _ = pred_tree.query(gt_points)

    pred_distances, _ = gt_tree.query(pred_points)


    recall = np.mean(gt_distances < threshold)
    precision = np.mean(pred_distances < threshold)

    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f_score


def compute_chamfer_distance(gt_mesh, pred_mesh):

    gt_points = gt_mesh.vertices
    pred_points = pred_mesh.vertices


    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)


    gt_to_pred_dist, _ = pred_tree.query(gt_points)

    pred_to_gt_dist, _ = gt_tree.query(pred_points)


    chamfer_dist = np.mean(gt_to_pred_dist ** 2) + np.mean(pred_to_gt_dist ** 2)
    return chamfer_dist

def normalize_and_align(gt_mesh, pred_mesh):


    gt_bounds = gt_mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    gt_center = (gt_bounds[0] + gt_bounds[1]) / 2
    gt_size = np.max(gt_bounds[1] - gt_bounds[0])


    pred_bounds = pred_mesh.bounds
    pred_center = (pred_bounds[0] + pred_bounds[1]) / 2
    pred_size = np.max(pred_bounds[1] - pred_bounds[0])


    scale_factor = gt_size / pred_size


    pred_mesh.apply_translation(-pred_center)
    pred_mesh.apply_scale(scale_factor)
    pred_mesh.apply_translation(gt_center)

    return gt_mesh, pred_mesh


def align_and_scale_mesh(mesh, target_scale=1.0):


    vertices = mesh.vertices


    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center


    scale = np.max(np.ptp(vertices_centered, axis=0))
    vertices_scaled = vertices_centered * (target_scale / scale)

    mesh.vertices = vertices_scaled
    return mesh


def compute_completion_ratio(gt_mesh, pred_mesh, num_samples=5000, threshold=0.15):


    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)


    gt_tree = cKDTree(np.asarray(gt_mesh.vertices))


    distances, _ = gt_tree.query(pred_points)


    completion_ratio = np.mean(distances < threshold)

    return completion_ratio

# 示例使用
if __name__ == "__main__":

    gt_mesh = trimesh.load('ship4.ply')
    pred_mesh = trimesh.load('gof_ship4.ply')
    #gt_mesh, pred_mesh = normalize_and_align(gt_mesh, pred_mesh)

    target_scale = 1
    gt_mesh_aligned = align_and_scale_mesh(gt_mesh, target_scale)
    pred_mesh_aligned = align_and_scale_mesh(pred_mesh, target_scale)


    accuracy = compute_accuracy(gt_mesh, pred_mesh)
    completion = compute_completion(gt_mesh, pred_mesh)
    f_score = compute_f_score(gt_mesh, pred_mesh)
    chamfer_dist = compute_chamfer_distance(gt_mesh, pred_mesh)
    completion_ratio = compute_completion_ratio(gt_mesh, pred_mesh)

    print(f"Accuracy: {accuracy}")
    print(f"Completion: {completion}")
    print(f"F-Score: {f_score}")
    print(f"Chamfer Distance: {chamfer_dist}")
    print(f"Completion Ratio: {completion_ratio}")