import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch

def align_orientation(vertices1, vertices2):

    def pca(vertices):

        covariance_matrix = np.cov(vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        return eigenvectors


    eigenvectors1 = pca(vertices1)
    eigenvectors2 = pca(vertices2)


    rotation_matrix = np.dot(eigenvectors1, eigenvectors2.T)


    aligned_vertices1 = vertices1 @ rotation_matrix.T
    aligned_vertices2 = vertices2 @ rotation_matrix.T

    return aligned_vertices1, aligned_vertices2

def fscore(dist1, dist2, threshold=0.001):

    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def normalize_mesh(vertices):

    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    scale = np.max(np.linalg.norm(vertices, axis=1))
    return vertices / scale, scale

def normalize_mesh1(vertices):

    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    return vertices

def load_mesh_from_ply(file_path):

    mesh = o3d.io.read_triangle_mesh(file_path)
    return np.asarray(mesh.vertices)

def L1_chamfer_distance(mesh1, mesh2):

    tree1 = cKDTree(mesh1)
    distances1, _ = tree1.query(mesh2)
    avg_distance1 = np.mean(np.abs(distances1))

    tree2 = cKDTree(mesh2)
    distances2, _ = tree2.query(mesh1)
    avg_distance2 = np.mean(np.abs(distances2))

    return avg_distance1 + avg_distance2

def L2_chamfer_distance(mesh1, mesh2):

    tree1 = cKDTree(mesh1)
    distances1, _ = tree1.query(mesh2)
    avg_distance1 = np.mean(distances1)

    tree2 = cKDTree(mesh2)
    distances2, _ = tree2.query(mesh1)
    avg_distance2 = np.mean(distances2)

    return avg_distance1 + avg_distance2

def hausdorff_distance(mesh1, mesh2, norm_type='L2'):

    if norm_type == 'L1':
        dist_func = lambda x, y: np.sum(np.abs(x - y))
    elif norm_type == 'L2':
        dist_func = lambda x, y: np.linalg.norm(x - y)
    else:
        raise ValueError("Invalid norm_type. Use 'L1' or 'L2'.")


    tree1 = cKDTree(mesh1)
    distances1, _ = tree1.query(mesh2)
    max_distance1 = np.max(distances1)


    tree2 = cKDTree(mesh2)
    distances2, _ = tree2.query(mesh1)
    max_distance2 = np.max(distances2)

    return max(max_distance1, max_distance2)


mesh1, scale1 = normalize_mesh(load_mesh_from_ply('.ply'))
mesh2, scale2 = normalize_mesh(load_mesh_from_ply('.ply'))
mesh1 = normalize_mesh1(mesh1)
mesh2 = normalize_mesh1(mesh2)
final_mesh1, final_mesh2 = align_orientation(mesh1, mesh2)

l1_chamfer_dist = L1_chamfer_distance(mesh1, mesh2)

l2_chamfer_dist = L2_chamfer_distance(mesh1, mesh2)



print(f"L1 Chamfer Distance (归一化): {l1_chamfer_dist}")
print(f"L2 Chamfer Distance (归一化): {l2_chamfer_dist}")


tree1 = cKDTree(final_mesh1)
tree2 = cKDTree(final_mesh2)
dist1, _ = tree1.query(final_mesh2)
dist2, _ = tree2.query(final_mesh1)


dist1_tensor = torch.tensor(dist1, dtype=torch.float32).unsqueeze(0)  # 添加一个批量维度
dist2_tensor = torch.tensor(dist2, dtype=torch.float32).unsqueeze(0)  # 添加一个批量维度


fscore_value, precision_1, precision_2 = fscore(dist1_tensor, dist2_tensor, threshold=0.4)


print(f"F-score: {fscore_value.mean().item()}")
print(f"Precision1: {precision_1.mean().item()}")
print(f"Precision2: {precision_2.mean().item()}")

