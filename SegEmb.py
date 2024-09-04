import torch
import torch.nn as nn
import numpy as np

class QuadTree(nn.Module):
    def __init__(self, layers, embedding_size=20):
        super(QuadTree, self).__init__()
        self.layers = layers
        self.layers_list = []
        for l in range(layers):
            self.layers_list.append(np.full(4**l, l))
        self.layers_list = np.concatenate(self.layers_list)
        # self.nodes = nn.ParameterList([nn.Parameter(torch.randn(embedding_size)) for _ in range((4 ** layers - 1) // 3)])
        self.len = (4 ** layers - 1) // 3
        self.nodes = nn.Embedding(self.len, embedding_size)
        self.x_y_l_list = []
        for id in range(self.len):
            self.x_y_l_list.append(self.get_node_position_init(id))


    def get_layer(self, id):
        return self.layers_list[id]

    def get_parent_id(self, id):
        if id == 0:
            return None  # Root node has no parent
        return (id - 1) // 4

    def get_children_ids(self, id):
        start = 4 * id + 1
        if start >= self.len:
            return []  # No children if starting index is out of bounds
        return [start, start + 1, start + 2, start + 3]

    def get_node_position_init(self, id):
        layer = self.get_layer(id)
        x, y = 0, 0
        node_id = id

        for l in range(layer+1, 1, -1):
            parent_id = self.get_parent_id(node_id)
            child_index = (node_id - 1) % 4

            if child_index == 0:
                pass  # top-left
            elif child_index == 1:
                x += 2 ** (self.layers - l)  # top-right
            elif child_index == 2:
                y += 2 ** (self.layers - l)  # bottom-left
            elif child_index == 3:
                x += 2 ** (self.layers - l)
                y += 2 ** (self.layers - l)  # bottom-right

            node_id = parent_id

        side_length = 2 ** (self.layers - layer-1)
        return (x, y, side_length)

    def get_node_position(self, id):
        return self.x_y_l_list[id]

    def generate_matrix_for_node(self, id):
        matrix_size=2 ** (self.layers - 1)
        matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        x, y, side_length = self.get_node_position(id)
        matrix[y:y + side_length, x:x + side_length] = 1
        return matrix


    def cover_matrix(self, matrix):
        matrix_size = len(matrix)
        assert len(matrix) != 2**(len(matrix)-1), "illegal input matrix"
        covered = matrix.clone()
        selected_nodes = []
        covered_sum = torch.sum(covered)
        for id in range(self.len):
            x, y, side_length = self.get_node_position(id)
            id_position_matrix = self.generate_matrix_for_node(id).astype(bool)
            if torch.sum(covered[id_position_matrix])==side_length**2:
                selected_nodes.append(id)
                covered[id_position_matrix]=0
            if covered_sum<0:
                raise ValueError("covered_sum should not <0")
            if covered_sum==0:
                break

        return selected_nodes

    def get_embedding_by_matrix(self,matrix):
        selected_nodes = self.cover_matrix(matrix)
        embedding_list = self.nodes(torch.tensor(selected_nodes).cuda())
        layers_list = self.layers_list[selected_nodes]
        coef_list = 0.25**layers_list
        # print(embedding_list.shape)
        # print(coef_list.shape)
        final_embedding = embedding_list*torch.tensor(coef_list).unsqueeze(-1).cuda()
        final_embedding = torch.sum(final_embedding, dim=0)
        return final_embedding


    def get_graph_by_matrixlist(self,matrix_list):
        emb_list = []
        for m in matrix_list:
            emb_list.append(self.get_embedding_by_matrix(m))
        A = torch.stack(emb_list)
        norms = torch.norm(A, p=1, dim=1, keepdim=True)
        normalized_A = A / norms
        G = torch.matmul(normalized_A, normalized_A.T)
        norms = torch.norm(G, p=1, dim=0, keepdim=True)
        normalized_G = G / norms
        return normalized_G


if __name__ == '__main__':

    np.set_printoptions(threshold=np.inf)
    
    # Example usage:
    layers = 3
    matrix_size = 2 ** (layers - 1)
    quadtree = QuadTree(layers)
    
    target_id = 8  # Example node ID to highlight
    print("layer number:",quadtree.get_layer(target_id))
    parent_id = quadtree.get_parent_id(target_id)
    children_ids = quadtree.get_children_ids(target_id)
    
    print(f"Node {target_id} -> Parent ID: {parent_id}, Children IDs: {children_ids}")
    
    matrix = quadtree.generate_matrix_for_node(target_id)
    
    # Print the matrix
    print(matrix)
    input_matrix = np.array(
    [[1 ,1 ,1 ,0],
     [1 ,1 ,0 ,0],
     [0 ,0 ,1 ,1],
     [0 ,0 ,1 ,1],])
    print(quadtree.cover_matrix(input_matrix))
    print(quadtree.nodes(torch.tensor([1, 4, 9])))
    print(quadtree.get_embedding_by_matrix(input_matrix))
    
    input_matrix_list = [np.array(
    [[1 ,1 ,1 ,0],
     [1 ,1 ,0 ,0],
     [0 ,0 ,1 ,1],
     [0 ,0 ,1 ,1],]),np.array(
    [[1 ,1 ,1 ,0],
     [1 ,1 ,0 ,0],
     [0 ,0 ,1 ,1],
     [0 ,0 ,1 ,1],])]
    
    print(quadtree.get_graph_by_matrixlist(input_matrix_list))


