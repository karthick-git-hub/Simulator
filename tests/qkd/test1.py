def generate_adjacency_matrix(topology, m):
    n = m;
    size = n * n

    if (topology == 'Grid') or (topology == 'grid'):
        adj_matrix = np.zeros((size, size))
        for row in range(n):
            for col in range(n):
                index = row * n + col

                # Connect to the right
                if col < n - 1:
                    adj_matrix[index][index + 1] = 1
                    adj_matrix[index + 1][index] = 1

                # Connect downwards
                if row < n - 1:
                    adj_matrix[index][index + n] = 1
                    adj_matrix[index + n][index] = 1
        return adj_matrix

    elif (topology == 'ring') or (topology == 'Ring'):
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            adjacency_matrix[i][(i - 1) % n] = 1  # Connect to previous node
            adjacency_matrix[i][(i + 1) % n] = 1  # Connect to next node
        return adjacency_matrix

    elif (topology == 'star') or (topology == 'Star'):
        adjacency_matrix = np.zeros((n + 1, n + 1))
        for i in range(1, n + 1):
            adjacency_matrix[i][0] = adjacency_matrix[0][i] = 1
        return adjacency_matrix

    elif (topology == 'torus') or (topology == 'Torus'):
        totalNodes = size
        totalNodesX = n
        totalNodesY = n
        adjacencyMatrix = np.zeros((totalNodes, totalNodes))

        for i in range(totalNodesX):
            for j in range(totalNodesY):
                node = i * totalNodesY + j

                # Calculate the left neighbor
                if j == 0:
                    leftNeighbor = node + (totalNodesY - 1)
                else:
                    leftNeighbor = node - 1

                # Calculate the upper neighbor
                if i == 0:
                    upperNeighbor = node + (totalNodesX - 1) * totalNodesY
                else:
                    upperNeighbor = node - totalNodesY

                # Update the adjacency matrix
                adjacencyMatrix[node, leftNeighbor] = 1
                adjacencyMatrix[leftNeighbor, node] = 1
                adjacencyMatrix[node, upperNeighbor] = 1
                adjacencyMatrix[upperNeighbor, node] = 1

        return adjacencyMatrix

def get_nodes(topology, n):
    topology = topology.lower()
    if topology in ['ring', 'grid', 'torus']:
        if topology == 'ring':
            adj_matrix = generate_adjacency_matrix('Ring', n)
            start_node = 0
            end_node = n // 2
        elif topology == 'grid':
            adj_matrix = generate_adjacency_matrix('Grid', n)
            start_node = 0
            end_node = n * n - 1
        elif topology == 'torus':
            adj_matrix = generate_adjacency_matrix('Torus', n)
            start_node = 0
            end_node = n * n - 1
        nodes_in_between = shortest_path_length_bfs(adj_matrix, start_node, end_node)
        return nodes_in_between
    else:
        raise ValueError('Error in topology: use either of ring, grid, or torus topology')

def shortest_path_length_bfs(adjacency_matrix, start_node, end_node):
    """
    Find the length of the shortest path in an unweighted graph using BFS.

    Args:
    adjacency_matrix (numpy.ndarray): The adjacency matrix of the graph.
    start_node (int): The starting node index.
    end_node (int): The ending node index.

    Returns:
    int: The number of nodes in between the start_node and end_node in the shortest path.
    """
    num_nodes = adjacency_matrix.shape[0]
    visited = [False] * num_nodes
    prev = [None] * num_nodes

    # BFS
    queue = deque([start_node])
    visited[start_node] = True

    while queue:
        node = queue.popleft()

        # Visit the neighbors
        for neighbor, connected in enumerate(adjacency_matrix[node]):
            if connected and not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                prev[neighbor] = node

    # Reconstruct the path
    path = []
    at = end_node
    while at is not None:
        path.append(at)
        at = prev[at]
    path.reverse()

    # Return the number of nodes in between start and end, excluding start and end
    nonodes = len(path) - 2 if path[0] == start_node and len(path) > 1 else 0
    return nonodes
