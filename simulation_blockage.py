import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque

INF = 10**9

class Edge:
    def __init__(self, from_, to, capacity, cost):
        self.from_ = from_
        self.to = to
        self.capacity = capacity
        self.cost = cost

# Function to find the shortest path in a network
def shortest_paths(n, v0, adj, cost, capacity):
    # Initialize distance, queue, predecessors, and counters
    distance = [INF] * n  # Initialize distances to infinity
    distance[v0] = 0  # Set the distance to the source node as 0
    inq = [False] * n  # Keep track of nodes in the queue
    q = deque([v0])  # Initialize queue with the source node
    pre = [-1] * n  # Predecessor array to reconstruct the path
    count = [0] * n  # Counter for negative cycle detection

    # Shortest path algorithm using Bellman-Ford
    while q:
        u = q.popleft()  # Dequeue a node
        inq[u] = False  # Mark it as not in the queue
        for v in adj[u]:  # Iterate through neighboring nodes
            # Relaxation step: update distance if shorter path found
            if capacity[u][v] > 0 and distance[v] > distance[u] + cost[u][v]:
                distance[v] = distance[u] + cost[u][v]  # Update distance
                pre[v] = u  # Update predecessor
                if not inq[v]:  # Add to queue if not already in
                    q.append(v)
                    inq[v] = True
                count[v] += 1  # Increment counter for negative cycle
                if count[v] > n:  # Check for negative cycle
                    return None  # If negative cycle detected

    return distance, pre  # Return distances and predecessors


# Function to compute minimum cost flow in a network
def min_cost_flow(N, edges, K, s, t):
    # Initialize adjacency list, cost, and capacity matrices
    adj = [[] for _ in range(N)]
    cost = [[0] * N for _ in range(N)]
    capacity = [[0] * N for _ in range(N)]

    # Populate adjacency list, cost, and capacity matrices
    for e in edges:
        adj[e.from_].append(e.to)
        adj[e.to].append(e.from_)
        cost[e.from_][e.to] = e.cost
        cost[e.to][e.from_] = -e.cost
        capacity[e.from_][e.to] = e.capacity

    flow = 0  # Initialize flow
    cost_ = 0  # Initialize cost of flow

    while flow < K:
        result = shortest_paths(N, s, adj, cost, capacity)
        if result is None:
            raise ValueError("Negative Cycle")

        d, p = result  # Retrieve distances and predecessors

        if d[t] == INF:
            break  # No path found to the target node

        f = K - flow  # Calculate remaining flow to be sent
        cur = t  # Current node initialized as target
        path = []  # Path to store the route

        # Reconstruct the path from target to source
        while cur != s:
            f = min(f, capacity[p[cur]][cur])  # Find minimum capacity on the path
            path.append(cur)  # Append nodes to the path
            cur = p[cur]  # Move to the predecessor
            
        path.append(s)
        path = path[::-1]  # Reverse the path to get source to target

        flow += f  # Update total flow
        cost_ += f * d[t]  # Update total cost of flow

        # Update the capacities in the network
        cur = t  # Reset current node as target
        
        # Display information about the current route
        print('Route:', ' -> '.join(map(str, path)))
        print('Flow on this route:', f)
        print('Cost per flow on this route:', d[t], "\n")
        
        while cur != s:
            capacity[p[cur]][cur] -= f  # Reduce capacity on the path
            capacity[cur][p[cur]] += f  # Increase reverse capacity
            cur = p[cur]  # Move to the predecessor

    if flow < K:
        return -1  # If unable to send the required flow
    else:
        return cost_  # Return the total cost of sending the flow

# create a 2d 5x10 grid graph
G = nx.grid_2d_graph(5, 10)

# add random cost and capacity for each edge
for (u, v) in G.edges():
    G.edges[u, v]['cost'] = random.randint(1, 100)
    G.edges[u, v]['capacity'] = random.randint(700, 2000)


# Convert edges into a list to ensure a sequence-like object
edges_list = list(G.edges())

# Introduce blocked routes randomly
num_blocked = random.randint(5, 15)  # Define the number of edges to block
blocked_edges = random.sample(edges_list, num_blocked)  # Select random edges to block

for (u, v) in blocked_edges:
    G.edges[u, v]['capacity'] = 0  # Set capacity to 0 for blocked edges


mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

pos = {i: (i % 10, 4 - i // 10) for i in range(50)}

source_node = 0
target_node = 49
node_colors = ['orange' if node == source_node else 'lightgreen' if node == target_node else 'lightgray' for node in G.nodes()]

# Set the figure size before displaying the graph
plt.figure(figsize=(14, 6))  # Set width and height in inches

# Define edge colors based on capacity
edge_colors = ['red' if G.edges[u, v]['capacity'] == 0 else 'black' for u, v in G.edges()]

nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=8, edge_color=edge_colors)

edge_labels = {(u, v): f"{G.edges[u, v]['cost']} | {G.edges[u, v]['capacity']}" for u, v in G.edges()}


nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', rotate=False)

edges = []
for u, v, attr in G.edges(data=True):
    edges.append(Edge(u, v, attr['capacity'], attr['cost']))
    edges.append(Edge(v, u, 0, -attr['cost']))

source = 0
destination = 49

# total flow to be sent
K = 500

# number of nodes
N = len(G.nodes())

cost = min_cost_flow(N, edges, K, source, destination)
print("Minimum cost considering flow:", cost)

# show the graph
plt.show()
