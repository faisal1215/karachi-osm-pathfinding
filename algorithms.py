from heapq import heappush, heappop

def bfs(graph, start, goal):
    queue = [[start]]
    visited = set()

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                queue.append(path + [neighbor])
    return None


def dfs(graph, start, goal):
    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                stack.append(path + [neighbor])
    return None


def astar(graph, start, goal):
    pq = []
    heappush(pq, (0, start, [start]))
    visited = set()

    while pq:
        cost, node, path = heappop(pq)

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                edge_data = list(graph.get_edge_data(node, neighbor).values())[0]
                length = edge_data.get("length", 1)
                heappush(pq, (cost + length, neighbor, path + [neighbor]))

    return None
