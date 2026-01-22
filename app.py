import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import math
from heapq import heappush, heappop

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Karachi Pathfinding (OSM)",
    layout="wide"
)

st.title("🚦 Karachi Pathfinding using OpenStreetMap")
st.markdown("Select **Start → Goal**, then choose an algorithm")

# -------------------------------
# SESSION STATE (IMPORTANT)
# -------------------------------
if "start" not in st.session_state:
    st.session_state.start = None
if "goal" not in st.session_state:
    st.session_state.goal = None
if "map" not in st.session_state:
    st.session_state.map = None

# -------------------------------
# LOAD KARACHI GRAPH (CACHED)
# -------------------------------
@st.cache_resource
def load_graph():
    G = ox.graph_from_place(
        "Karachi, Pakistan",
        network_type="drive",
        simplify=True
    )
    G = ox.convert.to_undirected(G)
    return G

graph = load_graph()

# -------------------------------
# ALGORITHMS
# -------------------------------
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


def heuristic(graph, a, b):
    x1, y1 = graph.nodes[a]["x"], graph.nodes[a]["y"]
    x2, y2 = graph.nodes[b]["x"], graph.nodes[b]["y"]
    return math.dist((x1, y1), (x2, y2))


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
                edge_data = graph.get_edge_data(node, neighbor)
                length = min(
                    d.get("length", 1) for d in edge_data.values()
                )
                priority = cost + length + heuristic(graph, neighbor, goal)
                heappush(pq, (priority, neighbor, path + [neighbor]))
    return None

# -------------------------------
# SIDEBAR
# -------------------------------
algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["BFS", "DFS", "A*"]
)

if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

# -------------------------------
# BASE MAP
# -------------------------------
base_map = folium.Map(
    location=[24.8607, 67.0011],
    zoom_start=12
)

st.markdown("### 🗺️ Click on the map to select **Start** and **Goal**")

map_data = st_folium(
    base_map,
    height=520,
    width=950
)

# -------------------------------
# CAPTURE MAP CLICKS
# -------------------------------
if map_data and map_data["last_clicked"]:
    click = map_data["last_clicked"]

    if st.session_state.start is None:
        st.session_state.start = click
        st.success("✅ Start point selected")

    elif st.session_state.goal is None:
        st.session_state.goal = click
        st.success("🎯 Goal point selected")

# -------------------------------
# FIND PATH BUTTON
# -------------------------------
if st.button("🚀 Find Path"):
    if st.session_state.start and st.session_state.goal:

        start = st.session_state.start
        goal = st.session_state.goal

        start_node = ox.distance.nearest_nodes(
            graph, start["lng"], start["lat"]
        )
        goal_node = ox.distance.nearest_nodes(
            graph, goal["lng"], goal["lat"]
        )

        if algo == "BFS":
            path = bfs(graph, start_node, goal_node)
        elif algo == "DFS":
            path = dfs(graph, start_node, goal_node)
        else:
            path = astar(graph, start_node, goal_node)

        if path:
            m = folium.Map(
                location=[24.8607, 67.0011],
                zoom_start=12
            )

            coords = [
                (graph.nodes[n]["y"], graph.nodes[n]["x"])
                for n in path
            ]

            folium.PolyLine(
                coords,
                color="blue",
                weight=5
            ).add_to(m)

            folium.Marker(
                coords[0],
                icon=folium.Icon(color="green"),
                tooltip="Start"
            ).add_to(m)

            folium.Marker(
                coords[-1],
                icon=folium.Icon(color="red"),
                tooltip="Goal"
            ).add_to(m)

            st.session_state.map = m
            st.success("✅ Path found successfully!")

        else:
            st.error("❌ No path found")

    else:
        st.warning("⚠️ Please select start and goal points")

# -------------------------------
# DISPLAY FINAL MAP (PERSISTENT)
# -------------------------------
if st.session_state.map is not None:
    st_folium(
        st.session_state.map,
        height=520,
        width=950
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "📍 **Data Source:** OpenStreetMap | "
    "🧠 **Algorithms:** BFS, DFS, A* | "
    "🎓 **AI Project**"
)
