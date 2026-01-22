# =========================================================
# Karachi OSM Pathfinding App
# BFS | DFS | A* using OpenStreetMap
# =========================================================

import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Karachi OSM Pathfinding",
    page_icon="🌆",
    layout="wide"
)

st.title("🌆 Karachi Pathfinding using OpenStreetMap")
st.markdown("**BFS | DFS | A*** on Karachi Road Network")

# ---------------------------------------------------------
# Load Karachi Graph
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_karachi_graph():
    G = ox.graph_from_place(
        "Karachi, Pakistan",
        network_type="drive",
        simplify=True
    )
    return G


G = load_karachi_graph()

# ---------------------------------------------------------
# Algorithms
# ---------------------------------------------------------
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
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                stack.append((neighbor, path + [neighbor]))
    return None


def astar(graph, start, goal):
    return nx.astar_path(
        graph,
        start,
        goal,
        heuristic=lambda a, b: ox.distance.euclidean_dist_vec(
            graph.nodes[a]["y"], graph.nodes[a]["x"],
            graph.nodes[b]["y"], graph.nodes[b]["x"]
        ),
        weight="length"
    )

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("⚙️ Controls")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["A*", "BFS", "DFS"]
)

st.sidebar.markdown("""
**Step 1:** Click Start point on map  
**Step 2:** Click Goal point on map  
**Step 3:** Click **Find Path**
""")

# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------
if "start" not in st.session_state:
    st.session_state.start = None
if "goal" not in st.session_state:
    st.session_state.goal = None
if "path" not in st.session_state:
    st.session_state.path = None

# ---------------------------------------------------------
# Map
# ---------------------------------------------------------
m = folium.Map(location=[24.8607, 67.0011], zoom_start=12)

if st.session_state.start:
    folium.Marker(
        st.session_state.start,
        tooltip="Start",
        icon=folium.Icon(color="green")
    ).add_to(m)

if st.session_state.goal:
    folium.Marker(
        st.session_state.goal,
        tooltip="Goal",
        icon=folium.Icon(color="red")
    ).add_to(m)

if st.session_state.path:
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in st.session_state.path]
    folium.PolyLine(coords, color="blue", weight=5).add_to(m)

map_data = st_folium(m, height=520, width=900)

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    if st.session_state.start is None:
        st.session_state.start = (lat, lon)
        st.success("✅ Start selected")

    elif st.session_state.goal is None:
        st.session_state.goal = (lat, lon)
        st.success("✅ Goal selected")

# ---------------------------------------------------------
# Find Path
# ---------------------------------------------------------
if st.button("🚀 Find Path"):
    if not st.session_state.start or not st.session_state.goal:
        st.error("❌ Please select both start and goal points.")
    else:
        start_node = ox.distance.nearest_nodes(
            G,
            X=st.session_state.start[1],
            Y=st.session_state.start[0]
        )
        goal_node = ox.distance.nearest_nodes(
            G,
            X=st.session_state.goal[1],
            Y=st.session_state.goal[0]
        )

        with st.spinner("Calculating path..."):
            if algorithm == "BFS":
                st.session_state.path = bfs(G, start_node, goal_node)
            elif algorithm == "DFS":
                st.session_state.path = dfs(G, start_node, goal_node)
            else:
                st.session_state.path = astar(G, start_node, goal_node)

        if st.session_state.path:
            st.success("✅ Path found successfully!")
        else:
            st.error("❌ No path found.")

# ---------------------------------------------------------
# Distance & Time Calculation
# ---------------------------------------------------------
if st.session_state.path:
    total_distance_m = 0

    for u, v in zip(st.session_state.path[:-1], st.session_state.path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            edge = list(edge_data.values())[0]
            total_distance_m += edge.get("length", 0)

    distance_km = total_distance_m / 1000
    avg_speed_kmh = 40  # Karachi average traffic speed
    time_minutes = (distance_km / avg_speed_kmh) * 60

    st.subheader("📍 Route Information")
    col1, col2 = st.columns(2)
    col1.metric("🛣️ Distance (km)", f"{distance_km:.2f}")
    col2.metric("⏱️ Estimated Time (minutes)", f"{time_minutes:.1f}")

# ---------------------------------------------------------
# Algorithm Comparison Table
# ---------------------------------------------------------
st.subheader("📊 Algorithm Comparison")

comparison_data = {
    "Algorithm": ["BFS", "DFS", "A*"],
    "Completeness": ["Yes", "Yes", "Yes"],
    "Optimal Path": ["Yes (Unweighted)", "No", "Yes"],
    "Time Complexity": ["O(V + E)", "O(V + E)", "O(E)"],
    "Space Complexity": ["O(V)", "O(V)", "O(V)"],
    "Best Use Case": [
        "Shortest path (unweighted)",
        "Fast exploration",
        "Shortest path on maps"
    ]
}

st.table(comparison_data)

# ---------------------------------------------------------
# Reset
# ---------------------------------------------------------
if st.sidebar.button("🔄 Reset"):
    st.session_state.start = None
    st.session_state.goal = None
    st.session_state.path = None
    st.rerun()
