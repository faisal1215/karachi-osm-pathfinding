# =========================================================
# Karachi OSM Pathfinding App
# BFS | DFS | A* using OpenStreetMap
# =========================================================

import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import math  # Added for the heuristic calculation
from map_loader import load_karachi_graph

# This will now be super fast after the first time!
G = load_karachi_graph()

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
    # Loading the graph. Note: This graph is unprojected (Lat/Lon)
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
    """
    A* Algorithm with Haversine Heuristic.
    Calculates great-circle distance in meters between nodes.
    """
    def heuristic(u, v):
        # 1. Get coordinates
        x1, y1 = graph.nodes[u]['x'], graph.nodes[u]['y']
        x2, y2 = graph.nodes[v]['x'], graph.nodes[v]['y']
        
        # 2. Haversine Formula (Lat/Lon to Meters)
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [x1, y1, x2, y2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in meters
        r = 6371000 
        return c * r

    # 3. Run A*
    return nx.astar_path(
        graph,
        start,
        goal,
        heuristic=heuristic,
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

# Draw Start Marker
if st.session_state.start:
    folium.Marker(
        st.session_state.start,
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

# Draw Goal Marker
if st.session_state.goal:
    folium.Marker(
        st.session_state.goal,
        tooltip="Goal",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

# Draw Path Polyline
if st.session_state.path:
    # Extract coordinates for the path
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in st.session_state.path]
    folium.PolyLine(coords, color="blue", weight=5, opacity=0.7).add_to(m)

# Render Map
map_data = st_folium(m, height=520, width=900)

# Handle Map Clicks
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    if st.session_state.start is None:
        st.session_state.start = (lat, lon)
        st.success(f"✅ Start selected: {lat:.4f}, {lon:.4f}")
        st.rerun() # Immediate rerun to update marker

    elif st.session_state.goal is None:
        st.session_state.goal = (lat, lon)
        st.success(f"✅ Goal selected: {lat:.4f}, {lon:.4f}")
        st.rerun() # Immediate rerun to update marker

# ---------------------------------------------------------
# Find Path
# ---------------------------------------------------------
if st.sidebar.button("🚀 Find Path"): # Moved button to sidebar for cleaner UI
    if not st.session_state.start or not st.session_state.goal:
        st.error("❌ Please select both start and goal points.")
    else:
        # Find nearest network nodes
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

        with st.spinner(f"Running {algorithm}..."):
            try:
                if algorithm == "BFS":
                    st.session_state.path = bfs(G, start_node, goal_node)
                elif algorithm == "DFS":
                    st.session_state.path = dfs(G, start_node, goal_node)
                else:
                    st.session_state.path = astar(G, start_node, goal_node)

                if st.session_state.path:
                    st.success("✅ Path found successfully!")
                    st.rerun()
                else:
                    st.error("❌ No path found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ---------------------------------------------------------
# Distance & Time Calculation
# ---------------------------------------------------------
if st.session_state.path:
    total_distance_m = 0

    for u, v in zip(st.session_state.path[:-1], st.session_state.path[1:]):
        # Handle multiple edges between nodes (multigraph)
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            # Pick the shortest edge if multiple exist (key=0 is usually sufficient)
            edge = min(edge_data.values(), key=lambda x: x.get("length", 0))
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