import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox

from map_loader import load_karachi_graph
from algorithms import bfs, dfs, astar

st.set_page_config(layout="wide")
st.title("🚦 Karachi Pathfinding using OpenStreetMap")

@st.cache_resource
def load_graph():
    return load_karachi_graph()

graph = load_graph()

# --- Algorithm Selector ---
algo = st.sidebar.selectbox("Select Algorithm", ["BFS", "DFS", "A*"])

# --- Map ---
m = folium.Map(location=[24.8607, 67.0011], zoom_start=12)

st.markdown("### Click to select **Start** then **Goal**")

map_data = st_folium(m, height=500, width=900)

# --- Capture Clicks ---
if map_data and map_data["last_clicked"]:
    if "start" not in st.session_state:
        st.session_state.start = map_data["last_clicked"]
        st.success("Start point selected")
    elif "goal" not in st.session_state:
        st.session_state.goal = map_data["last_clicked"]
        st.success("Goal point selected")

# --- Run Pathfinding ---
if st.button("Find Path"):
    if "start" in st.session_state and "goal" in st.session_state:

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
            m = folium.Map(location=[24.8607, 67.0011], zoom_start=12)

            coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in path]
            folium.PolyLine(coords, color="blue", weight=5).add_to(m)

            folium.Marker(coords[0], icon=folium.Icon(color="green"), tooltip="Start").add_to(m)
            folium.Marker(coords[-1], icon=folium.Icon(color="red"), tooltip="Goal").add_to(m)

            st_folium(m, height=500, width=900)
            st.success("Path found successfully!")

        else:
            st.error("No path found")
    else:
        st.warning("Please select start and goal points")
