import osmnx as ox
import os
import streamlit as st

# File path where the graph will be saved
GRAPH_FILENAME = "karachi_drive.graphml"

@st.cache_resource(show_spinner="Loading Karachi road network...")
def load_karachi_graph():
    # Check if the file already exists locally
    if os.path.exists(GRAPH_FILENAME):
        print(f"Loading {GRAPH_FILENAME} from disk...")
        # Load from local file (OSMnx 2.0+ standard)
        G = ox.io.load_graphml(GRAPH_FILENAME)
    else:
        print("Downloading Karachi road network from OSM...")
        # Download from OpenStreetMap
        G = ox.graph_from_place(
            "Karachi, Pakistan",
            network_type="drive",
            simplify=True
        )
        
        # Modern conversion to undirected
        G = ox.convert.to_undirected(G)
        
        # Save to local disk for next time
        ox.io.save_graphml(G, filepath=GRAPH_FILENAME)
        print(f"Graph saved to {GRAPH_FILENAME}")

    return G