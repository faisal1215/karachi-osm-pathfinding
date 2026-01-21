import osmnx as ox

def load_karachi_graph():
    print("Downloading Karachi road network...")
    
    G = ox.graph_from_place(
        "Karachi, Pakistan",
        network_type="drive",
        simplify=True
    )

    # ✅ NEW WAY (works with latest OSMnx)
    G = ox.convert.to_undirected(G)

    return G
