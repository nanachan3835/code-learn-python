import networkx as nx
import os
import random
import uuid


#def FatTree4(k, node_cap, link_cap):
#   pass


def FromGml(path: str, node_cap:list, link_cap:list) -> nx.DiGraph:
    topo_name = os.path.basename(path)
    graph = nx.read_gml(path)
    graph = nx.DiGraph(graph)
    nodes = list(graph.nodes)
    links = [(nodes.index(e[0]), nodes.index(e[1])) for e in list(graph.edges)]
    
    PHY_nodes = [(nodes.index(node), {"weight": random.randrange(node_cap[0], node_cap[1], node_cap[2])}) for node in nodes]
    PHY_links = [(link[0], link[1], {"weight": random.randrange(link_cap[0], link_cap[1], link_cap[2])}) for link in links]
    
    PHY = nx.DiGraph()
    PHY.name = f"{topo_name}@{uuid.uuid4()}"
    #thông số của mạng vật lý 
    PHY.attr = {
        "topology": topo_name, #tên của file gml
        "node_count": len(nodes),#số lượng nodes
        "link_count": len(links),#số lượng kết nối
        "node_cap_min": node_cap[0],#
        "node_cap_max": node_cap[1],# 
        "node_cap_step": node_cap[2],#
        "link_cap_min": link_cap[0],
        "link_cap_max": link_cap[1],
        "link_cap_step": link_cap[2]
    }
    PHY.add_nodes_from(PHY_nodes)
    PHY.add_edges_from(PHY_links)
    return PHY

#input_node_cap=[100, 200, 50] #list
#input_link_cap=[200, 300, 5] #list
#input_path=path file phy.gml #path
def ShowPHYGml(path: str, node_cap:list, link_cap:list):
    #path=input_path
    #node_cap=input_node_cap
    #link_cap=input_link_cap
    PHY = FromGml(path,node_cap,link_cap)
    for key,value in nx.get_edge_attributes(PHY, "weight").items():
        print(str(key) + ":" + str(value))
    for key,value in PHY.attr.items():
        print(str(key) + ":" + str(value))
    nx.draw_kamada_kawai(PHY, with_labels=True)