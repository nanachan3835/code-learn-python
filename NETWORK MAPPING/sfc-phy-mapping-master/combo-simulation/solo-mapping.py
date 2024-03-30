# %% [markdown]
# # SFC-PHY Mapping Problem Explained | Python Version
# SFC Graph and PHY Graph implementation and visualization with NetworkX.
# Solving problem with PuLP.

# %% [markdown]
# ## Prepare
# Import required libraries.

# %%
# Variables
input = "./"
output = "./"

# %%
# Install
# %pip install networkx
# %pip install pulp

# Imports
import networkx as nx
import pulp as pulp
from pulp import *
import matplotlib.pyplot as plt
import fnmatch
import random

problemName = "mapping-solo"

print(problemName)

problem = LpProblem(problemName, LpMinimize)

input = os.path.join(input, "test-cases")
output = os.path.join(output, "solo-mapping")

if not os.path.exists(input):
    raise Exception("No input!")

try:
    if os.path.exists(output):
        for f in os.listdir(output):
            os.remove(f"{output}/{f}")
    else:
        os.mkdir(output)
except OSError as error:
    print(error)


# %% [markdown]
# ## Input Graph and Visualization
# Graph implementation
# - SFC: directed graph
# - PHY: undirected graph

# %%
# PHY
G = nx.read_gml(f"{input}/phy.gml")


# %%
# SFC
files = fnmatch.filter(os.listdir(input), 'sfc_*.gml')
file = random.choice(files)
GS = nx.read_gml(f"{input}/{file}")

# %% [markdown]
# ## Variables initialization
# Fire up our variables!!!
# $$ \phi_{i}^{v} = \begin{cases}
# 1 & \text{when } v \text{ is placed at }i\\
# 0 & \text{otherwise}
# \end{cases} $$
# 
# $$ \phi_{ij}^{vw}=\begin{cases}
# 1 & \text{when }vw\text{ is placed at }ij\\
# 0 & \text{otherwise}
# \end{cases} $$

# %%
# Build Node Placement List
phiNode = LpVariable.dicts(
    name="phiNode", 
    indices=(GS.nodes, G.nodes), 
    cat="Binary"
)
phiNode

# %%
# Build Link Placement List
phiLink = LpVariable.dicts(
    name="phiLink",
    indices=(GS.edges, G.edges),
    cat="Binary"
)
phiLink

# %% [markdown]
# ## Building Constraints
# Build 5 constraints groups.

# %%
# clear any left-over data
problem.constraints.clear()


# %% [markdown]
# ### Constraints Group 1: Node Capacity 
# The total requested resources on a single node is under the maximum capacity of that node.
# $$ \sum_{v\in\mathcal{N}_{s}}\phi_{i}^{v}.r_{v}\leq a_{i}\qquad\forall i\in\mathcal{N} $$

# %%
for node in G.nodes:
    problem += (
        lpSum(
            phiNode[nodeS][node] * nx.get_node_attributes(GS, "Req")[nodeS] for nodeS in GS.nodes) 
            <= nx.get_node_attributes(G, "Cap")[node]
            ,
            f"C1_i{node}"
        )


# %% [markdown]
# ### Constraints Group 2: Link Capacity
# Total requested resouces on a single link is under the maximun capacity of that link
# $$ \sum_{vw\in\mathcal{E}_{s}}\phi_{ij}^{vw}.r_{vw}\leq a_{ij}\qquad\forall ij\in\mathcal{E} $$

# %%
for edge in G.edges:
    problem += (
        lpSum(
            phiLink[edgeS][edge] * nx.get_edge_attributes(GS, "Req")[edgeS]
            for edgeS in GS.edges
        ) 
        <= nx.get_edge_attributes(G, "Cap")[edge]
        , 
        f"C2_ij{edge}"
    )


# %% [markdown]
# ### Constraints Group 3: Single VNF in Node
# A node can only contain one VNF of a SFC.
# $$ \sum_{v\in\mathcal{N}_{s}}\phi_{i}^{v}\leq1\qquad\forall i\in\mathcal{N} $$

# %%
for node in G.nodes:
    problem += (
        lpSum(
            phiNode[nodeS][node] for nodeS in GS.nodes) 
            <= 1
            , 
            f"C3_i{node}"
    )

# %% [markdown]
# ### Constraints Group 4: No VNF is left behind
# Every VNF is placed in a Node
# $$ \sum_{i\in\mathcal{N}}\phi_{i}^{v}=1\qquad\forall v\in\mathcal{N}_{s} $$

# %%
for nodeS in GS.nodes:
    problem += (
        lpSum(
            phiNode[nodeS][node] for node in G.nodes
        )
        == 1
        , 
        f"C4_v{nodeS}"
    )

# %% [markdown]
# ### Constraints Group 5: Flow Conservation
# The Flow must be preserved
# $$ \sum_{j\in\mathcal{N}}\phi_{ij}^{vw}-\sum_{j\in\mathcal{N}}\phi_{ji}^{vw}=\phi_{i}^{v}-\phi_{i}^{w}\qquad\forall i\in\mathcal{N},\forall vw\in\mathcal{E}_{s} $$

# %%
for i in G.nodes:
    for vw in GS.edges:
        problem += (
            (
                lpSum(phiLink[vw].get((i, j)) for j in G.nodes) 
                - lpSum(phiLink[vw].get((j, i)) for j in G.nodes)
            ) 
            == phiNode[vw[0]][i] - phiNode[vw[1]][i]
            , 
            f"C5_i{i}_vw{vw}"
        )


# %% [markdown]
# ### Optional: Force put a VNF at a fixed Node
# We can forcefully place a VNF in a node of PHY network.
# $$ \phi_{i}^{v} = 1 $$

# %%
# Force starts at Node 4
# problem += phiNode[1][4] == 1
# Force ends at Node 7
# problem += phiNode[5][7] == 1

# %% [markdown]
# ## Building objective function: Use minimum edges and nodes
# Minimize the number of needed edges and nodes
# $$ \min\left(\phi_{\mathcal{G}}^{\mathcal{G}_{s}}\right)=\min\left(\sum_{i\in\mathcal{N}}\sum_{v\in\mathcal{N}_{s}}\phi_{i}^{v}+\sum_{ij\in\mathcal{E}}\sum_{vw\in\mathcal{E}_{s}}\phi_{ij}^{vw}\right) $$

# %%
problem += (
    lpSum(
        lpSum(
            phiNode[v][i] for v in GS.nodes
        ) 
        for i in G.nodes
    ) 
    + lpSum(
        lpSum(
            phiLink[vw][ij]
            for vw in GS.edges
        ) 
        for ij in G.edges
    )
)

# %% [markdown]
# ## Confirm before start
# Double-check the whole inputs. (it's in the generated file)

# %%
problem.writeLP(filename=f"{output}/{problemName}.lp")

# %% [markdown]
# ## Let's the show begins
# Fire the solver and wait for results. `Optimal` means that we found the solution. `Infeasible` means that we cannot solve the problem.

# %%
solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True) # GNNN/RL/ML
status = problem.solve(solver)
LpStatus[status]
print(LpStatus[status])
print(problem.solutionTime)
if (LpStatus[status] == "Infeasible"):
    raise Exception("Infeasible")

# %%
f=open(f"{output}/sfc_{files.index(file)}.txt", "w+")
print(f"sfc_{files.index(file)}")
for v in GS.nodes:
    for i in G.nodes:
        if (value(phiNode[v][i])):
            print(f"v{v} -> i{i}")
            f.write(f"v{v} -> i{i}\n")

selectedLinks = list()
for vw in GS.edges:
    for ij in G.edges:
        if (value(phiLink[vw][ij])):
            print(f"vw{vw} -> ij{ij}")
            f.write(f"vw{vw} -> ij{ij}\n")
f.close()



