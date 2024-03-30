# %% [markdown]
# # SFC-PHY Mapping Problem | Python Edition
# 
# **== TRADE OFFER ==**
# 
# I receive: *a bunch of SFCs' Graph and one PHY Graph*
# 
# You receive: *__optimal mapping result__*

# %% [markdown]
# ## Preparation
# Install libraries, import libraries, initialize this and that, ... Such boring lines of codes...

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

problemName = "mapping-combone-simple"

print(problemName)

problem = LpProblem(problemName, LpMinimize)

input = os.path.join(input, "test-cases")
output = os.path.join(output, "simple-combo-mapping")

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


# %%
# PHY
G = nx.read_gml(f"{input}/phy.gml")

# SFCs
SFCs = list()
files = fnmatch.filter(os.listdir(input), 'sfc_*.gml')
for f in files:
    SFCs.append(nx.read_gml(f"{input}/{f}"))

# %%
# Build Node Placement List
phiNode_S = list()
for sfc in SFCs:
    phiNode_S.append(
        LpVariable.dicts(
            name=f"phiNode_s{SFCs.index(sfc)}",
            indices=(sfc.nodes, G.nodes),
            cat = "Binary"
        )
    )

# Build Link Placement List
phiLink_S = list()
for sfc in SFCs:
    phiLink_S.append(
        LpVariable.dicts(
            name=f"phiLink_s{SFCs.index(sfc)}",
            indices=(sfc.edges, G.edges),
            cat="Binary"
        )
    )

# %% [markdown]
# ## Constraints list
# Attention: there are FIVE GROUPS of constraints.
# $$ \sum_{s\in S}\sum_{v\in N_{s}}\phi_{i}^{v,s}.r_{v}\leq a_{i}\qquad\forall i\in\mathcal{N} \qquad \text{(C1)} $$
# $$ \sum_{s\in S}\sum_{vw\in E_{s}}\phi_{ij}^{vw,s}.r_{vw}\leq a_{ij}\qquad\forall ij\in\mathcal{E} \qquad \text{(C2)} $$
# $$ \sum_{v\in\mathcal{N}_{s}}\phi_{i}^{v,s}\leq1\qquad\forall i\in\mathcal{N},\forall s\in\mathcal{S} \qquad \text{(C3)} $$
# $$ \sum_{i\in\mathcal{N}}\phi_{i}^{v,s}=1\qquad\forall v\in\mathcal{N}_{s},\forall s\in\mathcal{S} \qquad \text{(C4)} $$
# $$ \sum_{j\in N}\phi_{ij}^{vw,s}-\sum_{j\in N}\phi_{ji}^{vw,s}=\phi_{i}^{v,s}-\phi_{i}^{w,s}\qquad\forall i\in\mathcal{N},\forall vw\in\mathcal{E}_{s},\forall s\in\mathcal{S} \qquad \text{(C5)} $$

# %%
# Bulding Constraints
problem.constraints.clear()

## C1
for node in G.nodes:
    problem += (
        lpSum(
            lpSum(
                phiNode_S[SFCs.index(sfc)][node_S][node] * nx.get_node_attributes(sfc, "Req")[node_S]
                    for node_S in sfc.nodes
            ) 
            for sfc in SFCs
        )
            <= nx.get_node_attributes(G, "Cap"), 
            f"C1_i{node}"
    )

## C2
for edge in G.edges:
    problem += (
        lpSum(
            lpSum(
                phiLink_S[SFCs.index(sfc)][link_S][edge] * nx.get_edge_attributes(sfc, "Req")[link_S]
                for link_S in sfc.edges
            ) 
            for sfc in SFCs
        )
            <= nx.get_edge_attributes(G, "cap"), 
        f"C2_ij{edge}"
    )

## C3
for sfc in SFCs:
    for node in G.nodes:
        problem += (
            lpSum(
                phiNode_S[SFCs.index(sfc)][node_S][node]
                for node_S in sfc.nodes
            )
            <= 1
            ,
            f"C3_i{node}_s{SFCs.index(sfc)}"
        )

## C4
for sfc in SFCs:
    for node_S in sfc.nodes:
        problem += (
            lpSum(
                phiNode_S[SFCs.index(sfc)][node_S][node]
                for node in G.nodes
            )
            == 1
            ,
            f"C4_v{node_S}_s{SFCs.index(sfc)}"
        )

## C5
for sfc in SFCs:
    for edge_S in sfc.edges:
        for node in G.nodes:
            problem += (
                lpSum(
                    phiLink_S[SFCs.index(sfc)][edge_S].get((node, nodej))
                    for nodej in G.nodes
                ) 
                - 
                lpSum(
                    phiLink_S[SFCs.index(sfc)][edge_S].get((nodej, node))
                    for nodej in G.nodes
                )
                == phiNode_S[SFCs.index(sfc)][edge_S[0]][node] - phiNode_S[SFCs.index(sfc)][edge_S[1]][node]
                ,
                f"C5_s{SFCs.index(sfc)}_vw{edge_S}_i{node}"
            )

# %% [markdown]
# ## Building target function
# 
# The simpiest function ever: minium physical units !!!
# 
# $$ \min\left(\phi_{\mathcal{G}}^{\mathcal{S}}\right)=\min\left(\sum_{s\in\mathcal{S}}\sum_{i\in\mathcal{N}}\sum_{v\in\mathcal{N}_{s}}\phi_{i}^{v,s}+\sum_{s\in\mathcal{S}}\sum_{ij\in\mathcal{E}}\sum_{vw\in\mathcal{E}_{s}}\phi_{ij}^{vw,s}\right) $$

# %%
# Target function building
problem += (
    lpSum(
        phiNode_S[SFCs.index(sfc)][node_S][node]
        for sfc in SFCs
        for node_S in sfc.nodes
        for node in G.nodes
    )
    +
    lpSum(
        phiLink_S[SFCs.index(sfc)][link_S][link]
        for sfc in SFCs
        for link_S in sfc.edges
        for link in G.edges
    )
)

# %% [markdown]
# ## Confirm and execute
# 
# Cornfirm everything is right and execute the solver

# %%
# Write LP
problem.writeLP(f"{output}/{problemName}.lp")

# Initialize Solver
solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)

# %%
# Execute the solver
status = problem.solve(solver)

# %%
# Read the status
print(LpStatus[status])
print(problem.solutionTime)
if (LpStatus[status] == "Infeasible"):
    raise Exception("Infeasible")

# %% [markdown]
# ## Print out the results
# 
# Results is here.

# %%
# Print out the result
for sfc in SFCs:
    print(f"sfc_{SFCs.index(sfc)}")
    f = open(f"{output}/sfc_{SFCs.index(sfc)}.txt", "w+")
    for node_S in sfc.nodes:
        for node in G.nodes:
            if (value(phiNode_S[SFCs.index(sfc)][node_S][node])):
                # print(f"s{SFCs.index(sfc)}_v{node_S} -> i{node}")
                f.writelines([
                    f"s{SFCs.index(sfc)}_v{node_S} -> i{node}\n"
                ])
    for link_S in sfc.edges:
        for link in G.edges:
            if (value(phiLink_S[SFCs.index(sfc)][link_S][link])):
                # print(f"s{SFCs.index(sfc)}_vw{link_S} -> ij{link}")
                f.writelines([
                    f"s{SFCs.index(sfc)}_vw{link_S} -> ij{link}\n"
                ])
    f.close()

