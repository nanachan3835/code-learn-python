from .environment import * 

from gymnasium.envs.registration import register

register(
    id="graph_mapping/static_mapping-v0",
    entry_point="environment_and_agent.environment:StaticMappingEnv",
)

