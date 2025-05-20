import resco_benchmark.rewards as rewards
import resco_benchmark.states as states

from resco_benchmark.agents.maxpressure import MAXPRESSURE
from resco_benchmark.agents.pfrl_dqn import IDQN
from resco_benchmark.agents.pfrl_ppo import IPPO
from resco_benchmark.agents.mplight import MPLight

agent_configs = {
    'MAXPRESSURE': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 200
    },
    'IDQN': {
        'agent': IDQN,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500
    },
    'IPPO': {
        'agent': IPPO,
        'state': states.drq_norm,
        'reward': rewards.wait_norm,
        'max_distance': 200
    },
    'MPLight': {
        'agent': MPLight,
        'state': states.mplight,
        'reward': rewards.pressure,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500,
        'demand_shape': 1
    }
}
