import numpy as np
import torch
import utils
from environment import Environment

# vanilla
class ReplayBuffer:
    def __init__(self, m_size=50000, batch_size=64, statesize=None, actionsize=None, rewardsize=None, replace=False) -> None:
        if statesize is None:               ssz = (m_size,)
        elif isinstance(statesize, int):    ssz = (m_size, statesize)
        elif isinstance(statesize, tuple):  ssz = (m_size, *statesize)
        
        if actionsize is None:               asz = (m_size,)
        elif isinstance(actionsize, int):    asz = (m_size, actionsize)
        elif isinstance(actionsize, tuple):  asz = (m_size, *actionsize)
        
        if rewardsize is None:               rsz = (m_size,)
        elif isinstance(rewardsize, int):    rsz = (m_size, rewardsize)
        elif isinstance(rewardsize, tuple):  rsz = (m_size, *rewardsize)

        self.states  = torch.empty(ssz)
        self.actions = torch.empty(asz, dtype=torch.int64)
        self.rewards = torch.empty(rsz)
        self.next_states = torch.empty(ssz)
        self.terminateds = torch.empty(m_size, dtype=torch.bool)

        self.m_size = m_size
        self.size = 0; self._i = 0
        self.batch_size = batch_size
        self.replace = replace # are you allowed to replay the same sample in the batch?

    def store(self, sample):
        s, a, r, ns, t = sample
        self.states[self._i] = s
        self.actions[self._i] = a
        self.rewards[self._i] = r
        self.next_states[self._i] = ns
        self.terminateds[self._i] = t
        self._i = (1 + self._i)%self.m_size
        self.size = min(self.size+1, self.m_size)
    
    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        ixs = np.random.choice(self.size, batch_size, replace=self.replace)
        res = torch.vstack(list(self.states[ixs])), \
              torch.vstack(list(self.actions[ixs])), \
              torch.vstack(list(self.rewards[ixs])), \
              torch.vstack(list(self.next_states[ixs])), \
              torch.vstack(list(self.terminateds[ixs]))
        return res
    
    def __len__(self):
        return self.size
    
    def __repr__(self) -> str:
        return '\n'.join(str(elem) for elem in zip(self.states, self.actions, self.rewards, self.next_states, self.terminateds))

# PER
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, m_size=50000, batch_size=64, statesize=None, actionsize=None, rewardsize=None, rank_based=False, replace=False, alpha=0.6, beta0=0.1, beta_rate=0.9992) -> None:
        super().__init__(m_size, batch_size, statesize, actionsize, rewardsize, replace)

        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate
        self.eps = 1e-6
        self.td_errors = torch.empty(m_size, dtype=torch.float32) # stores ABS of td errors aka priorities

    def update(self, idxs: torch.Tensor, td_errors: torch.Tensor):
        self.td_errors[idxs] = torch.abs(td_errors)
        if self.rank_based: pass ## todo ##

    def store(self, sample):
        self.td_errors[self._i] = self.td_errors[:self.size].max().item() if self.size > 0 else 1.0 # give it highest priority of all
        super().store(sample) # store the other parts of the sample

    def _update_beta(self):
        self.beta = min(1.0, self.beta / self.beta_rate); return self.beta
    
    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self._update_beta()

        # the uniform case: ixs = np.random.choice(self.size, batch_size, replace=replace)
        
        # - is updated to -
        
        priorities = (self.td_errors[:self.size] + self.eps)**self.alpha
        probs = np.array(priorities, dtype=np.float32); probs /= probs.sum()
        weights = torch.tensor((self.size * probs)**(-self.beta))
        weights /= weights.max()
        
        # - this
        
        ixs = np.random.choice(self.size, batch_size, replace=self.replace, p=probs)

        res = torch.vstack(list(self.states[ixs])), \
              torch.vstack(list(self.actions[ixs])), \
              torch.vstack(list(self.rewards[ixs])), \
              torch.vstack(list(self.next_states[ixs])), \
              torch.vstack(list(self.terminateds[ixs]))
        return torch.tensor(ixs), weights[ixs], res

    def __repr__(self) -> str:
        return '\n'.join(str(elem) for elem in zip(self.states, self.actions, self.rewards, self.next_states, self.terminateds, self.td_errors))
    
BUF_TEST=True
if __name__ == '__main__' and BUF_TEST:
    t = torch.tensor
    buf = PrioritizedReplayBuffer(5, 2, alpha=1)
    buf.store((1, 2, 1, 2, 1))
    # print(buf, end='\n\n')
    buf.update([0], t(3.))
    # print(buf, end='\n\n')
    buf.store((1, 3, 1, 3, 1))
    buf.update([0, 1], t([2., 0.]))
    # print(buf, end='\n\n')
    # print(buf.sample(50, replace=True)[0], sep='\n')
    # print(buf, end='\n\n')

# small module for handling all computation required during a trajectory. Run internally by each agent
class ExperienceProcessor:
    def __init__(self,
                 state_dim,
                 gamma,
                 tau,
                 max_steps,
                 max_steps_per_episode,
                 device,
                 num_envs=16):
        print('procargs', state_dim, gamma, tau, max_steps_per_episode, max_steps, device, num_envs, flush=True)
        assert max_steps >= max_steps_per_episode

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.max_steps = max_steps
        self.max_steps_per_episode = max_steps_per_episode # used only for logging
        self.num_envs = num_envs

        self.device = torch.device(device)

        self.states_mem = np.empty(
            shape=np.concatenate(((self.max_steps, self.num_envs), self.state_dim)), dtype=np.float32)
        self.states_mem[:] = np.nan

        self.actions_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.int32)
        self.actions_mem[:] = -1

        self.rewards_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.rewards_mem[:] = np.nan

        self.returns_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.returns_mem[:] = np.nan

        self.gaes_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.gaes_mem[:] = np.nan

        self.logpas_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.logpas_mem[:] = np.nan

    def fill(self, env, policy_model, value_model):
        
        assert self.num_envs == env.num_envs

        # just logging information. Not used in the optimization
        ### CURRENTLY NOT COMPUTED.
        ep_terminated = [] # information per episode.
        ep_returns = []
        ep_dists = []
        ep_fidels = []
        ep_acts = []

        self.total_steps = 0
        state, info = env.reset() # note that this initial state will go into the state_mem on the first iteration
        
        gates, _, targets, _, _ = env.prepare_gatelist(env.qubits)
        targets2 = [target if len(target) > 1 else target + target for target in targets]
        targets2 = np.array(targets2, dtype=np.int32)
        penalty = np.ones((len(gates),), dtype=np.float32) * 0.5 / env.max_steps
        for i, gt in enumerate(gates):
            if gt.startswith('sdg') or gt.startswith('hsdgh'):
                penalty[i] *= 1
            else:
                penalty[i] *= 3

        dones = np.zeros((env.num_envs,), dtype=bool)
        rolling_returns = np.zeros((env.num_envs,), dtype=np.float32)
        step_number = np.zeros((env.num_envs,), dtype=np.int32)

        common_shape = tuple(self.actions_mem.shape)
        values = np.zeros(shape=common_shape)
        terminals = np.zeros(shape=common_shape)
        truncateds = np.zeros(shape=common_shape)
        
        prev_step_actions = np.zeros((env.num_envs, env.qubits, env.qubits), dtype=np.int32) - 1
        env_range = np.arange(env.num_envs)

        for timestep in range(self.max_steps):
            # if timestep %10 == 0:print(timestep, flush=True)
            
            # take a step (nowhere here do we need gradients, so torch.no_grad)
            with torch.no_grad():
                action, logpa = policy_model.np_pass(state)
                values[timestep] = value_model.forward(state).cpu().numpy()

            next_state, reward, terminal, truncated, info = env.step(action)
            # print('next_state', next_state, flush=True)
            # print('reward-agent', reward, flush=True)
            step_number += 1

            # # discourage the agent from taking the same action again
            # qubits_of_action = targets2[action]
            # # assert qubits_of_action.shape == (env.num_envs, 2), f'{qubits_of_action} {action}'
            # reward -= (prev_step_actions[env_range, qubits_of_action[:, 0], qubits_of_action[:, 1]] == action) * penalty[action]
            # prev_step_actions[env_range, qubits_of_action[:, 0], qubits_of_action[:, 1]] = action
            
            rolling_returns += self.gamma ** (step_number-1) * reward

            terminals[timestep] = terminal
            truncateds[timestep] = truncated
            done = terminal|truncated
            
            ep_returns.extend(rolling_returns[done])
            ep_fidels.extend(env.fidelity_of_resetted)
            ep_dists.extend(env.meta_actions_of_resetted)
            # ep_acts.extend([[2]*steps_taken for steps_taken in step_number[done]])
            ep_acts.extend(step_number[done])
            assert len(ep_acts) == len(ep_returns) == len(ep_dists) == len(ep_fidels)
            rolling_returns[done] = 0
            step_number[done] = 0

            self.states_mem[timestep] = state
            self.actions_mem[timestep]= action
            self.logpas_mem[timestep] = logpa
            self.rewards_mem[timestep]= reward

            state = next_state
            
            # new_dists, new_fidels = env.stats() # collect some stats too

        # edge bootstrap
        with torch.no_grad():
            next_value = value_model(state).reshape(1, -1).cpu().numpy()
        assert values.shape[-1] == next_value.shape[-1]
        # values = torch.cat((values, next_value), dim=0)
        # assert values.shape[-1] == next_value.shape[-1]

        # now we have a lot of experience. Compute the returns and gaes
        advantages = np.zeros(shape=self.rewards_mem.shape)
        # advantages[-1] = next_value

        assert timestep == self.max_steps - 1
        for t in range(timestep, -1, -1):
            delta = self.rewards_mem[t] + self.gamma * (values[t+1] if t != timestep else next_value) * (1 - terminals[t]) - values[t]
            delta *= (1 - truncateds[t])
            advantages[t] = delta + self.gamma * self.tau * (1 - terminals[t]) * (1 - truncateds[t]) * (advantages[t+1] if t != timestep else 0)

        # print('advantages', advantages, flush=True)
        self.returns_mem = values + advantages
        self.gaes_mem = advantages

        ep_idxs_int = np.arange(len(ep_fidels))#np.arange(self.max_episodes)[ep_idxs]
        print('filled', len(ep_fidels), 'episodes total')
        return {'idxs':ep_idxs_int, 'ter':ep_terminated, 'n_steps':ep_acts, 'returns':ep_returns, 'meta':ep_dists, 'fidels':ep_fidels}

    def get_stacks(self) -> tuple[np.array, np.array, np.array, np.array, np.array]:
        return (self.states_mem, self.actions_mem, 
                self.returns_mem, self.gaes_mem, self.logpas_mem)
    
    def update_max_episodes(self):
        # self.max_steps = int(self.max_steps * 1.001)
        # self.clear()
        pass

    # def __len__(self):
    #     return self.total_steps

# buf for trajectories
# class EpisodeBuffer:
#     def __init__(self,
#                  state_dim,
#                  gamma,
#                  tau,
#                  n_workers,
#                  max_steps,
#                  max_steps_per_episode):
        
#         assert max_steps >= n_workers * max_steps_per_episode

#         self.state_dim = state_dim
#         self.gamma = gamma
#         self.tau = tau
#         self.n_workers = n_workers
#         self.max_steps = max_steps
#         self.max_steps_per_episode = max_steps_per_episode

#         self.discounts = np.logspace(0, max_steps_per_episode, num=max_steps_per_episode+1, base=gamma, endpoint=False, dtype=np.float32)
#         self.tau_discounts = np.logspace(0, max_steps_per_episode, num=max_steps_per_episode+1, base=gamma*tau, endpoint=False, dtype=np.float32)

#         device = utils._globals['device']
#         self.device = torch.device(device)

#         self.states_mem = np.empty(
#             shape=np.concatenate(((self.max_steps,), self.state_dim)), dtype=np.float32)
#         self.states_mem[:] = np.nan

#         self.actions_mem = np.empty(shape=(self.max_steps,), dtype=np.int32)
#         self.actions_mem[:] = -1

#         self.returns_mem = np.empty(shape=(self.max_steps,), dtype=np.float32)
#         self.returns_mem[:] = np.nan

#         self.gaes_mem = np.empty(shape=(self.max_steps,), dtype=np.float32)
#         self.gaes_mem[:] = np.nan

#         self.logpas_mem = np.empty(shape=(self.max_steps,), dtype=np.float32)
#         self.logpas_mem[:] = np.nan

#         self.clear()

#     def clear(self):
#         self.head = 0 # how many steps do we have in mem at the current moment
        
#         # self.episode_steps = np.zeros(shape=(self.max_steps,), dtype=np.int32) # steps each episode took
#         # self.episode_reward = np.zeros(shape=(self.max_steps,), dtype=np.float32) # each episode's total return
#         # self.episode_dists = np.zeros(shape=(self.max_steps,), dtype=np.int32) # env's stats from each episode: num_distinct actions
#         # self.episode_fidels = np.zeros(shape=(self.max_steps,), dtype=np.float32) # env's stats from each episode: final state fidelity

#     def compute_gae(self, ep_rewards, ep_values):
#         T = len(ep_rewards)-1
#         deltas = ep_rewards[:-1] + self.gamma * ep_values[1:] - ep_values[:-1] # dang, nice
#         gaes = np.array([np.sum(self.tau_discounts[:T-t] * deltas[t:]) for t in range(T)])
#         return gaes

#     def fill(self, env, ok, okk):
        
#         # just logging information. Not used in the optimization
#         ep_terminated = []
#         ep_returns = []
#         ep_dists = []
#         ep_fidels = []
#         ep_acts = []

#         self.head = 0 # now clear() is not really required.
#         while self.head < self.max_steps:
            
#             # collect some episodes!

#             # rollouts = (states, actions, rewards, logpas, values, terminal)
#             rollouts = env.step()
#             new_dists, new_fidels = env.stats() # collect some stats too

#             new_head = min(self.max_steps, self.head + sum(len(rollout[1]) for rollout in rollouts))
#             # print('yeah', new_head, self.max_steps, self.states_mem.shape)
#             # print(self.head, new_head)

#             # update episode stats (for plotting)
#             # ep_idxs = list(range(self.n_workers)) # DUMMY
#             ep_dists.extend(new_dists)
#             ep_fidels.extend(new_fidels)
#             for rollout in rollouts:
#                 ep_acts.append(rollout[1])
#                 ep_terminated.append(rollout[5])
            
#             # compute the (received) returns for each step
#             new_returns = [
#                 [np.sum(self.discounts[:len(rollout[2])-t] * rollout[2][t:]) for t in range(len(rollout[2]))] 
#             for rollout in rollouts]
            
#             ep_returns.extend([new_return[0] for new_return in new_returns])
            
#             new_returns = np.concatenate(new_returns).flatten()

#             # compute the gaes at each step
#             new_gaes = np.concatenate([self.compute_gae(rollout[2], rollout[4]) for rollout in rollouts]).flatten()

#             # flatten everything
#             new_states  = np.concatenate([rollout[0][:-1] for rollout in rollouts])
#             new_actions = np.concatenate([rollout[1]      for rollout in rollouts]).flatten()
#             # new_rewards = np.concatenate([rollout[2][:-1] for rollout in rollouts])
#             new_logpas  = np.concatenate([rollout[3]      for rollout in rollouts]).flatten()
#             # new_values  = np.concatenate([rollout[4][:-1] for rollout in rollouts])

#             # update the buffers
#             rnge = new_head - self.head
#             assert len(new_states) >= rnge, f'{len(new_states)} wutt'
#             self.states_mem [self.head:new_head] = new_states [:rnge]
#             self.actions_mem[self.head:new_head] = new_actions[:rnge]
#             self.logpas_mem [self.head:new_head] = new_logpas [:rnge]
#             self.gaes_mem   [self.head:new_head] = new_gaes   [:rnge]
#             self.returns_mem[self.head:new_head] = new_returns[:rnge]

#             env.reset()
#             self.head = new_head
#             # print('steps done:', self.head, flush=True)

#         ep_idxs_int = np.arange(len(ep_fidels))#np.arange(self.max_episodes)[ep_idxs]
#         print('one fill done, damn took a while this', len(ep_fidels), 'episodes more')
#         return ep_idxs_int, ep_terminated, ep_acts, ep_returns, ep_dists, ep_fidels

#     def get_stacks(self) -> tuple[np.array, np.array, np.array, np.array, np.array]:
#         print('called', flush=True)
#         ans = (self.states_mem, self.actions_mem, 
#                 self.returns_mem, self.gaes_mem, self.logpas_mem)
#         print(ans, len(ans))

#     def update_max_episodes(self):
#         # self.max_steps = int(self.max_steps * 1.001)
#         self.clear()

#     def __len__(self):
#         return self.head
