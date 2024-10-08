# import torch
import numpy as np

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from color_codes import *
import utils
import stim
import inspect
import copy

from typing import Any, Union
import torch, torch.nn as nn, torch.optim as optim
class TabEmbed(nn.Module):
    def __init__(self, qubits, state_size):
        super(TabEmbed, self).__init__()
        self.qubits = qubits
        self.tabsize = state_size
        self.embed = nn.Embedding(4, 2)
        self.fc1 = nn.Linear(2 * self.tabsize, qubits * qubits)
        self.fc2 = nn.Linear(qubits * qubits, 2 * qubits) # final embedding is of size 2*qubits
        self.output_size = 2 * qubits

    def forward(self, x):
        # print(x.shape, flush=True) # 1000, 2n^2+n
        x = self.embed(x)
        # print(x.shape, flush=True) # 1000, 2n^2+n, 2
        x = x.reshape(-1, 2 * self.tabsize) # 2000, 2n^2+n
        # print(x.shape, flush=True) # 2000, 2n^2+n
        x = torch.relu(self.fc1(x))
        # print(x.shape, flush=True) # 2000, n^2
        x = self.fc2(x)
        # print(x.shape, flush=True) # 2000, 2n
        return x
    
    def train(self, tabgenerator, tab2tensor, batch_size=64, epochs=20):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            states = [tabgenerator(self.qubits) for _ in range(batch_size)]
            tab = tab2tensor(states)
            tab = torch.tensor(tab, dtype=torch.long)
            optimizer.zero_grad()
            output = self(tab)
            fidels = [[0.]*batch_size for _ in range(batch_size)]
            for i in range(batch_size):
                for j in range(i+1, batch_size):
                    fidels[i][j] = utils.fidelity(states[i], states[j])
                    fidels[j][i] = fidels[i][j]
                fidels[i][i] = 1.0
            fidels = torch.tensor(fidels, dtype=torch.float32)
            distances = torch.norm(output.unsqueeze(0) - output.unsqueeze(1), dim=-1)
            loss = criterion(distances, fidels)
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch+1}/{epochs}, loss={loss.item()}')


# environment for preparing clifford states. states are stored as tableaus.
class Environment:
    def __init__(self, num_envs: int, target_state: stim.Tableau, fidelity_tol: float, max_steps: int, seed: Union[int, list[int], None], _start_state=None) -> None:
        self.rcalls = 0
        self.qubits = len(target_state)
        self.seed = seed
        self.num_envs = num_envs
        if self.seed:
            print(self.seed, "haa", flush=True)
            if isinstance(self.seed, int): self.seed = [self.seed + i for i in range(self.num_envs)]
            print(self.seed, flush=True)
            self.rng = [np.random.default_rng(sd) for sd in self.seed]
            
        self.gates, self.ckts, self.targets, self.inv_qiskit_ckts, self.basic_gate_count = Environment.prepare_gatelist(self.qubits)
        # print('\n'.join(f'{i}: {g}' for i, g in enumerate(self.gates)))
        
        self.action_space = np.arange(len(self.gates))
        
        if self.seed:
            self.set_start_state()
        else:
            assert _start_state is not None, 'error environment.py: seed is None and _start_state is None.'
            assert isinstance(_start_state, stim.Tableau)
            self.start_state: list[stim.Tableau] = [_start_state.copy() for _ in range(self.num_envs)]

        self.target_state = target_state        
        self.state = copy.deepcopy(self.start_state)
        # print("yessir here")
        # print(self.start_state.to_state_vector())
        # print(self.target_state.to_state_vector())
        # print(self.state.to_state_vector())

        # print(f'{type(self.start_state)=}, {type(self.target_state)=}, {type(self.state)=}')
        # check that all are stim Tableaus.
        
        self.tol = fidelity_tol
        self.max_steps = max_steps
        self.steps_left = np.array([max_steps for _ in range(self.num_envs)])
        self.meta_actions = np.zeros((self.num_envs,)) # keep track of the count of a particular class of actions; currently actions that improved trace distance/fidelity etc
        self.device = 'cpu'#utils._globals['device'] # the device to return all tensors to. just return to cpu ig
        self.way = utils._globals['rewardtype']

        self.prev_action = -1
        self.prev_state = None
        self.disallowed = None
        self.state_size = self.qubits * (2 * self.qubits + 1) + 1
        self.logscale = False#self.way in [12]
        self.maxfid = self.curr_fidelity(logscale=self.logscale)
        self.k = int(np.sqrt(self.max_steps))
        self.maxfid_k = [self.maxfid] * self.k

        self.minl1 = 0#self.L1(self.start_state, self.target_state)

        self.max_fidel_change = np.maximum(1.0 - self.maxfid, self.tol)

        # dummy info dictionary
        self.info = {}

        self.to_reset = np.zeros(0, dtype=int)#np.zeros((self.num_envs,), dtype=int)
        self.fidelity_of_resetted = np.zeros(0, dtype=np.float32)

        self.use_embedding = False
        if self.use_embedding:
            print('training embedding')
            self.tabembed = TabEmbed(self.qubits, self.state_size-1)
            self.tabembed.train(utils.make_random_tableau, self._tab2tensor, batch_size=64, epochs=20)
        if self.use_embedding:
            self.state_size = self.tabembed.output_size + 1

        self.maxfid_used = False
        self.state_size = self.state_size - 1 + self.maxfid_used

        self.episode_frac = 0

    def set_start_state(self, idxs: Union[np.ndarray, None]=None):
        # self.start_state = utils.make_random_tableau(self.qubits) # seed is not passed, note!
        if idxs is not None:
            # print('idxs on reset', idxs, flush=True)
            for idx in idxs:
                # print(idx, flush=True)
                self.start_state[idx] = utils.make_random_tableau(self.qubits)
        else:
            self.start_state = [utils.make_random_tableau(self.qubits) for _ in range(self.num_envs)]

    @staticmethod
    def prepare_gatelist(qubits):
        gates = []
        inv_gates = []
        ckts = []
        targets = []
        basic_gate_count = []
        htab = stim.Tableau.from_named_gate('H')
        stab = stim.Tableau.from_named_gate('S').inverse()
        hshtab = htab.copy()
        hshtab.append(stab, [0])
        hshtab.append(htab, [0])
        ztab = stim.Tableau.from_named_gate('Z')
        xtab = stim.Tableau.from_named_gate('X')
        ytab = stim.Tableau.from_named_gate('Y')
        cnottab = stim.Tableau.from_named_gate('CNOT')
        for i in range(qubits):
            gates.append(f'h({i})')
            inv_gates.append(f'h({i})')
            ckts.append(htab)
            targets.append([i])
            basic_gate_count.append(1)

            # hsdgh before sdg is the right one for 5,7 qubit.

            # this is the right one for 5 qubit to work
            gates.append(f'hsdgh({i})')
            inv_gates.append(f'hsh({i})')
            ckts.append(hshtab)
            targets.append([i])
            basic_gate_count.append(3)
            
            gates.append(f'sdg({i})')
            inv_gates.append(f's({i})')
            ckts.append(stab)
            targets.append([i])
            basic_gate_count.append(1)

            # gates.append(f'z({i})')
            # inv_gates.append(f'z({i})')
            # ckts.append(ztab)
            # targets.append([i])

            # gates.append(f'x({i})')
            # inv_gates.append(f'x({i})')
            # ckts.append(xtab)
            # targets.append([i])

            # gates.append(f'y({i})')
            # inv_gates.append(f'y({i})')
            # ckts.append(ytab)
            # targets.append([i])

        for i in range(qubits):
            for j in range(qubits):
                if i == j: continue
                gates.append(f'cnot({i},{j})')
                inv_gates.append(f'cnot({i},{j})')
                ckts.append(cnottab)
                targets.append([i, j])
                basic_gate_count.append(1)

        return gates, ckts, targets, [], basic_gate_count
    
    def step(self, action: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        self.steps_left -= 1
        # print(f'{self.start_state.to_state_vector()=} {self.target_state.to_state_vector()=}')
        # print(f'{self.steps_left=}, {self.state.to_state_vector()}, {self.curr_fidelity()}')
        if utils._globals['debug']:
            print(BLUE + 'current:', self.state, CYAN, action)
            old_fid = self.curr_fidelity()
        # print('are they the same?', self.tab2tensor(self.state), self.tab2tensor(self.start_state), flush=True) - they were, big oof
        self.prev_action = action
        self.prev_state = self.state.copy()
        print('action!', action, flush=True)
        # print('state!', self.state[0].to_stabilizers(canonicalize=True), flush=True)
        self.take_action(action)
        # print('state-new!', self.state[0].to_stabilizers(canonicalize=True), flush=True)
        # self._reset(self.to_reset)

        newfid = self.curr_fidelity()
        self.maxfid_k = self.maxfid_k[1:] + [newfid] ### ADDED ###
        terminal = newfid > 1 - self.tol
        truncated = self.steps_left <= 0
        # if terminal:
        #     print('finished'); self.state_info()
        # elif truncated:
        #     print('truncated'); self.state_info()
        rew = self.reward_fn(self.disallowed)
        l1_prev_step = self.l1_dists_prev_step.copy()
        if self.way == 11:
            qiskit_dists_prev_step = self.qiskit_dists_prev_step.copy()
        
        self.meta_actions += (newfid >= self.maxfid + 1e-6)
        self.maxfid = np.maximum(self.maxfid, newfid)
        # self.minl1 = min(self.minl1, self.L1(self.state, self.target_state))
        self.to_reset = ((terminal|truncated) == True).nonzero()
        assert len(self.to_reset) == 1, 'ouch self.to_reset is large'
        self.to_reset = self.to_reset[0]
        # print(self.to_reset, flush=True)
        self.fidelity_of_resetted = newfid[self.to_reset] # for logging. Perhaps add these logging things to the info dict.
        self.meta_actions_of_resetted = self.meta_actions[self.to_reset]
        self._reset(self.to_reset) # moved it here, why was it before?
        if utils._globals['debug'] and self.way != 8:
            print(RED, f'stats: {old_fid:.2f}->{newfid:.2f} ({self.maxfid:.2f}, {terminal}, {truncated}, {rew}, {self.meta_actions})')
        new_states = self.state_to_tensor()
        if self.way in [8, 9, 10, 11]:
            # if np.random.random() > 1: 
            #     print(l1_prev_step, self.l1_dists_prev_step, flush=True, sep='\n')
            rew_tableau = (1 * l1_prev_step - self.l1_dists_prev_step)
            if self.way == 11:
                rew_qiskit  = (qiskit_dists_prev_step - self.qiskit_dists_prev_step)/self.max_steps
                rew_qiskit = np.clip(rew_qiskit, 0, 1) * 0.5
            if self.way >= 9:
                assert 0 <= self.episode_frac <= 1, 'episode frac is not in [0, 1]'
                if self.way == 9:
                    rew_tableau *= self.steps_left/self.max_steps # this works better
                elif self.way == 10:
                    rew_tableau *= 1#(1 - self.episode_frac)
                elif self.way == 11:
                    # print('rew_qiskit', rew_qiskit, end=' ')
                    rew_qiskit *= (self.steps_left/self.max_steps)
                    # print(rew_qiskit)
            if self.way in [8, 9, 10]:
                rew += rew_tableau
            elif self.way == 11:
                rew += rew_qiskit
            # import wandb
            # if utils._globals['wandb'] and wandb.run is not None:
            #     wandb.log({'reward_min': np.min(rew), 'reward_max': np.max(rew), 'reward_mean': np.mean(rew)})
            # print('reward', rew, l1_prev_step, self.l1_dists_prev_step, flush=True)
            if np.allclose(rew, 0):
                print('ZERO REWARD', self.state, self.prev_state, flush=True, sep='\n')
        return new_states, rew, terminal, truncated, self.info
    
    # sort of hack to record the episode number (roughly) to set the reward accordingly
    def set_episode(self, episode: int) -> None:
        self.episode_frac = episode

    def close(self):
        self.reset()
    
    # internal function
    def take_action(self, action: list[int]) -> None:
        def apply(s: stim.Tableau, a: int):
            s.append(self.ckts[a], self.targets[a])
        [apply(*arg) for arg in zip(self.state, action)]
        # self.state.append(self.ckts[action], self.targets[action])

    # internal function
    def curr_fidelity(self, idxs: Union[np.ndarray, None]=None, logscale=False) -> np.ndarray:
        # treat some L1 distance as part of fidelity as well?
        if idxs is not None:
            return np.array([utils.fidelity(self.state[idx], self.target_state, logscale=logscale) for idx in idxs])
        else:
            return np.array([utils.fidelity(s, self.target_state, logscale=logscale) for s in self.state])
    
    # internal function
    def state_to_tensor(self, states=None, aux=None) -> np.ndarray:
        s_arr = self.tab2tensor(states)
        if not self.maxfid_used:
            return s_arr
        aux_arr = aux if aux else np.array(self.maxfid)
        assert np.concatenate((s_arr, aux_arr[:, None]), axis=-1).shape == (self.num_envs, self.state_size), 'oopss'
        # print(s_arr.shape, aux_arr.shape)
        return np.concatenate((s_arr, aux_arr[:, None]), axis=-1)
        # return torch.cat((s_arr, aux_arr), dim=-1)

    def _tab2tensor(self, states=None) -> tuple[np.ndarray, np.ndarray]:
        assert states is None or (isinstance(states, list) and all(isinstance(state, stim.Tableau) for state in states)), f'{states=}, {type(states)=}, called from {inspect.stack()[1].function}'
        if states is None:
            states = self.state
        arrays = [None for _ in range(len(states))]; l1_dists = [1.]*len(states)
        # print('current states', states, flush=True)
        for i, state in enumerate(states):
            arrays[i], l1_dists[i] = utils.tableau2array(state)
        self.l1_dists_prev_step = np.array(l1_dists) # on step(), this has the state before the action is taken's l1 distances.
        self.qiskit_dists_prev_step = []
        need_to_calc = self.way == 11
        if not need_to_calc:
            return np.array(arrays), self.l1_dists_prev_step
        for state in states:
            gens = list(map(str, state.to_stabilizers(canonicalize=True)))
            qiskit_circ = bravyi_circuit(gens)
            dist = sum(qiskit_circ.count_ops().values())
            self.qiskit_dists_prev_step.append(dist)
        # print('qiskit-for-next-step', self.qiskit_dists_prev_step, flush=True)
        self.qiskit_dists_prev_step = np.array(self.qiskit_dists_prev_step)
        # print('l1-for-next-step', self.l1_dists_prev_step, flush=True)
        return np.array(arrays), self.l1_dists_prev_step
    
    def tab2tensor(self, states=None) -> np.ndarray:
        # apply the embedding too
        if self.use_embedding:
            return self.tabembed(torch.tensor(self._tab2tensor(states)[0], dtype=torch.long)).detach().numpy()
        return self._tab2tensor(states)[0]
            
    def reward_fn(self, disallowed: list[bool]) -> list[float]:
        # need a way to increase the allowed number of steps for hard to prepare states, I think that's why agent is stagnating. A certain class of states is difficult to prepare. So we need to give it more steps, but only in these cases. How do we catch these cases, and not penalize reward unnecessarily? Plus, we need to make sure that the agent is encouraged not to use the extra steps for easier states.
        # fidel: np.ndarray = self.curr_fidelity()
        if self.way == 3:
            fidel: np.ndarray = self.curr_fidelity()
            return fidel
        elif self.way == 2:
            return -trace_distance(self.state, self.target_state)
        elif self.way == 1:
            fidel: np.ndarray = self.curr_fidelity()
            return np.maximum(0, fidel - self.maxfid)
        elif self.way == 0:
            return np.ones((self.num_envs,)) * -1/self.max_steps
            return np.maximum(0, fidel - self.maxfid) - 1/self.max_steps #TODO change to window max-fidelity over last k=sqrt self.max_steps steps.
            #TODO more aggressive penalizing
        elif self.way == 4:
            fidel: np.ndarray = self.curr_fidelity()
            return np.maximum(0, fidel - self.maxfid) - 1/self.max_steps
        elif self.way == 5:
            fidel: np.ndarray = self.curr_fidelity()
            # goal_tensor = self.tab2tensor(self.target_state)
            # curr_tensor = self.tab2tensor(None)
            # prev_tensor = self.tab2tensor(self.prev_state)
            # return ((prev_tensor != goal_tensor).sum().item() - (curr_tensor != goal_tensor).sum().item())/(self.state_size - 1) - 1/self.max_steps + max(0, fidel - self.maxfid)
            L1_prev_curr = self.L1(self.prev_state, self.target_state)
            L1_curr_goal = self.L1(self.state, self.target_state)
                # - L1_curr_goal \
                # + (L1_curr_goal < L1_prev_curr) / (self.state_size - 1) \
                # + min(0, L1_curr_goal - L1_prev_curr) \
            return \
                + (L1_curr_goal - L1_prev_curr) \
                + min(0, L1_curr_goal - self.minl1) \
                + max(0, fidel - self.maxfid) \
                - 1/self.max_steps
        elif self.way == 6:
            # sliding window
            fidel: np.ndarray = self.curr_fidelity()
            maxfid_k = np.max(self.maxfid_k, axis=-1)
            return np.maximum(0, fidel - maxfid_k) - 1/self.max_steps
        elif self.way == 7:
            fidel: np.ndarray = self.curr_fidelity()
            # aggressive penalizing (as steps increase, penalize more? not markovian though)
            return np.maximum(0, fidel - self.maxfid) - 2 * (1/self.max_steps)
        elif self.way == 8:
            return np.zeros((self.num_envs,))
        elif self.way == 9:
            fidel: np.ndarray = self.curr_fidelity()
            return np.maximum(0, fidel - self.maxfid) - 1/self.max_steps
        elif self.way == 10:
            fidel: np.ndarray = self.curr_fidelity()
            return self.episode_frac * (np.maximum(0, fidel - self.maxfid) - 1/self.max_steps)
        elif self.way == 11:
            fidel: np.ndarray = self.curr_fidelity()
            return np.maximum(0, fidel - self.maxfid) - 1/self.max_steps
        elif self.way == 12:
            # log fidelity is used
            logfidel: np.ndarray = self.curr_fidelity(logscale=True)
            # maxfid is also in log, when reward is 12
            # -- not yet
            logmaxfid = np.log(np.array(self.maxfid, dtype=np.float64) + 2**-self.qubits) # check for underflow
            logmaxfid = np.array(logmaxfid, dtype=np.float32)
            assert np.min(logmaxfid) >= -self.qubits - 1e-6, 'logmaxfid is too small' + str(logmaxfid)
            assert np.min(logfidel) >= -self.qubits - 1e-6, 'logfidel is too small' + str(logfidel)
            return np.maximum(0, logfidel - logmaxfid)/self.qubits - 1/self.max_steps
        
    def L1(self, state1: stim.TableauSimulator, state2: stim.TableauSimulator):
        state1_as_tensor = self.tab2tensor(state1)
        state2_as_tensor = self.tab2tensor(state2)
        return (state1_as_tensor != state2_as_tensor).sum().item()/(self.state_size - 1)

    def stats(self, idxs: Union[list[int], None]=None):
        # change to returning a dict, of the self.fidelity_of_resetted, self.meta_actions_of_resetted values.
        self._stats = self.meta_actions_of_resetted, self.fidelity_of_resetted
        # self._stats =\
        #     self.meta_actions[idxs], \
        #     self.curr_fidelity(idxs)
        return self._stats
    
    def get_inverted_ckt(self, actions: list[int]):
        ckt = QuantumCircuit(self.qubits)
        for action in actions:
            ckt.compose(self.inv_qiskit_ckts[action], range(self.qubits), inplace=True)
        return ckt
    
    # set up for the next episode (whichever terminated or truncated)
    def reset(self) -> tuple[np.ndarray, Any]:
        self.rcalls += 1
        if utils._globals['debug']:
            print(GREEN + f'==============EPISODE COMPLETE ({self.max_steps - self.steps_left})==============')
        
        idxs = np.arange(self.num_envs)
        self._reset(idxs)
        # if self.seed is not None:
        #     self.set_start_state()
        # self.state = copy.deepcopy(self.start_state)
        # self.steps_left = np.array([self.max_steps for _ in range(self.num_envs)])
        # self.prev_state = None
        # self.prev_action = -1
        # self.meta_actions = 0
        # self.maxfid = self.curr_fidelity()
        ans = self.state_to_tensor(), self.info
        # print(ans[0])
        return ans
    
    def _reset(self, idxs: np.ndarray):
        # print(idxs, flush=True)
        if self.seed is not None:
            self.set_start_state(idxs)
        for idx in idxs:
            self.state[idx] = self.start_state[idx].copy()
        self.steps_left[idxs] = self.max_steps
        self.meta_actions[idxs] = 0
        fids_reset = self.curr_fidelity(idxs)
        if len(fids_reset) > 0:
            print(self.steps_left[idxs], fids_reset, fids_reset.min(), fids_reset.max(), fids_reset.mean(), flush=True)
            # print(self.tab2tensor(self.start_state), self.tab2tensor(self.state), self.tab2tensor([self.target_state]), flush=True)
        self.maxfid[idxs] = self.curr_fidelity(idxs)
    
    def _state_info(self):
        print(self.state_to_tensor(), self.state_to_tensor(self.target_state), sep='\n---\n')
        print('final fidelity', self.curr_fidelity())

    def num_basic_gates(self, actions: list[int]) -> int:
        return sum(self.basic_gate_count[a] for a in actions)

    # train an embedding for the tableaus, which we will use as the actual state. The embedding must have the property that the distance between two tableaus is proportional to the fidelity between them.

# use a weaker model as the measure of progress for the stronger one
def bravyi_circuit(stabilizer_generators: list[str]) -> QuantumCircuit:
    gens = [l.replace('_', 'I') for l in stabilizer_generators]
    return qi.StabilizerState.from_stabilizer_list(gens).clifford.to_circuit()

# perhaps it helps the reward function
from qiskit.quantum_info import DensityMatrix
def trace_distance(s1: qi.Statevector, s2: qi.Statevector) -> float:
    dm = DensityMatrix(s1).data - DensityMatrix(s2).data
    return np.linalg.norm(dm, ord='nuc').astype(np.float32)/2 # the 1-norm is equal to the nuclear norm and is the sum of the matrix's singular values.
