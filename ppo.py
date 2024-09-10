import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler

import numpy as np
import wandb
import utils

from multiproc_env import MultiprocessEnv
from environment import Environment
# from buffers import EpisodeBuffer
from typing import Union
from color_codes import *

palette = [BOLD_GREEN, BOLD_CYAN, BOLD_BLUE, BOLD_YELLOW, BOLD_RED]

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
class FCCA(nn.Module):
    def __init__(self,
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        orthogonal_init(self.input_layer)
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
        
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            orthogonal_init(hidden_layer)
            self.hidden_layers.append(hidden_layer)


        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        orthogonal_init(self.output_layer, gain=1e-3)

        self.device = utils._globals['device']
        self.to(self.device)
        
    def _format(self, states):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x

    def forward(self, states) -> torch.Tensor:
        x: torch.Tensor = self._format(states)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def np_pass(self, states):
        logits = self.forward(states)
        # np_logits = logits.detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions = actions.detach().cpu().numpy()
        logpas = dist.log_prob(actions)
        np_logpas = logpas.detach().cpu().numpy()
        # is_exploratory = np_actions != np.argmax(np_logits, axis=1)
        return np_actions, np_logpas
    
    def select_action(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions.detach().cpu()
    
    def get_predictions(self, states, actions):
        states, actions = self._format(states), self._format(actions)
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()
        # print('pred', entropies)
        return logpas, entropies
    
    def select_greedy_action(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = np.argmax(logits.detach().squeeze().cpu())
        logpa = dist.log_prob(action)
        return action.item(), logpa.item(), dist.entropy().item()
    
# basically the same as FCCA
class FCV(FCCA):
    def __init__(self,
                 input_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super().__init__(input_dim, 1, hidden_dims, activation_fc)

    def forward(self, states):
        ans = super().forward(states)
        # print('ans', states.shape, ans.shape)
        return ans.squeeze(-1)
    
class PPO:
    def __init__(self, 
                 env: Union[MultiprocessEnv, Environment],
                 policy_model_fn, 
                 policy_optimizer_fn,
                 value_model_fn, 
                 value_optimizer_fn,
                 episode_buffer,#None
                 policy_model_max_grad_norm:float=float('inf'),
                #  policy_optimizer_lr:float=3e-4,
                 policy_optimization_epochs:int=8,
                 policy_sample_ratio:float=0.125,
                 policy_clip_range:float=0.2,
                 policy_stopping_kl:float=0.02,
                 value_model_max_grad_norm:float=10,
                #  value_optimizer_lr:float=5e-4,
                 value_optimization_epochs:float=8,
                 value_sample_ratio:float=0.8,
                 value_clip_range:float=0.2,#0.2,#float('inf'),
                 value_stopping_mse:float=25,
                 entropy_loss_weight:float=0.01,#0.01,
                #  tau:float=0.97,
                #  n_workers:int=8
                 ):

        self.policy_model: FCCA = policy_model_fn()
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        # self.policy_optimizer_fn = policy_optimizer_fn
        # self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimization_epochs = policy_optimization_epochs
        self.policy_sample_ratio = policy_sample_ratio
        self.policy_clip_range = policy_clip_range
        self.policy_stopping_kl = policy_stopping_kl

        self.value_model: FCV = value_model_fn()
        self.value_model_max_grad_norm = value_model_max_grad_norm
        # self.value_optimizer_fn = value_optimizer_fn
        # self.value_optimizer_lr = value_optimizer_lr
        self.value_optimization_epochs = value_optimization_epochs
        self.value_sample_ratio = value_sample_ratio
        self.value_clip_range = value_clip_range
        self.value_stopping_mse = value_stopping_mse

        self.policy_optimizer = policy_optimizer_fn(self.policy_model)
        self.value_optimizer = value_optimizer_fn(self.value_model)
        # self.policy_scheduler = scheduler.ReduceLROnPlateau(self.policy_optimizer, patience=10, verbose=True, factor=0.5)
        # self.value_scheduler = scheduler.ReduceLROnPlateau(self.value_optimizer, patience=10, verbose=True, factor=0.5)
        
        self.episode_buffer = episode_buffer

        ### KNA Change. Entropy loss is far too high becz many actions. Make it far smaller, like 1% of total loss. Right now same order
        self.entropy_loss_weight = entropy_loss_weight
        # self.tau = tau
        # self.n_workers = n_workers
        self.env = env
        self.share_memory()

        self.episode = 0
        self.steps   = 0

    def optimize_model(self):
        results = self.env.step()
        new_eps = len(results[0][0]['fidels'])
        self.episode += new_eps
        self.env.set_episode(self.episode/self.n_eps)
        # self.entropy_loss_weight = 0.001 + 0.01*(1 - self.episode/self.n_eps)

        states, actions, returns, gaes, logpas = results[0][1]
        nsteps = len(actions) * len(actions[0])
        self.steps += nsteps
        
        if utils._globals['wandb']:
            # results[0][0] contains all data that we would like to plot - final fidelity and number of steps and returns. So plot these
            wandb.log({
                'mean_fidelity': np.mean(results[0][0]['fidels']),
                'stddev_fidelity': np.std(results[0][0]['fidels']),
                'mean_gate_count': np.mean(results[0][0]['n_steps']),
                'stddev_gate_count': np.std(results[0][0]['n_steps']), 
                'mean_return': np.mean(results[0][0]['returns']),
                'stddev_return': np.std(results[0][0]['returns']),
                'mean_meta(cnots)': np.mean(results[0][0]['meta']),
                'stddev_meta(cnots)': np.std(results[0][0]['meta']),
                'episodes_done': self.episode,
                'steps_done': self.steps,
            })
        
        # optimization (policy first (at fixed value), then value) #
        
        returns = torch.tensor(returns, device=utils._globals['device'])
        gaes = torch.tensor(gaes, device=utils._globals['device'])
        logpas = torch.tensor(logpas, device=utils._globals['device'])
        states = torch.tensor(states, device=utils._globals['device'])
        actions = torch.tensor(actions, device=utils._globals['device'])
        values = self.value_model(states).detach()
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-6) # batch normalization
        assert len(states) == len(actions) == len(returns) == len(gaes) == len(logpas) == len(values)
        n_samples = len(states)
        # print(f'{n_samples=}') -> max_steps
        import time
        
        batch_size = int(self.policy_sample_ratio * n_samples)
        
        for _ in range(int(self.policy_optimization_epochs * self.policy_sample_ratio + 0.5)):
            # st = time.time()
            # print('batch size', n_samples, self.policy_sample_ratio, batch_size, flush=True)

            for batch_no in range(n_samples//batch_size):
                batch_idxs = np.arange(batch_no*batch_size, (batch_no+1)*batch_size)
                states_batch = states[batch_idxs]
                actions_batch = actions[batch_idxs]
                gaes_batch = gaes[batch_idxs]
                logpas_batch = logpas[batch_idxs]

                logpas_pred, entropies_pred = self.policy_model.get_predictions(states_batch, actions_batch)
                # print('shapes', states_batch.shape, actions_batch.shape, logpas_pred.shape, logpas_batch.shape, entropies_pred.shape, gaes_batch.shape, flush=True)
                # exit()
                # print('0', time.time() - st)
                # st = time.time()
                ratios = (logpas_pred - logpas_batch).exp().squeeze()
                pi_obj = gaes_batch * ratios
                pi_obj_clipped = gaes_batch * ratios.clamp(1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range)
                policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
                entropy_loss = -entropies_pred.mean() * self.entropy_loss_weight

                self.policy_optimizer.zero_grad()
                loss = (policy_loss + entropy_loss)
                if np.random.rand() < 0.01:
                    print('loss', policy_loss.item(), entropy_loss.item(), loss.item())
                if utils._globals['wandb']:
                    wandb.log({'loss': loss.item()}, commit=False)
                    wandb.log({'policy_loss': policy_loss.item()}, commit=False)
                    wandb.log({'entropy_loss': entropy_loss.item()}, commit=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_model_max_grad_norm)
                self.policy_optimizer.step()
            # print('1', time.time() - st)
            with torch.no_grad():
                logpas_pred_all, _ = self.policy_model.get_predictions(states, actions)
                kl = (logpas - logpas_pred_all).mean()
                if kl.item() > self.policy_stopping_kl:
                    break
            # print('2', time.time() - st)
        
        print('done policy optim')
        
        if hasattr(self, 'policy_scheduler'):        
            self.policy_scheduler.step(loss)

        target_state = self.env.tab2tensor(None)
        # target_training_batch_size = 64
        # duplicate the target state to create a batch
        # target_state = target_state.repeat(target_training_batch_size, 0)

        batch_size = int(self.value_sample_ratio * n_samples)
        
        for _ in range(int(self.value_optimization_epochs * self.value_sample_ratio + 0.5)):
            target_loss = self.value_model(target_state).pow(2).mean()
            if (target_loss > self.value_stopping_mse):
                self.value_optimizer.zero_grad()
                target_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_grad_norm)
                self.value_optimizer.step()
                
            for batch_no in range(n_samples//batch_size):
                batch_idxs = np.arange(batch_no*batch_size, (batch_no+1)*batch_size)
                states_batch = states[batch_idxs]
                returns_batch = returns[batch_idxs]
                values_batch = values[batch_idxs]
                values_pred = self.value_model(states_batch)
                # print('shapes', returns.shape, returns_batch.shape, values_pred.shape, type(values_pred), type(states_batch), states_batch.shape, values_batch.shape, flush=True)
                values_pred_clipped = values_batch + (values_pred - values_batch).clamp(-self.value_clip_range, self.value_clip_range)
                v_loss = (returns_batch - values_pred).pow(2)
                v_loss_clipped = (returns_batch - values_pred_clipped).pow(2)
                value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

                # # fake data to make the value of the target state 0
                # target_loss = self.value_model(target_state).pow(2).mean()
                # if utils._globals['wandb']:
                #     wandb.log({'value_loss': value_loss.item(), 'target_loss': target_loss.item()}, commit=False)
                # value_loss += target_loss * 0.5
                # also, train the value model to assign the target state (|0...0>) a value of 0
                # random_fidelities = np.random.random((target_training_batch_size, 1))
                # target_state_with_random_fidelities = target_state#np.concatenate((target_state, random_fidelities), axis=-1)
                # target_value = self.value_model(target_state_with_random_fidelities)
                # target_loss = target_value.pow(2).mean()
                # if np.random.rand() < 0.01:
                #     print('loss', value_loss.item(), target_loss.item(), (value_loss + target_loss*0.5).item())
                # value_loss += target_loss * 0.5

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_grad_norm)
                self.value_optimizer.step()
        
            with torch.no_grad():
                values_pred_all = self.value_model(states)
                mse = (values - values_pred_all).pow(2).mul(0.5).mean()
                if mse.item() > self.value_stopping_mse:
                    print('wow, we are done here (value)')
                    break
        

        if hasattr(self, 'value_scheduler'):
            self.value_scheduler.step(value_loss)

        print('nsteps', nsteps)
        print('neps', new_eps)
        # return nsteps

    def train(self, n_eps: int, roll_ct: int=20, mean_bound:float=0, std_tol: float=0.1, dev_envs: list[Environment]=[], start_ep=0, plot_fn=None):
        
        self.n_eps = n_eps
        # if utils._globals['wandb']:
        #     wandb.init(project='aalto', name=f'ppo-3q0.05-rew6')
        if plot_fn is not None:
            print('mhmm?', flush=True)
            plotter = Plotter(10, plot_fn)
        else:
            plotter = None
        # print(plotter, flush=True)
        results = []
        # introduce a dev set to check for overfitting
        dev_performance = []
        self.env.update_model(self.policy_model, self.value_model)
        rd = 0 # round
        try:
            # collect some rollout
            while self.episode < n_eps:
                rd += 1
                print('train start', flush=True)
                # self.env.update_model(self.policy_model, self.value_model)
                # print('updated model')
                # batch_results = (ep_idxs, ep_ter, ep_acts, ep_rews, ep_dists, ep_fidels) = \
                #     self.episode_buffer.fill(self.env, self.policy_model, self.value_model)
                # print('hola end', flush=True)
                
                # n_ep_batch = len(ep_idxs) # number of episodes run in the batch - notice, our buffer now stores complete episodes, not steps!
                
                self.optimize_model()
                print(f'{self.episode=}', flush=True)
                ##### IS THIS USEFUL? #####
                # if False:#episode % 100 == 0:
                #     self.episode_buffer.update_max_episodes()
                # else:
                #     self.episode_buffer.clear()

                # episode += n_ep_batch

                # ep, ter, act, rew, state, fidelity
                # results.extend(zip(*batch_results))
                
                # plot
                print('skipping plotting')
                # if plotter:
                    # print("plotting, plotting")
                    # print(len(results))
                    # plotter.plot(results)

                if False:#rd % 5 == 0:
                    print(BLUE+'Episodes:', episode+start_ep, RESET)
                    print(ep_acts[0][:10], ep_acts[0][-10:], len(ep_acts[0]))
                    print(ep_fidels, len(ep_fidels), len(ep_rews), len(ep_acts))
                    print(min(ep_fidels), max(ep_fidels), np.mean(ep_fidels))
                    if utils._globals['wandb']:
                        wandb.log({'mean-fidelity': np.mean(ep_fidels)})
                        # log average reward and average number of gates used too
                        wandb.log({'mean-reward': np.mean(ep_rews)})
                        wandb.log({'mean-gates': sum(len(ep_act) for ep_act in ep_acts)/len(ep_acts)})
                
                # check whether we are good to go
                if len(results) < roll_ct: continue
                
                if rd % 5 == 0:
                    rolling_rews = torch.tensor([res[3] for res in results[-roll_ct:]], dtype=torch.float32)
                    rolling_truncated = roll_ct-sum([res[1] for res in results[-roll_ct:]])
                    print(BOLD_CYAN, rolling_rews.min().item(), rolling_rews.mean().item(), rolling_rews.std().item())
                    print(BOLD_GREEN, rolling_truncated, RESET)
                    print(palette[int((rolling_truncated/roll_ct)*(len(palette)-1) + 1e-6)], "█"*100)
                # if rolling_rews.mean().item() > mean_bound + 1e-6 and rolling_rews.std().item() < std_tol:
                #     print(rolling_rews.min().item(), rolling_rews.mean().item(), rolling_rews.std().item())
                #     print(rolling_truncated)
                #     break

                # check on the dev set
                # if len(dev_envs) > 0:
                #     dev_results = []
                #     for dev_env in dev_envs:
                #         dev_results.extend(self.evaluate(dev_env))
                #     dev_performance.append(np.mean([res[0] for res in dev_results]))
                #     print(BOLD_YELLOW, dev_performance[-1], RESET)
                #     if len(dev_performance) > n_eps/10 and all(dev_performance[-1] < dev_performance[-i-1] for i in range(1, roll_ct//2)):
                #         print(BOLD_RED, "Overfitting detected!", RESET)
                #         break

        except KeyboardInterrupt:
            print(BOLD_RED)
            print('Training interrupted by user.')
            print(RESET)
            print(f'Done training in {self.episode} episodes.')
            print(f'{len(results)=}')
            return results, self.episode+start_ep

        print(RESET)
        print(f'Done training in {self.episode} episodes.')
        return results, self.episode+start_ep

    def evaluate(self, eval_env: Environment, n_eps=1):
        terminal = truncated = False
        stats = []
        for _ in range(n_eps):
            rew = 0; acts = []
            s, _ = eval_env.reset()
            print('stt', s)
            a = None
            while not terminal and not truncated:
                a = self.policy_model.select_action(s)
                s, r, terminal, truncated, _ = eval_env.step(a)
                # print('i don"t believe it', s, a, eval_env.gates[a], flush=True)
                terminal = terminal[0]
                truncated = truncated[0]
                rew += r[0]; acts.append(a[0]) #### Future stats: Record the path (ie s2 or self.agent.state as well and plot fidelity vs time (steps).
            print('ter', terminal, truncated, s, eval_env.tab2tensor(eval_env.start_state), flush=True)
            meta, fidel = eval_env.stats()
            stats.append((rew, acts, meta[0], fidel[0]))
            terminal = truncated = False
        return stats
    
    def share_memory(self):
        dev = 'cuda:0' if utils._globals['device'] == 'cuda' else 'cpu'
        self.policy_model.to(dev).share_memory()
        self.value_model .to(dev).share_memory()

import time
class Plotter:
    def __init__(self, interval, plot_fn) -> None:
        self.interval = interval
        self.plot_fn = plot_fn
        self.tic = time.time()
    
    def plot(self, *plot_args):
        if time.time() - self.tic <= self.interval:
            return
        self.plot_fn(*plot_args)
        self.tic = time.time()

