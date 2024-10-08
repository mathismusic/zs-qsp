import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Union
import utils
from buffers import ExperienceProcessor

class MultiprocessEnv:
    """A parallel implementation of multiple environments. It achieves the same using workers that actually go forage in their own environments and send the results back to the main 'engine', which returns the vectorized results back to the calling program."""
    def __init__(self, make_env_fn, seed, n_workers):
        self.make_env_fn = make_env_fn
        self.seed = seed
        self.n_workers = n_workers
        self.policy_model = None
        self.value_model = None
        print('HERE', __name__)
        if __name__ == 'multiproc_env': # important
            mp.set_start_method('spawn')
            self.event = mp.Event()
            self.pipes = [mp.Pipe(duplex=True) for _ in range(self.n_workers)]
            self.workers = [mp.Process(
                target=MultiprocessEnv.work, 
                args=(
                    self.make_env_fn(None if self.seed is None else self.seed + i), 
                    i, 
                    self.pipes[i][1],
                    utils._globals['device'],
                    self.event,
                ),
                kwargs={
                    'bufsize':utils._globals['bufsize'],
                    'gamma':utils._globals['gamma'],
                    'tau':utils._globals['tau'],
                    'num_envs':utils._globals['num_envs'],
                    # 'device':utils._globals['device'],
                },
            ) for i in range(self.n_workers)]
            self.dones = {i:False for i in range(self.n_workers)}
            print(self.n_workers, "workers created")
            # [w.start() for w in self.workers]
            for w in self.workers:
                print("starting worker")
                w.start()
                print("worker started", flush=True)
            print('ok4', flush=True)
            self.reset()
    
    @staticmethod
    def work(env, rank: int, worker_end: Connection, device: str, event, **kwargs):
        """The function that each worker runs. Each worker receives (from the engine) a command and arguments, that it executes in its copy `env` of the environment and then sends the results to the engine"""
        if device == 'cuda':
            print(torch.cuda.device_count(), "this much devices available", flush=True)
            # torch.cuda.set_device(1 + rank%(torch.cuda.device_count()-1))
            torch.cuda.set_device(rank%(torch.cuda.device_count())) # if only one device available
        print(f'Thread {rank} STARTED on {torch.cuda.current_device() if device == "cuda" else "cpu"}', flush=True)
        # buf = buffers.EpisodeBuffer((env.state_size,), utils._globals['gamma'], utils._globals['tau'], 1, utils._globals['bufsize'], env.max_steps)

        # def next_action(state, policy_model):
        #     # print(state.shape)
        #     with torch.no_grad():
        #         action, logpa = policy_model.np_pass(state)
        #     # print(action.shape, logpa.shape)
        #     return action, logpa

        # state = None#torch.zeros((env.num_envs, env.state_size), dtype=torch.float32)
        while True:
            # print("yes")
            event.wait()
            cmd, kw = worker_end.recv() # recv from the engine
            if cmd == 'update-model':
                # these are shared across workers
                policy_model = kw['policy']
                value_model = kw['value']
            elif cmd == 'reset':
                state, info = env.reset()
                worker_end.send((state, info))
            elif cmd == 'step':
                # run a full buffer on the worker
                # print(kwargs)
                # print(device,flush=True)
                proc = ExperienceProcessor((env.state_size,), kwargs['gamma'], kwargs['tau'], max_steps=kwargs['bufsize'], max_steps_per_episode=env.max_steps, device=device, num_envs=kwargs['num_envs'])
                results = proc.fill(env, policy_model, value_model)
                worker_end.send((results, proc.get_stacks()))
                # print('confirmed-step')
            elif cmd == 'stats':
                # print("nope not allowed")
                worker_end.send(env.stats(**kw))
            elif cmd == 'close':
                env.close(); del env
                worker_end.close()
                return
            elif cmd == 'tab2tensor':
                if kw['states'] is None:
                    worker_end.send(env.tab2tensor([env.target_state]))
                else:
                    worker_end.send(env.tab2tensor(**kw))
            elif cmd == 'set-episode':
                env.set_episode(kw['episode'])
            else: 
                assert False, f'invalid command `{cmd}` received from engine'

    def step(self) -> list:
        # run a full episode on each worker

        """returns vectorized (ss, rs, ters, trs, infos). infos is a list, not a tensor."""
        self.event.set()
        [pipe[0].send(('step', {})) for pipe in self.pipes]
        # ss = []; rs = []; ters = []; trs = []; infos = []
        data = []
        for engine_end, worker_end in self.pipes:
            data.append(engine_end.recv()) # ith worker's (s', r, ter, tr) values over episode. Last value is the value of the last state
            # print(len(data), data[0], flush=True)
        self.event.clear()
        return data
    
    def close(self, **kwargs):
        self.broadcast(('close', kwargs))
        [w.join() for w in self.workers]

    def broadcast(self, msg):
        self.event.set()
        [pipe[0].send(msg) for pipe in self.pipes]
        self.event.clear()

    def reset(self, ranks: Union[list[int], None]=None, **kwargs):
        # ranks: list[int]|None=None
        if ranks is None:
            ranks = list(range(self.n_workers))
        
        # ss = []; infos = []
        self.event.set()
        for rank in ranks:
            parent_end, _ = self.pipes[rank]
            parent_end.send(('reset', kwargs))
            parent_end.recv() # clear the pipe
            # ss.append(s.cpu().numpy())
            # infos.append(info)
        self.event.clear()
        # return np.stack(ss), infos
 
    def stats(self, ranks: Union[list[int], None]=None) -> tuple[np.array, np.array]:
        # ranks: list[int]|None=None
        if ranks is None:
            ranks = list(range(self.n_workers))
        
        n_dist = [None]*len(ranks)
        fidels = [None]*len(ranks)
        self.event.set()
        for i, rank in enumerate(ranks):
            parent_end, _ = self.pipes[rank]
            parent_end.send(('stats', {}))
            n_dist[i], fidels[i] = parent_end.recv()
        self.event.clear()
        return np.array(n_dist), np.array(fidels)
    
    def update_model(self, policy_model, value_model):
        data = {'policy': policy_model, 'value': value_model}
        self.broadcast(('update-model', data))

    def tab2tensor(self, states=None):
        # just ask a random worker to do it
        self.event.set()
        parent_end, _ = self.pipes[0]
        parent_end.send(('tab2tensor', {'states':states}))
        ans = parent_end.recv()
        self.event.clear()
        return ans

    def set_episode(self, episode):
        self.broadcast(('set-episode', {'episode':episode}))