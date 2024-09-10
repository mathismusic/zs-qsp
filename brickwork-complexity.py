import os
os.system('which python3')
import subprocess
qbits = 9
n = 100
# model_hyp = f'{qbits},0.1,final-submission-nopenalty,08-09-2024'
model_hyp = f'{qbits},0.25,pushlimits-max2,04-09-2024'
layers = list(range(16, 19))

gen_string = lambda d: f'srun --gpus=1 --time=10:00 --mem=5G python3 random_testbench.py -n {n} -just-gen {qbits} -name random-test-{qbits}q-brick{d} -dist "clifford-brickwork{d}"'

run_string = lambda d: f'srun --gpus=1 --time=10:00 --mem=5G python3 random_testbench.py -hyp {model_hyp} -n {n} -name random-test-{qbits}q-brick{d} -v 0'

def process(output):
    print(output)
    acts        = eval(output[-1][5:])
    max_gates   = int(output[-2].split()[-1])
    min_gates   = int(output[-3].split()[-1])
    median_gates= int(output[-4].split()[-1])
    mean_gates  = float(output[-5].split()[-1])
    return acts, max_gates, min_gates, median_gates, mean_gates

def main(layers, gen_phase):
    outputs = []
    for i in layers:
        if gen_phase:
            out = subprocess.check_output(gen_string(i), shell=True)
        else:
            out = subprocess.check_output(run_string(i), shell=True)
            with open(f'random-test-{qbits}q-bricklog.txt', 'a') as f:
                f.write(out.decode())
            out = out.decode().split('\n')[-40:-1]
            print('\n'.join(out))
            bm_greedy_out = process(out[-5:])
            bm_out = process(out[-11:-6])
            ag_out = process(out[-17:-12])
            ours_out = process(out[-26:-21])
            outputs.append((ag_out, bm_out, bm_greedy_out, ours_out))
    return outputs

import sys
gen_phase = len(sys.argv) > 1 and sys.argv[1] == 'gen'
if gen_phase:
    outputs = main(layers, gen_phase)
else:
    outputs = main(layers, gen_phase)
    print(outputs, len(outputs))
    with open(f'brickwork-complexity-{qbits}.txt', 'a') as f:
        f.write(model_hyp + '\n')
        f.write(str(outputs))
    # plot_results(outputs)


    # outputs = [((23, 4, 14, 13.49), (24, 2, 10, 10.33)), ((33, 6, 18, 17.85), (27, 3, 12, 12.42)), ((32, 4, 19, 19.64), (23, 6, 13, 13.11))]
    # outputs += [((35, 6, 21, 20.99), (23, 5, 14, 13.8)), ((33, 8, 20, 20.45), (25, 7, 14, 14.11)), ((36, 11, 21, 20.98), (28, 8, 14, 14.56))]
    # outputs += [((33, 12, 22, 21.94), (22, 8, 14, 14.22)), ((31, 10, 22, 21.59), (27, 4, 14, 14.16)), ((34, 9, 22, 21.51), (25, 9, 14, 14.23)), ((35, 10, 22, 21.59), (28, 5, 14, 14.21)), ((31, 9, 23, 22.48), (29, 5, 14, 14.22)), ((32, 4, 21, 20.66), (24, 5, 14, 14.6)), ((34, 9, 22, 21.39), (26, 8, 14, 14.09)), ((34, 9, 21, 21.71), (26, 7, 15, 14.55)), ((33, 7, 23, 21.94), (27, 6, 14, 13.94)), ((32, 8, 21, 20.91), (24, 8, 14, 14.88))]


# def plot_results(outputs):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     feats = ['M', 'm', '$\\hat{\\mu}$', '$\\mu$']
#     x = np.arange(len(feats))
#     width = 0.35  # the width of the bars
#     qisk_outputs = [o[0][1:] for o in outputs]
#     our_outputs = [o[1][1:] for o in outputs]
#     # Plot
#     fig, ax = plt.subplots(1, len(outputs), sharey=True, figsize=(20, 5))
#     for i, (qisk, ours) in enumerate(zip(qisk_outputs, our_outputs)):
#         ax[i].bar(x - width/2, qisk, width, label='Qiskit')
#         ax[i].bar(x + width/2, ours, width, label='Ours')
#         # Add labels, title, and custom ticks
#         ax[i].set_xticks(x)
#         ax[i].set_xticklabels(feats)
#     ax[len(outputs)-1].legend()
#     ax[0].set_ylabel('Performance')

#     fig.suptitle(f'Brickwork Complexity of {qbits}-qubit clifford circuits')
#     plt.savefig(f'brickwork-complexity-{qbits}-bar.png')

#     # also plot over xaxis=layers, four graphs for each of the feats, blue for qiskit and green for ours
#     fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 20))
#     for i in range(4):
#         ax[i].plot(layers, [q[i] for q in qisk_outputs], label='Qiskit')
#         ax[i].plot(layers, [o[i] for o in our_outputs], label='Ours')
#         ax[i].set_title(feats[i])
#     ax[3].legend()
#     ax[0].set_ylabel('Performance')
#     ax[0].set_xlabel('Number of layers')
#     fig.suptitle(f'Brickwork Complexity of {qbits}-qubit clifford circuits')
#     plt.savefig(f'brickwork-complexity-{qbits}-line.png')
