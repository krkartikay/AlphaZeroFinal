import random
import numpy
from matplotlib import pyplot as plt

fail_probs = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]

st = numpy.zeros((len(fail_probs), 20))

for exp_num, fail_prob in enumerate(fail_probs):
    for sim_num in range(10000):
        for sim_step in range(20):
            is_fail = (random.random() < fail_prob)
            if is_fail:
                st[exp_num][sim_step] += 1
                break

print(st[-3:])

st = st.T
plt.imshow(st, origin='lower', cmap='Reds')
plt.savefig('stats.png')