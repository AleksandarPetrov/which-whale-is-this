from modelEvaluator import modelEvaluator
import sys
## Hyperopt ##
from hyperopt import fmin, tpe, hp, Trials
import numpy as np

architecture_space = {'n_conv_layers': hp.choice(label='n_conv_layers', options=[2,3,4]),
                      'initial_kernel_size': hp.choice(label='initial_kernel_size', options=[7,9,11,13]),
                      'initial_number_filters': hp.choice(label='initial_number_filters', options=[8, 16, 32, 64])
                      }

trials = Trials()
best = fmin(fn = modelEvaluator,
            space = architecture_space,
            algo=tpe.suggest,
            max_evals= 100,
            trials=trials) #lambda params: modelEvaluator(params,  trainEpochs = 25, parent_dir = sys.argv[1], iterations = 5)
print('best: ')
print(best)
np.save('best_arch_params.npy', best)

