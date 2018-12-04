"""
Toy train script
Usage:
    benchmark_script.py fit <experiment_dir> 
    benchmark_script.py best <experiment_dir> 
    benchmark_script.py predict <experiment_dir> 
    benchmark_script.py -h | --help
    benchmark_script.py --version

Common options:
    <experiment_dir>    Set experiment root directory. This script expects
                        a configuration file called "config.yml" to live
                        in this directory. See "Configuration file"
                        section below for more details.
Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml that describes the pipeline.
    ................... <experiment_dir>/config.yml ...................
    sequence_generator:
       name: ToyIterator
       params:
          duration_distribution: 
              name: PoissonDistribution
              params:
                  lam: 5
          num_distribution: 
              name: UniformDistribution
              params:
                  start: 2
          order_distribution: 
              name: MarkovDistribution
          weight_distribution: 
              name: UniformDistribution
          embedding_generator: 
              name: GaussEGenerator
              params:
                  n_features: 2
                  center_box: (-10.0, 10.0)
                  cluster_std: 0.5
          sequence_generator: 
              name: SGenerator
          duration: 30
          max_spk: 10
          batch_size: 32
          num_per_epoch: 200

    Clustering:
       name: ap
       params:
          db_path: af.db
          metric: euclidean

    ...................................................................
"fit" mode:
   Tune the hyper-parameters.

"best" mode:
    Return the best hyper-parameters.

"predict" mode:
    Do prediction.

"""

import sys
sys.path.append("/people/yin/projects/end2end_diarization/seq2seq-diarization") 
import os
import yaml
import numpy as np
from docopt import docopt
from seq2seq.generator.toy_iterator import ToyIterator
from model import ClusterOptimizer



if __name__ == '__main__':
    arguments = docopt(__doc__, version='benchmark')
    experiment_dir = arguments['<experiment_dir>']
    with open(experiment_dir+"config.yml", 'r') as stream:
        config = yaml.load(stream)

    num_distribution_name = config['sequence_generator']['params']['num_distribution']['name']
    num_distribution_params = config['sequence_generator']['params']['num_distribution']['params']
    num_distributions =  __import__('distribution.num_distribution',fromlist=[num_distribution_name])
    num_distribution = getattr(num_distributions, num_distribution_name)(**num_distribution_params)

    weight_distribution_name = config['sequence_generator']['params']['weight_distribution']['name']
    weight_distributions =  __import__('distribution.weight_distribution',fromlist=[weight_distribution_name])
    weight_distribution = getattr(weight_distributions, weight_distribution_name)()

    order_distribution_name = config['sequence_generator']['params']['order_distribution']['name']
    order_distributions =  __import__('distribution.order_distribution',fromlist=[order_distribution_name])
    order_distribution = getattr(order_distributions, order_distribution_name)()

    duration_distribution_name = config['sequence_generator']['params']['duration_distribution']['name']
    duration_distribution_params = config['sequence_generator']['params']['duration_distribution']['params']
    duration_distributions =  __import__('distribution.duration_distribution',fromlist=[duration_distribution_name])
    duration_distribution = getattr(duration_distributions, duration_distribution_name)(**duration_distribution_params)

    embedding_generator_name = config['sequence_generator']['params']['embedding_generator']['name']
    embedding_generator_params = config['sequence_generator']['params']['embedding_generator']['params']
    embedding_generators =  __import__('embedding_generator',fromlist=[embedding_generator_name])
    embedding_generator = getattr(embedding_generators, embedding_generator_name)(**embedding_generator_params)

    duration = config['sequence_generator']['params']['duration']
    max_spk = config['sequence_generator']['params']['max_spk']
    batch_size = config['sequence_generator']['params']['batch_size']
    num_per_epoch = config['sequence_generator']['params']['num_per_epoch']

    generator = ToyIterator(duration, num_distribution, weight_distribution, 
                    duration_distribution, order_distribution, embedding_generator, 
                     max_spk = max_spk,num_per_epoch=num_per_epoch,
                    batch_size=batch_size, device='cuda')


    X_test, y_test = [], []
    itr = iter(generator)
    for i in range(10):
        batch = next(itr)
        X_test.extend(batch.src.detach().cpu().numpy())
        y_test.extend(batch.tgt.max(-1)[1].detach().cpu().numpy())
    data = list(zip(X_test, y_test))

    cluster_name = config['Clustering']['name']
    opt = ClusterOptimizer(cluster_name, **config['Clustering']['params'])
    if arguments['fit']:
        opt.fit(data)
    if arguments['best']:
        print(opt.optimizer.best_params)
    if arguments['predict']:
        np.save(experiment_dir+'prediction',np.array(opt.predict(data)))
        print(opt.predict(data))










