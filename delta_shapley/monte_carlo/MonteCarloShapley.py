from torch.utils.data import Subset, DataLoader, SequentialSampler
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import delta_shapley.data as data
import delta_shapley.models as models
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch import optim
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import time
from torch.multiprocessing import Pool, set_start_method
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_start_method('spawn', force=True)

class Shapley():
    
    def __init__(self, trainset, testset, L, beta, c, sup, params, train_func, val_func):
        self.trainset = trainset
        self.testset = testset
        self.L = L
        self.beta = beta
        self.c = c
        self.shapley = 0
        self.sup = sup
        self.n = params.datasize
        self.De = params.datasize
        self.params = params
        self.train_func = train_func
        self.val_func = val_func
        
    def compute(self, indices, datapoint_idx, num_classes, params):
        """
        Compute the marginal contribution of a datapoint to a sample
        Args:
            indices: the indices of the train dataset to be used as a sample
            datapoint: the index of the evaluated datapoint
        """
        
        print(f"Training with {len(indices)} data point.")
        # compute a random point to insert the differing datapoint
        random_idx = np.random.randint(0, len(indices))

        # model without datapoint
        sample = Subset(self.trainset, list(indices))
        sampler = SequentialSampler(sample)
        trainloader = DataLoader(sample, batch_size=1, sampler=sampler)
        
        # insert in differential datapoint
        if random_idx == 0:
            indices_incl_datapoint = np.concatenate(([datapoint_idx], indices))
        else:
            indices_incl_datapoint = np.concatenate((np.concatenate((indices[:random_idx], [datapoint_idx])), indices[random_idx:]))
        # model with datapoint 
        sample_datapoint = Subset(self.trainset, list(indices_incl_datapoint))
        sampler_datapoint = SequentialSampler(sample_datapoint)
        trainloader_datapoint = DataLoader(sample_datapoint, batch_size=1, sampler=sampler_datapoint)
        
        with Pool(2) as pool:
            res1 = pool.apply_async(self.train_func, (models.return_model,num_classes, trainloader, params))
            res2 = pool.apply_async(self.train_func, (models.return_model,num_classes, trainloader_datapoint, params))
            pool.close()
            pool.join()
            
        trained = res1.get()
        trained_datapoint = res2.get()
        
        testloader = DataLoader(self.testset, batch_size=params.batch_size, shuffle=False)
        
        with Pool(2) as pool:
            res3 = pool.apply_async(self.val_func, (trained, testloader, params))
            res4 = pool.apply_async(self.val_func, (trained_datapoint, testloader, params))
            pool.close()
            pool.join()
            
        val = res3.get()
        val_datapoint = res4.get()
        return val_datapoint - val
    

class MonteCarloShapley(Shapley):
    """
    Take the algorithm from Ghorbani (Data Shapley) and adapted it
    """

    def __init__(self, trainset, testset, L, beta, c, a, b, sup, num_classes, params,train_func, val_func):
        """
        Args:
            trainset: the whole dataset from which samples are taken
            testset: the validation set
            datapoint: datapoint to be evaluated (index)
            L: Lipschitz constant
            beta: beta-smoothness constant
            c: learning rate at step 1, decaying with c/t
            a: the "a" parameter in the (a,b)-bound in the Shapley value estimation
            b: the "b" parameter in the (a,b)-bound in the Shapley value estimation
            sup: the supremum of the loss function
            num_classes:
            params
        """
        super().__init__(trainset, testset, L, beta, c, sup, params, train_func, val_func)
        self.trainset = trainset
        self.testset = testset
        self.L = L
        self.beta = beta
        self.c = c
        self.shapley = 0
        self.a = a
        self.b = b
        self.sup = sup
        self.n = params.datasize
        self.De = params.datasize
        self.num_classes = num_classes
        self.params = params
        self.SVs = []
        self.samples = []
        self.SVdf = None
        self.max_iter = 1000

    def run(self, datapoints, params):
        """
        Args:
            datapoint: the index of the datapoint in the trainset to be evaluated
            return: the approximate Shapley value
        """
        self.SVdf = pd.DataFrame(columns = sum([[str(i) + "_SV",str(i) + "_time",str(i) + "_layer"] for i in datapoints],[]))
        shapley_values = np.zeros(len(datapoints))
        iter = 1
        while (not self.check_convergence_rolling(iter, datapoints)) and iter <= self.max_iter:
            row_iteration = dict()
            if iter % 1 == 0:
                print("Monte Carlo running in iteration {}".format(iter))
            for i in tqdm.tqdm(range(len(datapoints))):
                datapoint = datapoints[i]
                if len(self.SVdf) > 0:
                    est_shapley = self.SVdf.iloc[-1][str(datapoint)+"_SV"]
                else:
                    est_shapley = 0

                permutation = np.arange(self.n)
                np.random.shuffle(permutation)
                # see https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                datapoint_index = np.where(permutation == datapoint)[0][0]

                # prevent the evaluated datapoint from being the first in the permutation
                while datapoint_index == 0:
                    np.random.shuffle(permutation)
                    # see https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/
                    datapoint_index = np.where(permutation == datapoint)[0][0]

                indices = permutation[:datapoint_index]
                self.samples = datapoint_index
                time_now = time.time()
                v = self.compute(indices, datapoint, self.num_classes, self.params)
                elapsed_time = time.time() - time_now
                est_shapley = est_shapley * ((iter - 1)/iter) + (v/iter)
                shapley_values[i] = est_shapley
                row_iteration[str(datapoint) + "_SV"] = est_shapley
                row_iteration[str(datapoint) + "_time"] = elapsed_time
                row_iteration[str(datapoint) + "_layer"] = datapoint_index
            row_iteration_df = pd.DataFrame([row_iteration])
            self.SVdf = pd.concat([self.SVdf, row_iteration_df], ignore_index=True)
            iter +=1
            if os.path.exists(params.save_dir + "/MonteCarlo_FM_CNN_n100_2.csv"):
                os.remove(params.save_dir + "/MonteCarlo_FM_CNN_n100_2.csv")
            self.SVdf.to_csv(params.save_dir+"/MonteCarlo_FM_CNN_n100_2.csv")
        return shapley_values

    def check_convergence_rolling(self, iteration, datapoints):
        if iteration < 102:
            return False
        else:
            current_row = self.SVdf.iloc[-1]
            old_row = self.SVdf.iloc[-101]
            small_deviation = [self.check_deviation((old_row[str(i)+"_SV"], current_row[str(i)+"_SV"])) for i in datapoints]
            if iteration % 100 == 0:
                print("Iteration {}, current convergence: {}".format(iteration, sum(small_deviation)/len(datapoints)))
            if (sum(small_deviation)/(len(datapoints))) < 0.05:
                return True
            else:
                return False

    def check_deviation(self, vals):
        old = vals[0]
        new = vals[1]
        # check whether either is 0 to avoid div by 0, return False if either is 0
        if old == 0 or new == 0:
            return 1e6
        ratio = abs(new - old) /abs(new)
        return ratio
    
