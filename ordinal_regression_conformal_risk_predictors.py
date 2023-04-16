from abc import ABC, abstractmethod
import numpy as np

# Base class for Ordinal Regression Ordinal Risk Predictor;
# a typical calling order is: find_lambda -> calc_loss -> get_prediction_set_bounds;
class OrdinalRegCRPredictor:
    
    # calculate the bound [l, u] for the optimal prediction set
    # such that sum of sy between the bound is greater than the lambda value.
    @abstractmethod
    def get_prediction_set_bounds(self, fyx, lambda_val):
        pass
    
    # calculate the incurred loss for a specific record
    @abstractmethod
    def calc_loss(self, fyx, y, lambda_val):
        pass
    
    # find the optimal value of lambda for a dataset 
    # such that the risk on this dataset is controlled by alpha
    @abstractmethod
    def find_lambda(self, fyxs, ys, alpha):
        pass
    
    # run prediction for a batch of records
    @abstractmethod
    def run_predictions(self, fyxs, ys, lambda_val):
        pass

class WeightedCRPredictor(OrdinalRegCRPredictor):
    
    # initialize the weights
    # normalize the weights so that the maximal value is 1
    def __init__(self, hy):
        max_hy = np.max(hy)
        self.hy = hy / max_hy 
        self.num_classes = hy.size    
        
    # get the prediction set for given fyx and lambda_val
    # this implementation is greedy
    # it starts at the index with the largest sy=hy*fyx value
    # then gradually extends the boundary until the risk meets the required limit
    # after that, it tries to shrinks the boundary to squeeze the ones with zero risks
    # the last step is needed to avoid producing a too large prediction set
    def get_prediction_set_bounds(self, fyx, lambda_val):
        sy = self.hy * fyx
        
        #b_val = sum(sy) ## conditional
        b_val = 1 ## marginal
        threshold = b_val - lambda_val
    
        index_max = np.argmax(sy)
        s = sy[index_max]
        l, u = index_max, index_max
        while s < threshold:
            if l - 1 >= 0 and u + 1 <= self.num_classes -1:
                if sy[l - 1] >= sy[u + 1]:
                    l = l - 1
                    s = s + sy[l]
                else:
                    u = u + 1
                    s = s + sy[u]
            elif l - 1 >= 0:
                l = l - 1
                s = s + sy[l]
            elif u + 1 <= self.num_classes - 1:
                u = u + 1
                s = s + sy[u]
            else:
                break
          
        while sy[l] == 0 and l < u:
            l = l + 1
        while sy[u] == 0 and l < u:
            u = u - 1  
            
        return l, u

    # calculate the incurred loss for a specific record
    # hy:         weights of different labels;
    # fyx:        model scores of different labels;
    # y:          true label;
    # lambda_val: a proposed risk bound;
    def calc_loss(self, fyx, y, lambda_val):
        lower_bound, upper_bound = self.get_prediction_set_bounds(fyx, lambda_val)
        if (y >= lower_bound) and (y <= upper_bound):
            return 0.0
        else:
            return self.hy[int(y)]    
        
    # find the optimal value of lambda such that the risk is controlled by alpha
    # fyxs:  the matrix of model scores, where each row is for one record, each column is for one class;
    # ys:    the array of true labels;
    # alpha: risk bound, value between 0 and 1;
    def find_lambda(self, fyxs, ys, alpha):
        (num_records, num_classes) = fyxs.shape
        b_val = 1 #max(hy)
        threshold = (num_records + 1.0) / num_records * alpha - b_val / num_records

        cur_lambda = 0.5
        delta = 0.5
        delta_threshold = 0.0005
        while delta > delta_threshold:
            total_r = 0.0
            for i in range(num_records):
                total_r = total_r + self.calc_loss(fyxs[i, :], ys[i], cur_lambda)
            avg_r = total_r / num_records
            if avg_r > threshold:
                cur_lambda = cur_lambda - delta / 2
            elif avg_r < threshold:
                cur_lambda = cur_lambda + delta / 2
            else:
                break
            delta = delta / 2
        return cur_lambda
    
    def run_predictions(self, fyxs, ys, lambda_val):
        num_records = fyxs.shape[0]
        prediction_sets = []
        losses = []
        for i in range(num_records):
            lower_bound, upper_bound = self.get_prediction_set_bounds(fyxs[i, :], lambda_val)
            prediction_sets.append((lower_bound, upper_bound))
            label = int(ys[i])
            if (label >= lower_bound) and (label <= upper_bound):
                losses.append(0.0)
            else:
                losses.append(self.hy[label])
        return prediction_sets, losses    
    
class DivergenceCRPredictor(OrdinalRegCRPredictor):
    
    # Given fyx, calculate the cumulative head scores
    # head_j = sum_{i=0}^{j}(fyx_i) / (K-1)
    def get_head_scores(self, fyx):
        num_classes = fyx.size
        head_scores = np.zeros(num_classes)
        prev = 0
        for i in range(num_classes):
            head_scores[i] = prev + fyx[i]
            prev = head_scores[i]
        return [v / (num_classes - 1) for v in head_scores]


    # Given fyx, calculate the cumulative tail scores
    # tail_j = sum_{i=j}^{K-1}(fyx_i) / (K-1)
    def get_tail_scores(self, fyx):
        num_classes = fyx.size
        tail_scores = np.zeros(num_classes)
        prev = 0
        for i in range(num_classes - 1, -1, -1):
            tail_scores[i] = prev + fyx[i]
            prev = tail_scores[i]
        return [v / (num_classes - 1) for v in tail_scores]
    
    # Given fyx, as well as a threshold,
    # find out the optimal bound [y_l, y_u] of the prediction set,
    # such that the total risk is less than the threshold.
    # this implementation is greedy.
    def get_prediction_set_bounds(self, fyx, lambda_val):
        num_classes = fyx.size
        head_scores = self.get_head_scores(fyx)
        tail_scores = self.get_tail_scores(fyx)

        l, u = np.argmax(fyx), np.argmax(fyx)
        sums = np.zeros(num_classes)
        steps = [None] * num_classes
        steps[0] = (l, u)
        s = 0
        for i in range(num_classes - 1):
            if l > 0 and u < num_classes - 1:
                if head_scores[l - 1] >= tail_scores[u + 1]:
                    s = s + head_scores[l - 1]
                    l = l - 1
                else:
                    s = s + tail_scores[u + 1]
                    u = u + 1
            elif l == 0:
                s = s + tail_scores[u + 1]
                u = u + 1
            else:
                s = s + head_scores[l - 1]
                l = l - 1
            sums[i + 1] = s    
            steps[i + 1] = (l, u)

        for i in range(num_classes):
            if sums[num_classes - 1] - sums[i] <= lambda_val:
                l, u = steps[i][0], steps[i][1]
                break

        return l, u
    
    # calculate the incurred loss for a specific record
    # fyx:        model scores of different labels;
    # y:          true label;
    # lambda_val: a proposed risk bound;
    def calc_loss(self, fyx, y, lambda_val):
        num_classes = fyx.size
        lower_bound, upper_bound = self.get_prediction_set_bounds(fyx, lambda_val)
        if y < lower_bound:
            return (lower_bound - y) / (num_classes - 1)
        elif y > upper_bound:
            return (y - upper_bound) / (num_classes - 1)
        else:
            return 0.0
    
    # find the optimal value of lambda such that the risk is controlled by alpha
    # fyxs:  the matrix of model scores, where each row is for one record, each column is for one class;
    # ys:    the array of true labels;
    # alpha: risk bound, value between 0 and 1;
    def find_lambda(self, fyxs, ys, alpha):
        (num_records, num_classes) = fyxs.shape
        b_val = 1
        threshold = (num_records + 1.0) / num_records * alpha - b_val / num_records

        cur_lambda = 0.5
        delta = 0.5
        delta_threshold = 0.0001
        while delta > delta_threshold:
            total_r = 0.0
            for i in range(num_records):
                total_r = total_r + self.calc_loss(fyxs[i, :], ys[i], cur_lambda)
            avg_r = total_r / num_records
            if avg_r > threshold:
                cur_lambda = cur_lambda - delta / 2
            elif avg_r < threshold:
                cur_lambda = cur_lambda + delta / 2
            else:
                break
            delta = delta / 2
        return cur_lambda
    
    def run_predictions(self, fyxs, ys, lambda_val):
        num_records = fyxs.shape[0]
        prediction_sets = []
        losses = []
        for i in range(num_records):
            lower_bound, upper_bound = self.get_prediction_set_bounds(fyxs[i, :], lambda_val)
            prediction_sets.append((lower_bound, upper_bound))
            label = int(ys[i])
            if label < lower_bound:
                losses.append((lower_bound - label) / (num_classes - 1))
            elif label > upper_bound:
                losses.append((label - upper_bound) / (num_classes - 1))
            else:
                losses.append(0.0)
        return prediction_sets, losses    