import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import BayesianGaussianMixture
import sys

start = 1
end = 200

title = f'BGM_{start}_{end}.csv'
print(title)

data = pd.read_csv('../data/data.csv')

best_data =[
'f_07','f_08', 'f_09', 'f_10',
'f_11', 'f_12', 'f_13', 'f_22',
'f_23', 'f_24', 'f_25','f_26',
'f_27', 'f_28']

pt = PowerTransformer()
data_scaled = pt.fit_transform(data[best_data])
data_scaled = pd.DataFrame(data_scaled, columns = best_data)

values = [0,1,2,3,4,5,6]
pred_test = pd.DataFrame(np.zeros((data_scaled.shape[0],7)), columns = values)

for seed in tqdm(range(start,end)):
    
    df = pd.DataFrame(index = data.index)
    gmm = BayesianGaussianMixture(
            n_components=7,
            random_state = seed,
            tol = 0.01,
            covariance_type = 'full',
            max_iter = 100,
            n_init=3
          )
    
    # fitting and probability prediction
    gmm.fit(data_scaled)
    pred_seed = gmm.predict_proba(data_scaled) # predict_proba for probabilities
    
    # the clusters prediction for the current seed :
    MAX = np.argmax(pred_seed, axis=1)
    df[f'pred_{seed}'] = MAX
    
    # Sort of the prediction by same value of cluster (for addition of every seed)
    pred_keys = df[f'pred_{seed}'].value_counts().index.tolist()
    pred_dict = dict(zip(pred_keys, values))
    df[f'pred_{seed}'] = df[f'pred_{seed}'].map(pred_dict)

    pred_new = pd.DataFrame(pred_seed).rename(columns = pred_dict)
    pred_new = pred_new.reindex(sorted(pred_new.columns), axis=1)
    pred_test += pred_new # Soft voting by probabiliy addition

predictions = np.argmax(np.array(pred_test), axis=1)

submission = pd.read_csv("../data/sample_submission.csv")
submission["Predicted"] = predictions
submission.to_csv(title,index = False)