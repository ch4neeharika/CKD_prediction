import algolib as algo
from scipy.stats import randint

raw_train_data_path = './raw_data/chronic_kidney_disease_full.csv'
raw_test_data_path = './raw_data/test.csv'

train_set_path = './clean_data/clean_data_to_train.csv'
test_set_path = './clean_data/clean_data_to_test.csv'

# list of implemented algorithms
algo_list = ['dtc','linsvc','svc','mlp','knn','gaus','lda','logreg']

# choose sample size to train and test on
sample_count = None

# 1. process dataset and save it
c = ['sg','sc','sod','hemo','class']
t = ['sg','sc','sod','hemo']

# algo.extractTrainData(path=raw_train_data_path, savepath=train_set_path, columns=c, row_count=sample_count)
# algo.extractTestData(path=raw_test_data_path, savepath=test_set_path, columns=t, row_count=sample_count)

# #2. analize and clean data
algo.analyze(path=raw_train_data_path, sample_count=sample_count, save=False)

# 4. run all algorithms to find best fit for the data
algo.run_multiple(path=train_set_path, algos=algo_list, sample_count=sample_count, exportBest=False)

# 5. set your hyper parameters and choose algorithms to test
dtc_hyperparameters = {"criterion":["gini","entropy"],"max_depth":[1,5,10],"min_samples_split":[5,10],
                    "min_impurity_decrease":[1e-3,1e-4,1e-5,1e-6,1e-7], "random_state":[0]}

linsvc_hyperparameters = {"max_iter":[5000], "dual":[True,False]}

svc_hyperparameters = {"kernel":["rbf","sigmoid"], "C":[0.5,0.6,0.7,0.8,0.9,1.0], "tol":[1e-3,1e-4,1e-5,1e-6,1e-7]}

mlp_hyperparameters = {"hidden_layer_sizes":[(100,),(8,8,8),(3,3,3),(5,5,5)], "solver":['lbfgs', 'sgd', 'adam'], 
                    "alpha":[0.0001,0.001,0.01],"learning_rate":["constant"], "learning_rate_init":[0.00001,0.0001,0.001], 
                    "max_iter":[1000,2000,5000], "tol":[0.00001,0.0001,0.001], "warm_start":[False, True]}

knn_hyperparameters = {"n_neighbors":[2,3,4,5,6,7,8,9,10,11,12,13,14,15], "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'], "leaf_size":randint(10,100)}

gaus_hyperparameters = {"var_smoothing":[1e-9,1e-8,1e-7,1e-6,1e-5]}

lda_hyperparameters = {"solver":['svd', 'lsqr','eigen'], "store_covariance":[False,True], "tol":[0.01,0.001,0.00001,0.0000001]}

logreg_hyperparameters = {"C":[0.5,0.6,0.7,0.8,0.9,1.0], "dual":[False],"max_iter":[1000,2000,5000], "multi_class":['ovr'],
                        "n_jobs":[4,5,6,7,8,9], "random_state":[0], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
                        "tol":[1e-3,1e-4,1e-5,1e-6,1e-7], "warm_start":[False, True]}

hyperparameters = {}
hyperparameters.update({algo_list[0]:dtc_hyperparameters})
hyperparameters.update({algo_list[1]:linsvc_hyperparameters})
hyperparameters.update({algo_list[2]:svc_hyperparameters})
hyperparameters.update({algo_list[3]:mlp_hyperparameters})
hyperparameters.update({algo_list[4]:knn_hyperparameters})
hyperparameters.update({algo_list[5]:gaus_hyperparameters})
hyperparameters.update({algo_list[6]:lda_hyperparameters})
hyperparameters.update({algo_list[7]:logreg_hyperparameters})

algo.testHypersOnAlgo(path=train_set_path, algo=['mpc'], hparameters=hyperparameters, samples=sample_count, folds=5, save_best=True, search='grid')

# 6. run single algorithm from best test results
training_accuracy, test_accuracy = algo.testAlgo(path=train_set_path, algo='dtc', samples=sample_count, export=True, log=True, standard_scaler=True)
print("Classifier score on train set: {0} %".format(training_accuracy*100))
print("Classifier accuracy on test set: {0} %".format(test_accuracy*100)+ "\n")

# 7. predict on saved model
algo.predict_on_model('./models/dtc_classifier.pkl',[1.01,13.3,124.0,3.1])