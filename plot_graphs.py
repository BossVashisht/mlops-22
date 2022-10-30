from sklearn import datasets, svm, metrics, tree
import pdb
import statistics

def test_labels_counts(predicted_dev):

    num_labels  = 0

    for i in range(len(predicted_dev)-1):
        if predicted_dev[i] != predicted_dev[i+1]:
            num_labels += 1

    #print(num_labels)

    return num_labels == len(label) 

def test_bias(predicted_dev):

    num_labels  = 0

    for i in range(len(predicted_dev)-1):
        if predicted_dev[i] != predicted_dev[i+1]:
            num_labels += 1

    #print(num_labels)
    
    return num_labels != 0 



from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

################################ this is new

max_depth_list = [2, 10, 20, 50, 100]   

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

##################################

digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score

results = {}
for n in range(5):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    models = {
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }

    clf = models['svm']
    print("[{}] Running hyper param tuning for {}".format(n,'svm'))
    actual_model_path = tune_and_save(
        clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb['svm'], model_path=None
    )
    best_model = load(actual_model_path)
    predicted = best_model.predict(x_test)
    print("number of labels predicted matches total number of labels = ",test_labels_counts(predicted))
    print("model didn't learnt just single label = ",test_bias(predicted))
    

    if not 'svm' in results:
        results['svm']=[]    

    results['svm'].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
    print("  ")


    clf = models['decision_tree']
    print("[{}] Running hyper param tuning for {}".format(n,'decision_tree'))
    actual_model_path = tune_and_save(
        clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb['decision_tree'], model_path=None
    )
    best_model = load(actual_model_path)
    predicted = best_model.predict(x_test)
    print("number of labels predicted matches total number of labels = ",test_labels_counts(predicted))
    print("model didn't overfit to single label = ",test_bias(predicted))
    if not 'decision_tree' in results:
        results['decision_tree']=[]    

    results['decision_tree'].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
    print("   ")


print(results)
#print("svm list is ")
#print(results['svm'])

total1 = []
for i in range(len(results['svm'])):
    total1.append(results['svm'][i]['accuracy_score'])

print("mean of svm is : ",statistics.mean(total1))
print("standard deviation of svm is : ", statistics.pstdev(total1))

#print("dicision list is ")
#print(results['decision_tree'])
total2 = []
for i in range(len(results['decision_tree'])):
    total2.append(results['decision_tree'][i]['accuracy_score'])

print("mean of decision_tree :",statistics.mean(total1))
print("standard deviation of decision_tree is : ", statistics.pstdev(total2))