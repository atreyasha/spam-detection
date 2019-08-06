#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pickle
import warnings
import argparse
import numpy as np
from glob import glob
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score

##############################
# define key functions
##############################

def optimalThreshold(pr_list):
    pr_list = np.asarray(pr_list)
    pr_list = pr_list[np.where(pr_list[:,2] != 0)]
    pr_list = pr_list[np.where(pr_list[:,2] != 1)]
    loc = np.where(pr_list[:,2]>=0.998)[0]
    if len(loc) == 0:
        return pr_list[np.argmax(pr_list[:,2])][0]
    else:
        filtered = pr_list[loc]
        return filtered[np.argmax(filtered[:,1])][0]

def thresholdRNN(pickle_file):
    y_test = np.load("./data/rnn/y_test.npy")
    # main processing
    probs = np.load(glob("./pickles/"+pickle_file+"/prob*")[0])
    thresholds = np.linspace(0.00001,1,50)
    pr_list = []
    for value in thresholds:
        out = np.where(probs >= value, 1, 0)
        res = precision_recall_fscore_support(y_test,out)[:2]
        pr_list.append([value,res[0][0],res[1][0]])
    with open("./pickles/"+pickle_file+"/precision_recall_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(i for i in ["threshold","precision","recall"])
        writer.writerows(pr_list)
    optimal = optimalThreshold(pr_list)
    roc = roc_auc_score(y_test,probs)
    out = np.where(probs >= 0.5, 1, 0)
    with open("./pickles/"+pickle_file+"/classification_report_test.txt", "w") as f:
        f.write("ROC: "+str(roc)+"\n")
        f.write(classification_report(y_test,out,digits=4))
    out = np.where(probs >= optimal, 1, 0)
    with open("./pickles/"+pickle_file+"/classification_report_test_optimal.txt", "w") as f:
        f.write("Optimal threshold: "+str(optimal)+"\n")
        f.write(classification_report(y_test,out,digits=4))

def thresholdSVM(pickle_file):
    y_test = np.load("./data/svm/y_test.npy")
    # main processing
    probs = np.load(glob("./pickles/"+pickle_file+"/prob*")[0])
    mean = np.mean(probs)
    std = np.std(probs)
    thresholds = np.linspace(mean-std,mean+std,50)
    pr_list = []
    for value in thresholds:
        out = np.where(probs >= value, 1, -1)
        res = precision_recall_fscore_support(y_test,out)[:2]
        pr_list.append([value,res[0][0],res[1][0]])
    with open("./pickles/"+pickle_file+"/precision_recall_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(i for i in ["threshold","precision","recall"])
        writer.writerows(pr_list)
    optimal = optimalThreshold(pr_list)
    roc = roc_auc_score(y_test,probs)
    out = np.where(probs >= 0, 1, -1)
    with open("./pickles/"+pickle_file+"/classification_report_test.txt", "w") as f:
        f.write("ROC: "+str(roc)+"\n")
        f.write(classification_report(y_test,out,digits=4))
    out = np.where(probs >= optimal, 1, -1)
    with open("./pickles/"+pickle_file+"/classification_report_test_optimal.txt", "w") as f:
        f.write("Optimal threshold: "+str(optimal)+"\n")
        f.write(classification_report(y_test,out,digits=4))
        
def importanceSVM(pickle_file):
    with open("./data/svm/words/integer_index_tokens.pickle","rb") as f:
        word_dict = pickle.load(f)
    full_name = glob("./pickles/"+pickle_file+"/best*")[0]
    with open(full_name,"rb") as f:
        model = pickle.load(f)
    word_dict = {v:k for k,v in word_dict.items()}
    pos_ind = np.where(model.coef_[0] > 0)[0]
    neg_ind = np.where(model.coef_[0] < 0)[0]
    spam_top = np.abs(model.coef_[0][pos_ind][np.argsort(model.coef_[0][pos_ind])][-10:])
    ham_top = np.abs(model.coef_[0][neg_ind][np.argsort(model.coef_[0][neg_ind])][:10])
    pos_ind = pos_ind[np.argsort(model.coef_[0][pos_ind])[-10:]]
    neg_ind = neg_ind[np.argsort(model.coef_[0][neg_ind])[:10]]
    spam_top_words = [word_dict[el] for el in pos_ind]
    ham_top_words = [word_dict[el] for el in neg_ind]
    with open("./pickles/"+pickle_file+"/spam_top_words.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(i for i in ["word","coefficient"])
        writer.writerows(zip(ham_top_words,ham_top))
    with open("./pickles/"+pickle_file+"/ham_top_words.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(i for i in ["word","coefficient"])
        writer.writerows(zip(spam_top_words,spam_top))
    
# roc = roc_auc_score(y_blind,out)
# out = np.where(out >= 0.5, 1, 0)
# with open("./pickles/"+pickle_file+"/classification_report_blind.txt", "w") as f:
#     f.write("ROC: "+str(roc)+"\n")
#     f.write(classification_report(y_blind,out,digits=4))    

# roc = roc_auc_score(y_svm_blind,out)
# with open("./pickles/"+pickle_file+"/classification_report_blind.txt", "w") as f:
#     f.write("ROC: "+str(roc)+"\n")
#     f.write(classification_report(y_svm_blind,model.predict(X_blind_words),digits=4))

##############################
# main command call
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding-tokens", type=int, default = 500,
                        help="maximum length of email padding for tokens <default:500>")
    parser.add_argument("--padding-char", type=int, default = 1000,
                        help="maximum length of email padding for characters <default:1000>")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str,
                               help="pickle directory name for stored model, or input 'all' to run on all models", 
                               required=True)
    args = parser.parse_args()
    files = glob("./pickles/20*")
    # run evaluations based on pickle input
    if args.pickle != "all":
        if "rnn" in args.pickle:
            thresholdRNN(args.pickle)
            warnings.warn("combined plots only possible with '-p all' option")
        elif "svm" in args.pickle:
            thresholdSVM(args.pickle)
            warnings.warn("combined plots only possible with '-p all' option")
            if "linear" in args.pickle:
                importanceSVM(args.pickle)
                os.system("Rscript plot_models.R --type svm")
    else:
        for file in files:
            filename = os.path.basename(file)
            if "rnn" in file:
                thresholdRNN(filename)
            elif "svm" in file:
                thresholdSVM(filename)
                if "linear" in file:
                    importanceSVM(filename)
                    os.system("Rscript plot_models.R --type svm")
        os.system("Rscript plot_models.R --type combined")