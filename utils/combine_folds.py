import os
import argparse
import json
from compute_combined_mean_sdev import comb_sdev
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='Select directory to combine folds')
args = parser.parse_args()

if __name__ == "__main__":
    if args.path is not None:
        PATH = args.path
    else:
        raise ValueError("please specify a path")
    file_list = [filename for dirpath, _, filenames in os.walk(PATH) for filename in filenames if filename.endswith('.txt')] 
    config = [os.path.join(dirpath, filenames[0]) for dirpath, _, filenames in os.walk(PATH) for filename in filenames if filename.endswith('config.json')]
    file_list = np.unique(file_list)
    print(file_list)
    print(config)
    per_kp = [x for x in file_list if x not in ['test_res.txt', 'train_loss.txt', 'val_loss.txt']]
    for cfg in config:
        with open(cfg) as params:
            try:
                exp_config = json.load(params)
            except json.decoder.JSONDecodeError as err:
                print("something went wrong with decoding this json: ", cfg)
                break
        dataname = str(exp_config["data_config"]["dataset"])
        folds = int(exp_config["data_config"]["n_folds"])
        cfg_path = "/".join(cfg.split("/")[:-1]+["run_01/"])
        
        # combine per kp:
        for kp in per_kp:
            n = int(exp_config["training_params"]["report_interval"]) * int(exp_config["training_params"]["num_gpu"]) if "train" in str(kp) else int(exp_config["training_params"]["validation_steps"]) * int(exp_config["training_params"]["num_gpu"])
            fold_list = []
            for i in range(folds):
                with open(os.path.join(cfg_path, "fold_0"+str(i)+"/"+kp), 'r') as results:
                    data = np.array(results.read().split('\n'))[:-1]
                data = [np.array(i.split(","), dtype=np.float) for i in data]
                if dataname == "cephal":
                    data = [x[:-1] for x in data] # drop last "empty" landmark
                fold_list.append(zip(data[0::2], data[1::2]))
            prepped = np.transpose(np.array(list(zip(*fold_list))), axes=(0,3,2,1))
            
            res = np.array(list(map(lambda x: list(map(lambda y: comb_sdev(y[0], y[1], n)[1:], x)), prepped))) # apply comb_sdev and only return mean and sdev
            avg_res = np.array(list(map(lambda x: comb_sdev(x[0], x[1], n*folds)[1:], np.transpose(res, (0, 2, 1)))))
            
            np.savetxt(os.path.join(cfg_path, "mean_"+kp),res[:,:,0], fmt='%.3f', delimiter=',')
            np.savetxt(os.path.join(cfg_path, "sdev_"+kp),res[:,:,1], fmt='%.3f', delimiter=',')
            np.savetxt(os.path.join(cfg_path, "avg_mean_"+kp),avg_res[:, 0], fmt='%.3f', delimiter=',')
            np.savetxt(os.path.join(cfg_path, "avg_sdev_"+kp),avg_res[:, 1], fmt='%.3f', delimiter=',')

        # combine loss
        for loss in ['train_loss.txt', 'val_loss.txt']:
            n = int(exp_config["training_params"]["report_interval"]) * int(exp_config["training_params"]["num_gpu"]) if "train" in loss else int(exp_config["training_params"]["validation_steps"]) * int(exp_config["training_params"]["num_gpu"])
            fold_list = []
            for i in range(folds):
                with open(os.path.join(cfg_path, "fold_0"+str(i)+"/"+loss), 'r') as results:
                    data = np.array(results.read().split('\n'))[:-1].astype(np.float)
                fold_list.append(zip(data[0::2], data[1::2]))
            prepped = np.transpose(np.array(list(zip(*fold_list))), (0,2,1))
            res = np.array(list(map(lambda x: comb_sdev(x[0], x[1], n)[1:], prepped)))
            np.savetxt(os.path.join(cfg_path, "mean_"+loss),res[:, 0], fmt='%.3f', delimiter=',')
            np.savetxt(os.path.join(cfg_path, "sdev_"+loss),res[:, 1], fmt='%.3f', delimiter=',')
        
        # combine test_res
        n = 250
        fold_list_loss = []
        fold_list_kp = []
        fold_list_time = []
        for i in range(folds):
            with open(os.path.join(cfg_path, "fold_0"+str(i)+'/test_res.txt'), 'r') as results:
                data = np.array(results.read().split('\n'))[:-1]
            fold_list_loss.append(data[0:2].astype(np.float))
            fold_kp = [np.array(data[i].split(",")[:-1], dtype=np.float) for i in range(2,22)] if dataname == "cephal" else [np.array(data[i].split(","), dtype=np.float) for i in range(2,22)]
            fold_kp = np.array(fold_kp)
            # print(fold_kp)
            fold_list_kp.append(np.array(list(zip(fold_kp[0::2], fold_kp[1::2]))))
            fold_list_time.append(float(data[-1]))
            
        prepped_loss = np.array(list(zip(*fold_list_loss)))
        prepped_kp = np.transpose(np.array(list(zip(*fold_list_kp))), (0,3,2,1))
        mean_time = np.mean(fold_list_time)
        res_loss = np.array(comb_sdev(prepped_loss[0], prepped_loss[1], n)[1:])
        res_kp = np.array(list(map(lambda x: list(map(lambda y: comb_sdev(y[0], y[1], n)[1:], x)), prepped_kp))) # apply comb_sdev and only return mean and sdev
        with open(os.path.join(cfg_path,'test_res.txt'), 'a') as test_res:
            for i in res_loss:
                test_res.write("%.3f \n" % i)
            for x,y in zip(res_kp[:,:,0], res_kp[:,:,1]):
                test_res.write(','.join(["%.3f" % elem for elem in x])+'\n')
                test_res.write(','.join(["%.3f" % elem for elem in y])+'\n')
                test_res.write('\n'.join(["%.3f" % j for j in comb_sdev(x,y,n*folds)[1:]])+'\n')
            test_res.write("%.3f \n" % mean_time)
        


            