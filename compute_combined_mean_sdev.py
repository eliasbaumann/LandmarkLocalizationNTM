import sys
import numpy as np

def comb_sdev(means,sdevs,n):
    sig_x = []
    sig_x2 =[]
    for m,s in zip(means,sdevs):
        sig_x.append((m*n))
        sig_x2.append(s**2 * (n-1) + ((m*n)**2/n))
    tn = len(means)*n
    tx = np.sum(sig_x)
    txx = np.sum(sig_x2)
    sd = np.sqrt((txx-tx**2/tn) / (tn-1))
    return tn, tx/tn, sd


if __name__ == '__main__':
    rows = []
    with open(sys.argv[1], 'r') as test_res:
        data = np.array(test_res.read().split('\n'))[:-1]
        data = [np.array(i.split(","), dtype=np.float) for i in data]
        
    
    n = len(data[2])
    if n == 20:
        n = 19
        for i in range(2,10):
            data[i] = data[i][:-1]
    print("loss: ")
    print("mean:", data[0][0])
    print("sd:", data[1][0])
    print("----------------------------")
    total_n, mean_kp, std_kp = comb_sdev(data[2],data[3],n)
    print("per kp loss: ")
    print("mean:", mean_kp)
    print("sd:", std_kp)
    print("----------------------------")
    total_n, mean_dst, std_dst = comb_sdev(data[4],data[5],n)
    print("Coordinate distance ")
    print("mean:", mean_dst)
    print("sd:", std_dst)
    print("----------------------------")
    total_n, mean_mrg, std_mrg = comb_sdev(data[6],data[7],n)
    print("within margin (2px): ")
    print("mean:", mean_mrg)
    print("sd:", std_mrg)
    print("----------------------------")
    total_n, mean_cgt, std_cgt = comb_sdev(data[8],data[9],n)
    print("closest to gt: ")
    print("mean:", mean_cgt)
    print("sd:", std_cgt)

    print("copy wihtout per kp:")
    print("%.2f +- %.2f & %.2f +- %.2f & %.2f +- %.2f & %.2f +- %.2f & %d" % (data[0][0], data[1][0], mean_dst, std_dst, mean_mrg, std_mrg, mean_cgt, std_cgt, data[10][0]))
    
    input("Press Enter to continue...")