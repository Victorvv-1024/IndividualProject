import argparse
import numpy  as np

def load_eddy(args):
    """
    """
    path = args.path
    thresholda = args.t0
    thresholdb = args.t1
    thresholdc = args.r0
    thresholdd = args.r1
    thresholde = args.out
    lineno = args.num
    
    filename = '/QAfrom-eddylog.txt'

    pfile = open(path + filename, 'r')

    lines = pfile.readlines()

    if lineno is None:
        lineno = len(lines) - 1
        
    move = []

    #for line in lines-1:
    for i in range(lineno):
        line = lines[i]
        m = 0
        t0, _, t1, _, r0, _, r1, _, out, _, _ = line.split(' ')
        t0, t1, r0, r1, out = float(t0), float(t1), float(r0), float(r1), float(out)
        if t0 < thresholda and t1 < thresholdb and r0 < thresholdc and r1 < thresholdd and out < thresholde:
            m = 1

        move.append(m)
    pfile.close()

    pfile = open(path + '/move_t0-'+str(thresholda)+'_t1-'+str(thresholdb)+'_r0-'+str(thresholdc)+'_r1-'+str(thresholdd)+'_out-'+str(thresholde)+'.txt', 'w')
    pfile.writelines("%s " % str(item) for item in move)
    pfile.close()

def order_volumes(args):
    """ 
    """
    path = args.path
    thresholda = args.t0
    thresholdb = args.t1
    thresholdc = args.r0
    thresholdd = args.r1
    thresholde = args.out
    lineno = args.num
    
    filename = '/QAfrom-eddylog.txt'

    pfile = open(path + filename, 'r')

    lines = pfile.readlines()

    if lineno is None:
        lineno = len(lines) - 1
        
    move = []
    move_score = []
    good_move = []
    bad_move = []

    for i in range(lineno):
        line = lines[i]
        m = 0
        t0, _, t1, _, r0, _, r1, _, out, _, _ = line.split(' ')
        t0, t1, r0, r1, out = float(t0), float(t1), float(r0), float(r1), float(out)
        move_score.append((t0,t1,r0,r1,out))
        if t0 < thresholda and t1 < thresholdb and r0 < thresholdc and r1 < thresholdd and out < thresholde:
            m = 1
            good_move.append((t0,t1,r0,r1,out))
        else: bad_move.append((t0,t1,r0,r1,out))

        move.append(m)
    pfile.close()

    t0_mu = np.average(list(zip(*good_move))[0])
    t1_mu = np.average(list(zip(*good_move))[1])
    r0_mu = np.average(list(zip(*good_move))[2])
    r1_mu = np.average(list(zip(*good_move))[3])
    out_mu = np.average(list(zip(*good_move))[4])

    bad_move_var = []
    for i in bad_move:
        t0_var = abs(t0_mu-i[0])
        t1_var = abs(t1_mu-i[1])
        r0_var = abs(r0_mu-i[2])
        r1_var = abs(r1_mu-i[3])
        out_var = abs(out_mu-i[4])
        bad_move_var.append(t0_var+t1_var+r0_var+r1_var+out_var)

    
    j = 0
    for i in range(len(move)):
        if move[i] == 1:
            move_score[i] = 0
        else: 
            move_score[i] = bad_move_var[j]
            j+=1
    
    sort_index = np.argsort(np.asarray(move_score))


    pfile = open(path + '/move_t0-'+str(thresholda)+'_t1-'+str(thresholdb)+'_r0-'+str(thresholdc)+'_r1-'+str(thresholdd)+'_out-'+str(thresholde)+'.txt', 'w')
    pfile.writelines("%s " % str(item) for item in move)
    pfile.close()

    return sort_index


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--t0", type=float)
    parser.add_argument("--t1", type=float)
    parser.add_argument("--r0", type=float)
    parser.add_argument("--r1", type=float)
    parser.add_argument("--out", type=float)
    parser.add_argument("--num", type=int)
    

    return parser

if __name__ == '__main__':
    args = parser().parse_args()
    sorted_index = order_volumes(args)