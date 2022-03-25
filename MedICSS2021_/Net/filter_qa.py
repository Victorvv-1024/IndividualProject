import argparse

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

                # print line.split(' ')
    pfile.close()

    pfile = open(path + '/move_t0-'+str(thresholda)+'_t1-'+str(thresholdb)+'_r0-'+str(thresholdc)+'_r1-'+str(thresholdd)+'_out-'+str(thresholde)+'.txt', 'w')
    pfile.writelines("%s " % str(item) for item in move)
    pfile.close()

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
    load_eddy(args)