from util import getTables

if __name__ == "__main__":
    table = getTables(r'WS_P[0-9]_S1.mat')
    print len(table)
    table = getTables(r'WS_P1_S[0-9].mat')
    print len(table)