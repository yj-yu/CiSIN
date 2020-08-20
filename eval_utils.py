def calc_acc(sN, ss_gt, ss_pred):
    n0, t0, n1, t1 = 0,0,0,0
    for si in range(0,sN):
        for sj in range(si+1,sN):
            if ss_gt[si,sj] == 0:
                t0 += 1
                if ss_pred[si,sj] < 0.5:
                    n0 += 1
            else:
                t1 += 1
                if ss_pred[si,sj] >= 0.5:
                    n1 += 1
    score = 0
    if t0 == 0:
        score += 0.5
    else:
        score += 0.5 * n0 / t0
    if t1 == 0:
        score += 0.5
    else:
        score += 0.5 * n1 / t1
    #new_acc.append(score)
    if t0 +  t1 == 0:
        return 1
    else:
        return (n0 + n1) / (t0 + t1)