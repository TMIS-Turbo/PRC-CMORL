import numpy as np
import math

np.set_printoptions(precision=3)


def get_ep_indices(obj_batch):
    # return sorted indices of undominated objs
    if len(obj_batch) == 0: return np.array([])
    sort_indices = np.lexsort((obj_batch.T[1], obj_batch.T[0]))
    ep_indices = []
    max_val = -np.inf
    for idx in sort_indices[::-1]:
        if obj_batch[idx][1] > max_val:
            max_val = obj_batch[idx][1]
            ep_indices.append(idx)
    return ep_indices[::-1]


# compute the hypervolume given the pareto points, only for 2-dim objectives now
def compute_index(obj_batch, ref_point):
    if obj_batch.shape[1] != 2:
        return 0, 0, 0, 0
    objs = obj_batch[get_ep_indices(obj_batch)]

    objs = np.array(objs)
    objs_len = len(objs)
    obj_batch_len = len(obj_batch)
    print('input size : {}, pareto size : {}'.format(obj_batch_len, objs_len))

    ref_x, ref_y = ref_point  # set referent point as (0, 0)
    x, hypervolume = ref_x, 0.0
    ang_01, ang_12, ang_23, ang_34, ang_45, ang_56, ang_67, ang_78, ang_89 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(objs_len):
        objs_i_x = objs[i][0]
        objs_i_y = objs[i][1]
        hypervolume += (max(ref_x, objs_i_x) - x) * (max(ref_y, objs_i_y) - ref_y)
        x = max(ref_x, objs_i_x)

        ang_i = math.atan(objs_i_x/objs_i_y)*180/math.pi
        if 0 <= ang_i < 10:
            ang_01 += 1
            # print("#0-10#", ang_01)
        elif 10 <= ang_i < 20:
            ang_12 += 1
            # print("#10-20#", ang_12)
        elif 20 <= ang_i < 30:
            ang_23 += 1
            # print("#20-30#", ang_23)
        elif 30 <= ang_i < 40:
            ang_34 += 1
            # print("#30-40#", ang_34)
        elif 40 <= ang_i < 50:
            ang_45 += 1
            # print("#40-50#", ang_45)
        elif 50 <= ang_i < 60:
            ang_56 += 1
            # print("#50-60#", ang_56)
        elif 60 <= ang_i < 70:
            ang_67 += 1
            # print("#60-70#", ang_67)
        elif 70 <= ang_i < 80:
            ang_78 += 1
            # print("#70-80#", ang_78)
        elif 80 <= ang_i <= 90:
            ang_89 += 1
            # print("#80-90#", ang_89)

    e = 1e-9
    ang_01_p, ang_12_p, ang_23_p, ang_34_p, ang_45_p, ang_56_p, ang_67_p, ang_78_p, ang_89_p = ang_01/objs_len, ang_12/objs_len, ang_23/objs_len, ang_34/objs_len, ang_45/objs_len, ang_56/objs_len, ang_67/objs_len, ang_78/objs_len, ang_89/objs_len
    H = -(ang_01_p*math.log(ang_01_p+e)+ang_12_p*math.log(ang_12_p+e)+ang_23_p*math.log(ang_23_p+e)+ang_34_p*math.log(ang_34_p+e)+ang_45_p*math.log(ang_45_p+e)+ang_56_p*math.log(ang_56_p+e)+ang_67_p*math.log(ang_67_p+e)+ang_78_p*math.log(ang_78_p+e)+ang_89_p*math.log(ang_89_p+e))
    H_max = math.log(9)
    E = H/H_max
    D = objs_len/obj_batch_len
    index_ = hypervolume*(1+E)

    print("----------***hypervolume & Evenness & Density & Index***----------:", hypervolume, E, D, index_)
    return index_, hypervolume, E, D
