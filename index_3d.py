import numpy as np
import math
from hypervolume import InnerHyperVolume

np.set_printoptions(precision=3)


# compute the hypervolume given the pareto points, for 3-dim objectives
def compute_index(obj_batch, ref_point):
    HV = InnerHyperVolume(ref_point)
    hv = HV.compute(obj_batch)

    objs_len = len(obj_batch)
    ang1, ang2, ang3, ang4, ang5, ang6, ang7, ang8, ang9 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(objs_len):
        objs_i_x = obj_batch[i][0]
        objs_i_y = obj_batch[i][1]
        objs_i_z = obj_batch[i][2]
        ang_xy = math.atan(objs_i_x/objs_i_y)*180/math.pi
        ang_z_xy = math.atan(objs_i_z / ((objs_i_x**2+objs_i_y**2)**0.5)) * 180 / math.pi
        if 0 <= ang_xy < 30 and 0 <= ang_z_xy < 30:
            ang1 += 1
            print("#ang1#", ang1)
        elif 0 <= ang_xy < 30 and 30 <= ang_z_xy < 60:
            ang2 += 1
            print("#ang2#", ang2)
        elif 0 <= ang_xy < 30 and 60 <= ang_z_xy <= 90:
            ang3 += 1
            print("#ang3#", ang3)
        elif 30 <= ang_xy < 60 and 0 <= ang_z_xy < 30:
            ang4 += 1
            print("#ang4#", ang4)
        elif 30 <= ang_xy < 60 and 30 <= ang_z_xy < 60:
            ang5 += 1
            print("#ang5#", ang5)
        elif 30 <= ang_xy < 60 and 60 <= ang_z_xy <= 90:
            ang6 += 1
            print("#ang6#", ang6)
        elif 60 <= ang_xy <= 90 and 0 <= ang_z_xy < 30:
            ang7 += 1
            print("#ang7#", ang7)
        elif 60 <= ang_xy <= 90 and 30 <= ang_z_xy < 60:
            ang8 += 1
            print("#ang8#", ang8)
        elif 60 <= ang_xy <= 90 and 60 <= ang_z_xy <= 90:
            ang9 += 1
            print("#ang9#", ang9)
    e = 1e-9
    ang_01_p, ang_12_p, ang_23_p, ang_34_p, ang_45_p, ang_56_p, ang_67_p, ang_78_p, ang_89_p = ang1/objs_len, ang2/objs_len, ang3/objs_len, ang4/objs_len, ang5/objs_len, ang6/objs_len, ang7/objs_len, ang8/objs_len, ang9/objs_len
    H = -(ang_01_p*math.log(ang_01_p+e)+ang_12_p*math.log(ang_12_p+e)+ang_23_p*math.log(ang_23_p+e)+ang_34_p*math.log(ang_34_p+e)+ang_45_p*math.log(ang_45_p+e)+ang_56_p*math.log(ang_56_p+e)+ang_67_p*math.log(ang_67_p+e)+ang_78_p*math.log(ang_78_p+e)+ang_89_p*math.log(ang_89_p+e))
    H_max = math.log(9)
    E = H/H_max
    index_ = hv*(1+E)

    print("----------***hypervolume & Evenness & Index***----------:", hv, E, index_)
    return index_, hv, E
