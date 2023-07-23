import numpy as np
from scipy.integrate import simps
from scipy.interpolate import PchipInterpolator

def get_reverse_and_forward(J, V):
    V_r = np.array(V[100:200])
    V_f = np.array(V[200:300])
    J_r = np.array(J[100:200])
    J_f = np.array(J[200:300])
    return V_r, V_f, J_r, J_f

def get_hysteresis(J, V, v_oc_forward, v_oc_reverse):
    reverse = J[v_oc_reverse:200]
    forward = J[199:v_oc_forward]
    reverse_size = 200 - v_oc_reverse
    forward_size = v_oc_forward - 199
    max_ = max(reverse_size, forward_size)
    if reverse_size == forward_size:
        difference = np.abs(reverse-forward).flatten()
        return simps(difference) * 0.012
    else:
        if reverse_size == max_:
            bigger = np.flip(reverse[reverse_size-forward_size:])
            smaller = forward
            extra = reverse[:reverse_size-forward_size+1]
        else:
            bigger = np.flip(forward[:reverse_size])
            smaller = reverse
            extra = forward[reverse_size-1:]
        
        difference = np.abs(bigger - smaller).flatten()
        extra = extra.flatten()
        
        region_1 = simps(difference) * 0.12
        region_2 = simps(extra) * 0.12
        return round((region_1 + region_2), 4)
    
def calculate_max_powers(J, V):
    V_r, V_f, J_r, J_f = get_reverse_and_forward(J, V)
    power_r = V_r * J_r
    power_f = V_f * J_f
    return np.max(power_r), np.max(power_f) 

def calculate_outputs(J, V):
    mp_r, mp_f = calculate_max_powers(J, V)
    J, V = np.array(J), np.array(V)
    j_sc = J[199]
    v_oc_reverse = np.where(J[100:200] < 0, J[100:200], -np.inf).argmin() + 100
    v_oc_forward = np.where(J[200:300] > 0, J[200:300], -np.inf).argmin() + 200
    reverse_lower_limit = 100 if v_oc_reverse-5 < 100 else v_oc_reverse-5
    pchip_reverse = PchipInterpolator(y=V[reverse_lower_limit:v_oc_reverse+5].flatten(), x=J[reverse_lower_limit:v_oc_reverse+5].flatten())
    v_oc_reverse_actual = pchip_reverse(x=[0])
    forward_upper_limit = 300 if v_oc_forward+5 > 300 else v_oc_forward+5
    pchip_forward = PchipInterpolator(y=V[v_oc_forward-5:forward_upper_limit].flatten(), x=-J[v_oc_forward-5:forward_upper_limit].flatten())
    v_oc_forward_actual = pchip_forward(x=[0])

    DoH = get_hysteresis(J, V, v_oc_forward, v_oc_reverse)

    return np.array([round(j_sc[0], 4), round(v_oc_reverse_actual[0], 4), round(v_oc_forward_actual[0], 4), round(mp_r, 4), round(mp_f, 4)])