import pygsti
import numpy as np
from scipy.linalg import expm
from pygsti.circuits import Circuit
from matplotlib import pyplot as plt
from pygsti.processors import QuditProcessorSpec

from quapack.pyRPE import RobustPhaseEstimation
from quapack.pyRPE.quantum import Q as _rpeQ

# Gell-Mann matrices
gellmann_matrices = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
]

# ==================================================================================================
# unitary models
# we only consider errors that commute with the target operation  
X01_gen = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
X12_gen = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Gamma_01 = np.diag([1, 1, -2])
Gamma_12 = np.diag([-2, 1, 1])

target_phis = np.array([0, 0, 0, -2*np.pi/3, -4*np.pi/3, 0, -4*np.pi/3, -2*np.pi/3])

def modelX01(theta, gamma):
    return expm(-(1j/2)*((np.pi/2 + theta)*X01_gen + gamma*Gamma_01))

def modelZ01():
    return np.diag([np.exp(1j*np.pi/2), 1, 1])

def modelZ12():
    return  np.diag([1, 1, np.exp(-1j*np.pi/2)])

def modelX12(theta, gamma):
    return expm(-(1j/2)*((np.pi/2 + theta)*X12_gen + gamma*Gamma_12))

def modelCZ(phis):
    phis = target_phis + np.array(phis)
    return np.diag([1, *np.exp(-1j*phis)])


# ==================================================================================================
# pygsti model construction
# we only consider errors that commute with the target operation  

def parse_error_vector(x, qids=['Q0', 'Q1']):
    info = {
        'single_qutrit': {
            qids[0]: {
                'X01' : x[0],
                'phase01': x[1],
                'X12' : x[2], 
                'phase12': x[3]
            },
            qids[1]: {
                'X01' : x[4],
                'phase01': x[5],
                'X12' : x[6],
                'phase12': x[7]
            }
        },
        'two_qutrit': {
            'phi1': x[8],
            'phi2': x[9],
            'phi3': x[10],
            'phi4': x[11],
            'phi5': x[12],
            'phi6': x[13],
            'phi7': x[14],
            'phi8': x[15]
        }  
    }
    return info

def random_error_vector(single_qutrit_rates, two_qutrit_rates):
    q1_vec = np.random.multivariate_normal(np.zeros(4), np.eye(4)*single_qutrit_rates)
    q2_vec = np.random.multivariate_normal(np.zeros(4), np.eye(4)*single_qutrit_rates)
    two_qubit_vec = np.random.multivariate_normal(np.zeros(8), np.eye(8)*two_qutrit_rates)
    return np.concatenate((q1_vec, q2_vec, two_qubit_vec))


from pygsti.baseobjs import ExplicitStateSpace
from pygsti.models import ExplicitOpModel
from pygsti.models.modelconstruction import create_spam_vector
from pygsti.modelmembers.povms import UnconstrainedPOVM, FullPOVMEffect
from pygsti.modelmembers.states import FullState
from pygsti.tools import change_basis
from pygsti.baseobjs import Basis
from pygsti.tools import unitary_to_std_process_mx

def make_two_qutrit_model(error_vector, single_qutrit_depol=0., two_qutrit_depol=0., qids=['Q0', 'Q1']):
    
    # Parse error vector
    errors = parse_error_vector(error_vector, qids)
    x01_Q0 = errors['single_qutrit'][qids[0]]['X01']
    x12_Q0 = errors['single_qutrit'][qids[0]]['X12']
    x01_Q1 = errors['single_qutrit'][qids[1]]['X01']
    x12_Q1 = errors['single_qutrit'][qids[1]]['X12']
    
    phase01_Q0 = errors['single_qutrit'][qids[0]]['phase01']
    phase12_Q0 = errors['single_qutrit'][qids[0]]['phase12']
    phase01_Q1 = errors['single_qutrit'][qids[1]]['phase01']
    phase12_Q1 = errors['single_qutrit'][qids[1]]['phase12']

    phi1 = errors['two_qutrit']['phi1']
    phi2 = errors['two_qutrit']['phi2']
    phi3 = errors['two_qutrit']['phi3']
    phi4 = errors['two_qutrit']['phi4']
    phi5 = errors['two_qutrit']['phi5']
    phi6 = errors['two_qutrit']['phi6']
    phi7 = errors['two_qutrit']['phi7']
    phi8 = errors['two_qutrit']['phi8']
    phis = [phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8]

    # Define single qutrit unitaries
    X01_Q0_unitary = modelX01(x01_Q0, phase01_Q0)
    Z01_unitary = modelZ01()
    X12_Q0_unitary = modelX12(x12_Q0, phase12_Q0)
    Z12_unitary = modelZ12()

    X01_Q1_unitary = np.kron(np.eye(3), modelX01(x01_Q1, phase01_Q1))
    X12_Q1_unitary = np.kron(np.eye(3), modelX12(x12_Q1, phase12_Q1))

    # Define two qutrit unitary
    phis_target = np.array([0, 0, 0, -2*np.pi/3, -4*np.pi/3, 0, -2*np.pi/3, -4*np.pi/3])
    CZ_unitary = modelCZ(np.array(phis) + phis_target)

    target_unitary_mapping = {
        'Gx01': X01_Q0_unitary,
        'Gx12': X12_Q0_unitary,
        'Gz01': Z01_unitary,
        'Gz12': Z12_unitary,
        'Gcz': CZ_unitary
    }

    rho0vec = np.asarray([1, 0, 0])
    rho00= np.kron(rho0vec, rho0vec)

    E0vec = np.asarray([1, 0, 0])
    E1vec = np.asarray([0, 1, 0])
    E2vec = np.asarray([0, 0, 1])

    E00 = np.kron(E0vec, E0vec)
    E01 = np.kron(E0vec, E1vec)
    E02 = np.kron(E0vec, E2vec)
    E10 = np.kron(E1vec, E0vec)
    E11 = np.kron(E1vec, E1vec)
    E12 = np.kron(E1vec, E2vec)
    E20 = np.kron(E2vec, E0vec)
    E21 = np.kron(E2vec, E1vec)
    E22 = np.kron(E2vec, E2vec)

    povm_dict = {'00' : E00, '01' : E01, '02' : E02, '10' : E10, '11' : E11, '12' : E12, '20' : E20, '21' : E21, '22' : E22}

    
    gate_names = ['Gx01', 'Gx12', 'Gz01', 'Gz12', 'Gcz']
    availability = {
        'Gx01': [(qids[0], ), (qids[1], )], 
        'Gx12': [(qids[0], ), (qids[1], )], 
        'Gz01': [(qids[0], ), (qids[1], )], 
        'Gz12': [(qids[0], ), (qids[1], )], 
        'Gcz': [(qids[0], qids[1])]
    }

    # define the processor spec and make the model
    pspec = QuditProcessorSpec([qids[0], qids[1]], qudit_udims=[3, 3], gate_names=gate_names,
                    nonstd_gate_unitaries=target_unitary_mapping,
                    prep_names = ['rho0'], povm_names = ['Mdefault'],
                    nonstd_preps = {'rho0': rho00}, nonstd_povms = {'Mdefault': povm_dict},
                    availability=availability
                    )
    model = pygsti.models.modelconstruction.create_explicit_model(pspec,ideal_gate_type='full TP', ideal_spam_type='full pure', basis='gm')

    # set the X gates on Q1, which is different than on Q0
    model.operations[('Gx01', qids[1])] = change_basis(unitary_to_std_process_mx(X01_Q1_unitary), 'std', 'gm')
    model.operations[('Gx12', qids[1])] = change_basis(unitary_to_std_process_mx(X12_Q1_unitary), 'std', 'gm')

    # add depolarizing noise
    model.operations[('Gx01', qids[0])].depolarize(single_qutrit_depol)
    model.operations[('Gz01', qids[0])].depolarize(single_qutrit_depol)
    model.operations[('Gx12', qids[0])].depolarize(single_qutrit_depol)
    model.operations[('Gz12', qids[0])].depolarize(single_qutrit_depol)
    model.operations[('Gx01', qids[1])].depolarize(single_qutrit_depol)
    model.operations[('Gz01', qids[1])].depolarize(single_qutrit_depol)
    model.operations[('Gx12', qids[1])].depolarize(single_qutrit_depol)
    model.operations[('Gz12', qids[1])].depolarize(single_qutrit_depol)
    model.operations[('Gcz', qids[0], qids[1])].depolarize(two_qutrit_depol)
    return model


    # ==================================================================================================
    # Circuit Definition

from numpy.linalg import matrix_power

def gX01(qid):
    return [(f'Gx01', qid)]

def gY01(qid):
    return [(f'Gz01', qid)] + [(f'Gx01', qid)] + [(f'Gz01', qid)]*3

def gX12(qid):
    return [(f'Gx12', qid)]

def gY12(qid):
    return [(f'Gz12', qid)] + [(f'Gx12', qid)] + [(f'Gz12', qid)]*3

def gZ01(qid):
    return [(f'Gz01', qid)]

def gZ12(qid):
    return [(f'Gz12', qid)]

def gX01_inv(qid):
    return [(f'Gx01', qid)]*3 + [(f'Gz12', qid)]*2

def gX01pi_inv(qid):
    return [(f'Gx01', qid)]*2 + [(f'Gz12', qid)]*2

def gX12_inv(qid):
    return [(f'Gx12', qid)]*3 + [(f'Gz01', qid)]*2

def gX12pi_inv(qid):
    return [(f'Gx12', qid)]*2 + [(f'Gz01', qid)]*2

def gY01_inv(qid):
    return gY01(qid)*3 + [(f'Gz12', qid)]*2

def gY12_inv(qid):
    return gY12(qid)*3 + [(f'Gz01', qid)]*2

def gY01pi_inv(qid):
    return gY01(qid)*2 + [(f'Gz12', qid)]*2

def check_inverse_defs():
    """
    Check the inverse definitions are correct
    """
    ux01 = modelX01(0,0)
    uz01 = modelZ01()
    ux12 = modelX12(0,0)
    uz12 = modelZ12()
    uy01 = matrix_power(uz01, 3)@ux01@uz01
    uy12 = matrix_power(uz12, 3)@ux12@uz12

    print(np.all(np.isclose(ux01@(ux01@ux01@ux01@uz12@uz12), -np.eye(3))))
    print(np.all(np.isclose(ux12@(ux12@ux12@ux12@uz01@uz01), -np.eye(3))))
    print(np.all(np.isclose(uy01@(uy01@uy01@uy01@uz12@uz12), -np.eye(3))))
    print(np.all(np.isclose(uy12@(uy12@uy12@uy12@uz01@uz01), -np.eye(3))))
    print(np.all(np.isclose(ux01@ux01@(ux01@ux01@uz12@uz12), -np.eye(3))))
    print(np.all(np.isclose(ux12@ux12@(ux12@ux12@uz01@uz01), -np.eye(3))))


def make_rpe_circuit(germ, prep, meas, depth, line_labels):
    prep_circ = Circuit(prep, line_labels=line_labels)
    germ_circ = Circuit(germ, line_labels=line_labels)
    meas_circ = Circuit(meas, line_labels=line_labels)
    return prep_circ + germ_circ * depth + meas_circ

class RPEDesign1QT:
    def __init__(self, depths, qid, line_labels=None):
        self.depths = depths
        self.qid = qid
        if line_labels is None:
            line_labels = [qid]
        self.line_labels = line_labels
        self.circuit_dict = self._construct()
        self.circ_list = self._make_circuit_list()


    def _make_circuit_list(self):
        circs = []
        for param_label in self.circuit_dict.keys():
            for type_label in self.circuit_dict[param_label].keys():
                circs.extend(self.circuit_dict[param_label][type_label])
        return pygsti.remove_duplicates(circs)    

    def _construct(self):
        circ_dict = {
            'Phase01': {},
            'Phase12': {},
            'X01 overrot': {},
            'X12 overrot': {}
        }
        circ_dict['X01 overrot']['I'] = self.make_x01_overrot_cos_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['X01 overrot']['Q'] = self.make_x01_overrot_sin_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['X12 overrot']['I'] = self.make_x12_overrot_cos_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['X12 overrot']['Q'] = self.make_x12_overrot_sin_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['Phase01']['I'] = self.make_phase01_cos_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['Phase01']['Q'] = self.make_phase01_sin_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['Phase12']['I'] = self.make_phase12_cos_circuits(self.depths, self.qid, self.line_labels)
        circ_dict['Phase12']['Q'] = self.make_phase12_sin_circuits(self.depths, self.qid, self.line_labels)
        return circ_dict

    def make_x01_overrot_cos_circuits(self, depths, qid, line_labels):
        return [make_rpe_circuit(gX01(qid), [], [], depth, line_labels) for depth in depths]
    
    def make_x01_overrot_sin_circuits(self, depths, qid, line_labels):
        return [make_rpe_circuit(gX01(qid), gX01(qid), [], depth, line_labels) for depth in depths]
    
    def make_x12_overrot_cos_circuits(self, depths, qid, line_labels):
        prep = gX01(qid)+gX01(qid)
        return [make_rpe_circuit(gX12(qid), prep, [], depth, line_labels) for depth in depths]
    
    def make_x12_overrot_sin_circuits(self, depths, qid, line_labels):
        prep = gX01(qid)+gX01(qid) + gX12(qid)
        return [make_rpe_circuit(gX12(qid), prep, [], depth, line_labels) for depth in depths]
    
    def make_phase01_cos_circuits(self, depths, qid, line_labels):
        germ = gX01(qid) + gZ01(qid)*2 + gX01(qid) + gZ01(qid)*2
        prep = gX01(qid) + gX01(qid) + gY12(qid)
        meas = gY12_inv(qid) + gX01pi_inv(qid)
        # prep = gX01(qid) + gX01(qid) + gY12(qid) + gX01(qid) + gZ01(qid)
        # meas = gZ01(qid)*3 + gX01_inv(qid) + gY12_inv(qid) + gX01pi_inv(qid)
        # prep = gX01(qid) + gX01(qid) + gY12(qid)
        # meas = gY12_inv(qid) + gX01pi_inv(qid)
        return [make_rpe_circuit(germ, prep, meas, depth, line_labels) for depth in depths]

    def make_phase01_sin_circuits(self, depths, qid, line_labels):
        germ = gX01(qid) + gZ01(qid)*2 + gX01(qid) + gZ01(qid)*2
        prep = gX01(qid) + gX01(qid) + gX12(qid)
        meas = gY12_inv(qid) + gX01pi_inv(qid)
        # prep = gX01(qid) + gX01(qid) + gY12(qid) + gX01(qid)
        # meas = gY12_inv(qid) + gX01pi_inv(qid)
        return [make_rpe_circuit(germ, prep, meas, depth, line_labels) for depth in depths]
        
    
    def make_phase12_cos_circuits(self, depths, qid, line_labels):
        germ = gX12(qid) + gZ12(qid)*2 + gX12(qid) + gZ12(qid)*2
        prep = gY01(qid) 
        meas = gY01_inv(qid) 
        return [make_rpe_circuit(germ, prep, meas, depth, line_labels) for depth in depths]
    
    def make_phase12_sin_circuits(self, depths, qid, line_labels):
        germ = gX12(qid) + gZ12(qid)*2 + gX12(qid) + gZ12(qid)*2
        prep = gX01_inv(qid) 
        meas = gY01_inv(qid) 
        return [make_rpe_circuit(germ, prep, meas, depth, line_labels) for depth in depths]
    
    def save_circuits(self, filename=None):
        if filename is None:
            filename = f'rpe_design_1qt_{self.qid}.txt'
        pygsti.io.write_circuit_list(filename, self.circ_list)

class RPEDesign2QT:
    def __init__(self, depths, qids):
        self.depths = depths
        self.qids = qids
        self.circuit_dict = self._construct()
        self.circ_list = self._make_circuit_list()

    def _make_circuit_list(self):
        circs = []
        for param_label in self.circuit_dict.keys():
            for type_label in self.circuit_dict[param_label].keys():
                circs.extend(self.circuit_dict[param_label][type_label])
        return pygsti.remove_duplicates(circs)    

    def _construct(self):
        circ_dict = {
            'theta1': {},
            'theta2': {},
            'theta3': {},
            'theta4': {},
            'theta5': {},
            'theta6': {},
            'theta7': {},
            'theta8': {},
        }
        circ_dict['theta1']['I'] = self.make_theta1_cos_circuits()
        circ_dict['theta1']['Q'] = self.make_theta1_sin_circuits()
        circ_dict['theta2']['I'] = self.make_theta2_cos_circuits()
        circ_dict['theta2']['Q'] = self.make_theta2_sin_circuits()
        circ_dict['theta3']['I'] = self.make_theta3_cos_circuits()
        circ_dict['theta3']['Q'] = self.make_theta3_sin_circuits()
        circ_dict['theta4']['I'] = self.make_theta4_cos_circuits()
        circ_dict['theta4']['Q'] = self.make_theta4_sin_circuits()
        circ_dict['theta5']['I'] = self.make_theta5_cos_circuits()
        circ_dict['theta5']['Q'] = self.make_theta5_sin_circuits()
        circ_dict['theta6']['I'] = self.make_theta6_cos_circuits()
        circ_dict['theta6']['Q'] = self.make_theta6_sin_circuits()
        circ_dict['theta7']['I'] = self.make_theta7_cos_circuits()
        circ_dict['theta7']['Q'] = self.make_theta7_sin_circuits()
        circ_dict['theta8']['I'] = self.make_theta8_cos_circuits()
        circ_dict['theta8']['Q'] = self.make_theta8_sin_circuits()
        return circ_dict

    def make_theta1_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gY01(qid1)
        meas = gY01_inv(qid1)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]
    
    def make_theta1_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)
        meas = gY01_inv(qid1)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]
    
    def make_theta2_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)*2 + gY12(qid1)
        meas = gY12_inv(qid1) + gX01pi_inv(qid1)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]
    
    def make_theta2_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)*2 + gX12(qid1)
        meas = gY12_inv(qid1) + gX01pi_inv(qid1)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta3_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gY01(qid1) + gX01(qid0)*2
        meas = gY01_inv(qid1) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta3_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1) + gX01(qid0)*2
        meas = gY01_inv(qid1) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta4_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)*2 + gY12(qid1) + gX01(qid0)*2
        meas = gY12_inv(qid1) + gX01pi_inv(qid1) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta4_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)*2 + gX12(qid1) + gX01(qid0)*2
        meas = gY12_inv(qid1) + gX01pi_inv(qid1) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta5_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid0)*2 + gX12(qid0)*2 + gY01(qid1)
        meas = gY01_inv(qid1) + gX12pi_inv(qid0) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta5_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid0)*2 + gX12(qid0)*2 + gX01(qid1) 
        meas = gY01_inv(qid1) + gX12pi_inv(qid0) + gX01pi_inv(qid0) 
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta6_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)*2 + gY12(qid1) + gX01(qid0)*2 + gX12(qid0)*2
        meas = gY12_inv(qid1) + gX01pi_inv(qid1) + gX12pi_inv(qid0) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta6_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid1)*2 + gX12(qid1) + gX01(qid0)*2 + gX12(qid0)*2
        meas = gY12_inv(qid1) + gX01pi_inv(qid1) + gX12pi_inv(qid0) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta7_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gY01(qid0) + gX01(qid1)*2 
        meas = gY01_inv(qid0) + gX01pi_inv(qid1)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta7_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid0) + gX01(qid1)*2 
        meas = gY01_inv(qid0) + gX01pi_inv(qid1)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta8_cos_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid0)*2 + gY12(qid0) + gX01(qid1)*2 
        meas = gY12_inv(qid0) + gX01pi_inv(qid1) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def make_theta8_sin_circuits(self):
        line_labels = self.qids
        qid0 = self.qids[0]
        qid1 = self.qids[1]
        prep = gX01(qid0)*2 + gX12(qid0) + gX01(qid1)*2 
        meas = gY12_inv(qid0) + gX01pi_inv(qid1) + gX01pi_inv(qid0)
        return [make_rpe_circuit([('Gcz', qid0, qid1)], prep, meas, d, line_labels) for d in self.depths]

    def save_circuits(self, filename=None):
        if filename is None:
            filename = f'rpe_design_1qt_{self.qid}.txt'
        pygsti.io.write_circuit_list(filename, self.circ_list)

# ==================================================================================================
# Estimation and display

def rectify_angle(angle):
    # return in range [-pi, pi]
    return (angle + np.pi) % (2*np.pi) - np.pi

def plot_outcome_dist_2qt(outcomes, target=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # keys a
    keys = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
    vals = np.zeros(9)
    if target is not None:
        tag_vals = np.zeros(9)
        for idx, key in enumerate(keys):
            if key in target:
                tag_vals[idx] = target[key]
    for idx, key in enumerate(keys):
        if key in outcomes:
            vals[idx] = outcomes[key]
    if target is None:
        ax.bar(keys, vals)
    else:
        ax.bar(keys, vals, label='Simulated', alpha=0.5)
        ax.bar(keys, tag_vals, label='Target', alpha=0.5)
        ax.legend()


def estimate_phase_from_counts(cos_plus_counts, cos_minus_counts, sin_plus_counts, sin_minus_counts, depths):
    experiment = _rpeQ()
    for idx, d in enumerate(depths):
        if d == 0:
            continue
        experiment.process_cos(d, (int(cos_plus_counts[idx]), int(cos_minus_counts[idx])))
        experiment.process_sin(d, (int(sin_plus_counts[idx]), int(sin_minus_counts[idx])))
    analysis = RobustPhaseEstimation(experiment)
    last_good_generation = analysis.check_unif_local(historical=True)
    estimates = analysis.angle_estimates
    return estimates, last_good_generation

class RPEEstimatorBase:
    def __init__(self, dataset, edesign, germ_quadrature_labels):
        self.edesign = edesign
        self.dataset = dataset
        self.germ_quadrature_labels = germ_quadrature_labels
        self.rpe_outcome_dict = self._construct_rpe_outcome_dict()
        self.raw_trig_estimates, self.trig_last_good_gens = self._estimate_trig_params()
        self.signals = self.extract_signals(dataset)

    @property
    def param_estimates(self):
        raise NotImplementedError
    
    def _construct_rpe_outcome_dict(self):
        rpe_count_dict = {}
        for param_label in self.edesign.circuit_dict.keys():
            rpe_count_dict[param_label] = {}
            for type_label in self.edesign.circuit_dict[param_label].keys():
                rpe_count_dict[param_label][type_label] = []
                for depth_idx, circ in enumerate(self.edesign.circuit_dict[param_label][type_label]):
                    counts = self.dataset[circ].counts
                    plus_count_labels = self.germ_quadrature_labels[param_label]['+']
                    minus_count_labels = self.germ_quadrature_labels[param_label]['-']
                    plus_counts = sum([counts[label] for label in plus_count_labels])
                    minus_counts = sum([counts[label] for label in minus_count_labels])
                    rpe_count_dict[param_label][type_label].append((plus_counts, minus_counts))
        return rpe_count_dict
    
    def _estimate_trig_params(self):
        trig_estimates = {}
        lggs = {}
        for param_label in self.rpe_outcome_dict.keys():
            cos_counts = self.rpe_outcome_dict[param_label]['I']
            sin_counts = self.rpe_outcome_dict[param_label]['Q']
            cos_plus_counts = [c[0] for c in cos_counts]
            cos_minus_counts = [c[1] for c in cos_counts]
            sin_plus_counts = [c[0] for c in sin_counts]
            sin_minus_counts = [c[1] for c in sin_counts]
            try:
                estimates, last_good_generation = estimate_phase_from_counts(cos_plus_counts, cos_minus_counts, sin_plus_counts, sin_minus_counts, self.edesign.depths)
                trig_estimates[param_label] = estimates
                lggs[param_label] = last_good_generation
            except:
                print(f'Failed to estimate for {param_label}')
                trig_estimates[param_label] = np.zeros(len(self.edesign.depths))
                lggs[param_label] = np.zeros(len(self.edesign.depths))
        return trig_estimates, lggs
            

    def plot_all_outcomes(self, target_ds=None):
        for param_label in self.rpe_outcome_dict.keys():
            fig, axs = plt.subplots(2, len(self.edesign.depths), sharey=True, figsize=(3*len(self.edesign.depths), 6))
            for idx, type_label in enumerate(self.rpe_outcome_dict[param_label].keys()):
                for depth_idx, circ in enumerate(self.edesign.circuit_dict[param_label][type_label]):
                    ax = axs[idx][depth_idx]
                    outcomes = self.dataset[circ].counts
                    if target_ds is not None:
                        target = target_ds[circ].counts
                        plot_outcome_dist_2qt(outcomes, target=target, ax=ax)
                    else: 
                        plot_outcome_dist_2qt(outcomes, ax=ax)
                    ax.set_title(f'{type_label} at {self.edesign.depths[depth_idx]}')
            plt.suptitle(param_label)
            plt.tight_layout()
            plt.show()
            plt.figure()
                

    def extract_signals(self, dataset):
        signals = {}
        for param_label in self.rpe_outcome_dict.keys():
            signals[param_label] = []
            inphase_counts = self.rpe_outcome_dict[param_label]['I']
            quadrature_counts = self.rpe_outcome_dict[param_label]['Q']
            for idx, d in enumerate(self.edesign.depths):
                inphase_plus = inphase_counts[idx][0]
                inphase_minus = inphase_counts[idx][1]
                quadrature_plus = quadrature_counts[idx][0]
                quadrature_minus = quadrature_counts[idx][1]
                try:
                    s_real = 1 - 2 * inphase_plus/(inphase_plus + inphase_minus)
                    s_imag = 1 - 2 * quadrature_plus/(quadrature_plus + quadrature_minus)
                except:
                    s_real = 0
                    s_imag = 0
                signals[param_label].append(s_real + 1j*s_imag)
        return signals
    
    def plot_signal_on_circle(self, signal, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 6))
        # plot the signals on the complex plane with a colormap for the depth
        depths = self.edesign.depths
        for idx, d in enumerate(depths):
            ax.scatter(signal[idx].real, signal[idx].imag, color=plt.cm.viridis(idx/len(depths)))
        ax.set_title(title)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_aspect('equal')
        ax.grid()
        # add colorbar 
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(depths)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Depth index')
        # draw the unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black')
        ax.add_artist(circle)
        # set the axis limits
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

    def plot_all_signals(self):
        for param_label in self.signals.keys():
            fig, ax = plt.subplots(1, figsize=(12, 6))
            self.plot_signal_on_circle(self.signals[param_label], ax=ax, title=param_label)
            plt.show()
        
class RPEEstimator1QT(RPEEstimatorBase):
    def __init__(self, dataset, edesign, germ_quadrature_labels):
        super().__init__(dataset, edesign, germ_quadrature_labels)


    @property
    def param_estimates(self):
        return{
            'Phase01': rectify_angle(self.raw_trig_estimates['Phase01'][self.trig_last_good_gens['Phase01']])/3,
            'Phase12': rectify_angle(self.raw_trig_estimates['Phase12'][self.trig_last_good_gens['Phase12']])/3,
            'X01 overrot': -rectify_angle(self.raw_trig_estimates['X01 overrot'][self.trig_last_good_gens['X01 overrot']]) - np.pi/2,
            'X12 overrot': -rectify_angle(self.raw_trig_estimates['X12 overrot'][self.trig_last_good_gens['X12 overrot']]) - np.pi/2,
        }


def unwrap_phase_pair(estimate, reference):
    """
    Unwrap a phase estimate to a reference phase
    """
    # ensure the reference is in the range (-pi, pi]
    reference = rectify_angle(reference)

    unwrapped_phase = (estimate - reference) % (2*np.pi) + reference

    if unwrapped_phase - reference > np.pi:
        unwrapped_phase -= 2*np.pi
    elif unwrapped_phase - reference <= -np.pi:
        unwrapped_phase += 2*np.pi
    return unwrapped_phase

class RPEEstimator2QT(RPEEstimatorBase):
    def __init__(self, dataset, edesign, germ_quadrature_labels):
        super().__init__(dataset, edesign, germ_quadrature_labels)

    @property
    def raw_estimates(self):
        return np.array([
            self.raw_trig_estimates['theta1'][self.trig_last_good_gens['theta1']],
            self.raw_trig_estimates['theta2'][self.trig_last_good_gens['theta2']],
            self.raw_trig_estimates['theta3'][self.trig_last_good_gens['theta3']],
            self.raw_trig_estimates['theta4'][self.trig_last_good_gens['theta4']],
            self.raw_trig_estimates['theta5'][self.trig_last_good_gens['theta5']],
            self.raw_trig_estimates['theta6'][self.trig_last_good_gens['theta6']],
            self.raw_trig_estimates['theta7'][self.trig_last_good_gens['theta7']],
            self.raw_trig_estimates['theta8'][self.trig_last_good_gens['theta8']]
        ])

    @property
    def param_estimates(self):
        eigvals_from_raw = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0, -1, 0, 1],
            [-1, 0, 0, 1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 1, 0]
        ])
        raw_estimates = self.raw_estimates
        eigvals = np.linalg.inv(eigvals_from_raw) @ raw_estimates
        return eigvals


