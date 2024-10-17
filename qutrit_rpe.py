import pygsti
import numpy as np
from scipy.linalg import expm
from pygsti.circuits import Circuit
from matplotlib import pyplot as plt
from pygsti.processors import QuditProcessorSpec

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
    # Circuit Construction 