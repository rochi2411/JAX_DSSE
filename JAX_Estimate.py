import pandas as pd
from functools import partial
from opendss_wrapper import OpenDSS
import py_dss_interface
import datetime as dt
import scipy
import numpy as np
from jaxopt import GradientDescent
import jax.numpy as jnp
import time
from jax import jacobian,grad,jit,vmap,lax
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import ctypes

# 4 Bus data
#DSS_file_path = r"C:\Users\rdutta24\SURI_Project\openpy_dsse\NEW_DSSE\4Bus_DSS\Master_4node.dss"

# 13 bus data
#DSS_file_path=r"C:\Users\rdutta24\SURI_Project\OpenPy-DSSE\openpy_dsse\examples\13NodeIEEE\OpenDSS files\Master13NodeIEEE.dss"

# 37 bus data
#DSS_file_path=r"C:\Users\rdutta24\SURI_Project\OpenPy-DSSE\openpy_dsse\examples\37NodeIEEE\OpenDSS files\Master_ieee37.DSS"

# 123 bus data
DSS_file_path=r"C:\Users\rdutta24\SURI_Project\OpenPy-DSSE\openpy_dsse\examples\123Bus\IEEE123Master.dss"

dss = py_dss_interface.DSSDLL()
dss.text(f"compile [{DSS_file_path}]")

system = OpenDSS(DSS_file_path, dt.timedelta(minutes=1), dt.datetime(2022, 1, 1))
dss.solution_solve()


def Ymatrix_noPU_dss(Y_size):
    Y_list = dss.circuit_system_y()
    Y_dense = jnp.zeros([Y_size, Y_size], dtype=complex)
    
    for y in range(Y_size):
        for z in range(Y_size):
            list_idx = 2 * (Y_size * y + z)
            Y_dense = Y_dense.at[y,z].set(Y_list[list_idx] + 1j * Y_list[list_idx + 1])
    
    # TODO [Future Task] Directly get sparse without converting into dense
    Ybus_sparse = scipy.sparse.csr_matrix(Y_dense)

    return Ybus_sparse

dss.dss_obj.SolutionI(ctypes.c_int32(36), ctypes.c_int32(1))

#Y = dss.YMatrix.getYsparse()
element=['vsources', 'transformers', 'lines', 'loads', 'capacitors']

Y_order = dss.circuit_y_node_order()
Y_order=[node.lower() for node in Y_order]
#Y_mat_sp = scipy.sparse.csr_matrix(Y)
Y_mat_sp=Ymatrix_noPU_dss(len(Y_order))

Y_original = Y_mat_sp.toarray()

buses = system.get_all_buses()   # or dss.circuit_all_bus_names()
nodes = sorted(dss.circuit_all_node_names(), key=lambda x: (buses.index(x.split('.')[0]), x.split('.')[1]))
index_map = [Y_order.index(node) for node in nodes] # Arrange the y_node order
lines = system.get_all_elements('Line')  # or dss.Lines.AllNames()
volt= system.get_all_bus_voltages()   # Dict {bus_with_node : Voltage}

Y_matrix = Y_original[np.ix_(index_map, index_map)] # Final sorted Y matrix

n_buses =  len(buses)
n_nodes = len(nodes)
n_lines = len(lines)
n_vi=24  # no. of voltage measurements
n_pQi=22 # no. of power injection measurements
Bus_Nodes=[]

for i in range(n_buses):
    dss.circuit_set_active_bus(buses[i])
    Bus_Nodes.append(dss.bus_nodes())

expand_nodes=[]
for bus in buses:
    for i in range(1,4):
        expand_nodes.append(bus+'.'+str(i))

node1,node2,node3=[],[],[]
for i in range(len(nodes)):
    if(nodes[i].split('.')[1]=='1'):
        node1.append(i)
    elif(nodes[i].split('.')[1]=='2'):
        node2.append(i)
    elif(nodes[i].split('.')[1]=='3'):
        node3.append(i)

node1 = jnp.array(node1)
node2 = jnp.array(node2)
node3 = jnp.array(node3)

def expand_ybus_general(ybus, original_nodes, all_node):
    # Create a mapping from original nodes to their indices
    original_indices = {node: i for i, node in enumerate(original_nodes)}
    
    # Initialize the expanded Ybus matrix with zeros
    n = len(all_node)
    expanded_ybus = jnp.zeros((n, n), dtype=complex)
    
    # Populate the expanded Ybus matrix
    for i, node_i in enumerate(all_node):
        for j, node_j in enumerate(all_node):
            if node_i in original_indices and node_j in original_indices:
                expanded_ybus = expanded_ybus.at[i, j].set(ybus[original_indices[node_i], original_indices[node_j]])
    
    return expanded_ybus

def Volt_Ang_node_no_PU() -> pd.DataFrame:
    """
    Obtains from OpenDSS the voltages and angles per bus or node in a dataFrame with the following columns:
    ['bus_name', 'num_nodes', 'phase_1', 'phase_2', 'phase_3', 'voltage_bus1', 'angle_bus1', 'voltage_bus2',
    'angle_bus2', 'voltage_bus3', 'angle_bus3']

    :return: DF_voltage_angle_node
    """

    bus_names = dss.circuit_all_bus_names()
    Volt_1, Volt_2, Volt_3 = list(), list(), list()
    angle_bus1, angle_bus2, angle_bus3 = list(), list(), list()
    bus_terminal1, bus_terminal2, bus_terminal3 = list(), list(), list()
    bus_name, num_nodes = list(), list()
    node_3, node_2, node_1 = 0, 0, 0

    for bus in bus_names:
        dss.circuit_set_active_bus(bus)
        if dss.bus_num_nodes() == 1:
            node_1 += 1
            num_nodes.append(dss.bus_num_nodes())
            bus_name.append(bus)
            if dss.bus_nodes()[0] == 1:
                bus_terminal1.append(1)
                bus_terminal2.append(0)
                bus_terminal3.append(0)
                Volt_1.append(dss.bus_vmag_angle()[0])
                angle_bus1.append(dss.bus_vmag_angle()[1])
                Volt_2.append(0)
                angle_bus2.append(0)
                Volt_3.append(0)
                angle_bus3.append(0)
            elif dss.bus_nodes()[0] == 2:
                bus_terminal1.append(0)
                bus_terminal2.append(1)
                bus_terminal3.append(0)
                Volt_1.append(0)
                angle_bus1.append(0)
                Volt_2.append(dss.bus_vmag_angle()[0])
                angle_bus2.append(dss.bus_vmag_angle()[1])
                Volt_3.append(0)
                angle_bus3.append(0)
            elif dss.bus_nodes()[0] == 3:
                bus_terminal1.append(0)
                bus_terminal2.append(0)
                bus_terminal3.append(1)
                Volt_1.append(0)
                angle_bus1.append(0)
                Volt_2.append(0)
                angle_bus2.append(0)
                Volt_3.append(dss.bus_vmag_angle()[0])
                angle_bus3.append(dss.bus_vmag_angle()[1])

        elif dss.bus_num_nodes() == 2:
            node_2 += 1
            num_nodes.append(dss.bus_num_nodes())
            bus_name.append(bus)
            if dss.bus_nodes()[0] == 1 and dss.bus_nodes()[1] == 2:
                bus_terminal1.append(1)
                bus_terminal2.append(1)
                bus_terminal3.append(0)
                Volt_1.append(dss.bus_vmag_angle()[0])
                angle_bus1.append(dss.bus_vmag_angle()[1])
                Volt_2.append(dss.bus_vmag_angle()[2])
                angle_bus2.append(dss.bus_vmag_angle()[3])
                Volt_3.append(0)
                angle_bus3.append(0)

            elif dss.bus_nodes()[0] == 1 and dss.bus_nodes()[1] == 3:
                bus_terminal1.append(1)
                bus_terminal2.append(0)
                bus_terminal3.append(1)
                Volt_1.append(dss.bus_vmag_angle()[0])
                angle_bus1.append(dss.bus_vmag_angle()[1])
                Volt_2.append(0)
                angle_bus2.append(0)
                Volt_3.append(dss.bus_vmag_angle()[2])
                angle_bus3.append(dss.bus_vmag_angle()[3])

            elif dss.bus_nodes()[0] == 2 and dss.bus_nodes()[1] == 3:
                bus_terminal1.append(0)
                bus_terminal2.append(1)
                bus_terminal3.append(1)
                Volt_1.append(0)
                angle_bus1.append(0)
                Volt_2.append(dss.bus_vmag_angle()[0])
                angle_bus2.append(dss.bus_vmag_angle()[1])
                Volt_3.append(dss.bus_vmag_angle()[2])
                angle_bus3.append(dss.bus_vmag_angle()[3])

        elif dss.bus_num_nodes() == 3:
            node_3 += 1
            num_nodes.append(dss.bus_num_nodes())
            bus_name.append(bus)
            bus_terminal1.append(1)
            bus_terminal2.append(1)
            bus_terminal3.append(1)
            Volt_1.append(dss.bus_vmag_angle()[0])
            angle_bus1.append(dss.bus_vmag_angle()[1])
            Volt_2.append(dss.bus_vmag_angle()[2])
            angle_bus2.append(dss.bus_vmag_angle()[3])
            Volt_3.append(dss.bus_vmag_angle()[4])
            angle_bus3.append(dss.bus_vmag_angle()[5])

        elif dss.bus_num_nodes() == 4:
            node_3 += 1
            num_nodes.append(dss.bus_num_nodes())
            bus_name.append(bus)
            bus_terminal1.append(1)
            bus_terminal2.append(1)
            bus_terminal3.append(1)
            Volt_1.append(dss.bus_vmag_angle()[0])
            angle_bus1.append(dss.bus_vmag_angle()[1])
            Volt_2.append(dss.bus_vmag_angle()[2])
            angle_bus2.append(dss.bus_vmag_angle()[3])
            Volt_3.append(dss.bus_vmag_angle()[4])
            angle_bus3.append(dss.bus_vmag_angle()[5])

    Volt_1 = [x / 1000 for x in Volt_1]
    Volt_2 = [x / 1000 for x in Volt_2]
    Volt_3 = [x / 1000 for x in Volt_3]

    voltage_angle_list = list(
        zip(bus_name, num_nodes, bus_terminal1, bus_terminal2, bus_terminal3, Volt_1, angle_bus1,
            Volt_2, angle_bus2, Volt_3, angle_bus3))

    DF_voltage_angle_node = pd.DataFrame(
        voltage_angle_list, columns=['bus_name', 'num_nodes', 'ph_1', 'ph_2', 'ph_3',
                                        'V1(kV)', 'Ang1(deg)', 'V2(kV)', 'Ang2(deg)', 'V3(kV)', 'Ang3(deg)'])

    return DF_voltage_angle_node


def Volt_Ang_node_PU() -> pd.DataFrame:
    """
    Convert the result of function Volt_Ang_node_no_PU to value per unit

    :param dss: COM interface between OpenDSS and Python -> Electrical Network
    :param DF_Vmag_Ang_no_PU: Result of function Volt_Ang_node_no_PU(dss)
    :return:DF_Vmag_Ang_no_PU
    """

    DF_Vmag_Ang_PU = Volt_Ang_node_no_PU()

    for n in range(len(DF_Vmag_Ang_PU)):
        dss.circuit_set_active_bus(DF_Vmag_Ang_PU['bus_name'][n])
        kVBas = dss.bus_kv_base()
        'Voltage'
        DF_Vmag_Ang_PU.at[n, 'V1(kV)'] = DF_Vmag_Ang_PU.at[n, 'V1(kV)'] / (kVBas)
        DF_Vmag_Ang_PU.at[n, 'V2(kV)'] = DF_Vmag_Ang_PU.at[n, 'V2(kV)'] / (kVBas)
        DF_Vmag_Ang_PU.at[n, 'V3(kV)'] = DF_Vmag_Ang_PU.at[n, 'V3(kV)'] / (kVBas)

    DF_Vmag_Ang_PU = DF_Vmag_Ang_PU.rename(
        columns={'V1(kV)': 'V1(pu)', 'V2(kV)': 'V2(pu)', 'V3(kV)': 'V3(pu)'})

    return DF_Vmag_Ang_PU

def element_PQij_PU(df_element_power: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the output of the function element_powers_PQij to values per unit.

    :param SbasMVA_3ph: Circuit Base Power
    :param df_element_power: It comes from the function: element_powers_PQij
    :return: df_element_power
    """
    # Note: 1. For 4 bus system, sbase = 40.33 MVA
    Sbas_1ph = 1#40.33 # MVA

    for k in range(len(df_element_power)):
        if df_element_power.at[k, 'from_bus'] != '' and df_element_power.at[k, 'to_bus'] == '':
            df_element_power.at[k, 'P1(kW)'] = ((df_element_power.at[k, 'P1(kW)']) * -1) / (Sbas_1ph)
            df_element_power.at[k, 'P2(kW)'] = ((df_element_power.at[k, 'P2(kW)']) * -1) / (Sbas_1ph)
            df_element_power.at[k, 'P3(kW)'] = ((df_element_power.at[k, 'P3(kW)']) * -1) / (Sbas_1ph)

            df_element_power.at[k, 'Q1(kvar)'] = ((df_element_power.at[k, 'Q1(kvar)']) * -1) / (Sbas_1ph)
            df_element_power.at[k, 'Q2(kvar)'] = ((df_element_power.at[k, 'Q2(kvar)']) * -1) / (Sbas_1ph)
            df_element_power.at[k, 'Q3(kvar)'] = ((df_element_power.at[k, 'Q3(kvar)']) * -1) / (Sbas_1ph)

        elif df_element_power.at[k, 'from_bus'] != '' and df_element_power.at[k, 'to_bus'] != '':
            df_element_power.at[k, 'P1(kW)'] = ((df_element_power.at[k, 'P1(kW)']) * 1) / (Sbas_1ph)
            df_element_power.at[k, 'P2(kW)'] = ((df_element_power.at[k, 'P2(kW)']) * 1) / (Sbas_1ph)
            df_element_power.at[k, 'P3(kW)'] = ((df_element_power.at[k, 'P3(kW)']) * 1) / (Sbas_1ph)

            df_element_power.at[k, 'Q1(kvar)'] = ((df_element_power.at[k, 'Q1(kvar)']) * 1)/ (Sbas_1ph)
            df_element_power.at[k, 'Q2(kvar)'] = ((df_element_power.at[k, 'Q2(kvar)']) * 1) / (Sbas_1ph)
            df_element_power.at[k, 'Q3(kvar)'] = ((df_element_power.at[k, 'Q3(kvar)']) * 1) / (Sbas_1ph)

    # df_element_power = df_element_power.rename(
    #     columns={'P1(kW)': 'P1(pu)', 'P2(kW)': 'P2(pu)', 'P3(kW)': 'P3(pu)',
    #                 'Q1(kvar)': 'Q1(pu)', 'Q2(kvar)': 'Q2(pu)', 'Q3(kvar)': 'Q3(pu)'})

    return df_element_power

def element_powers_PQij(self, element) -> pd.DataFrame:
    """
    Obtains from OpenDSS the active and reactive powers according to list(element) in a dataFrame with the following
    columns: ['element_name', 'num_phases', 'num_cond', 'conn', 'from_bus', 'to_bus', 'bus1', 'bus2', 'phase_1',
    'phase_2', 'phase_3', 'P1(kW)', 'Q1(kvar)', 'P2(kW)', 'Q2(kvar)', 'P3(kW)', 'Q3(kvar)']

    :param dss: COM interface between OpenDSS and Python: Electrical Network
    :param element: list['vsources', 'transformers', 'lines', 'loads', 'capacitors']
    :return: DF_element_PQ
    """
    element_name_list = list()
    num_cond_list = list()
    num_phases_list = list()
    conn_list = list()
    from_bus_list = list()
    to_bus_list = list()
    element_bus1_list = list()
    element_bus2_list = list()
    element_terminal1_list = list()
    element_terminal2_list = list()
    element_terminal3_list = list()
    P1, P2, P3 = list(), list(), list()
    Q1, Q2, Q3 = list(), list(), list()

    for ii in element:
        if ii == 'vsources':
            num_element = dss.vsources_count()
            dss.vsources_first()
        elif ii == 'transformers':
            num_element = dss.transformers_count()
            dss.transformers_first()
        elif ii == 'lines':
            num_element = dss.lines_count()
            dss.lines_first()
        elif ii == 'loads':
            num_element = dss.loads_count()
            dss.loads_first()
        elif ii == 'capacitors':
            num_element = dss.capacitors_count()
            dss.capacitors_first()

        for num in range(num_element):
            if ii == 'loads':
                bus2 = ''
                if dss.loads_read_is_delta() == 1:
                    conn = 'delta'
                else:
                    conn = 'wye'
            else:
                bus2 = dss.cktelement_read_bus_names()[1]
                conn = ''

            num_cond, node_order, bus1, bus2 = dss.cktelement_num_conductors(), dss.cktelement_node_order(), \
                                                dss.cktelement_read_bus_names()[0], bus2
            from_bus, to_bus = from_to_bus(ii, num_cond, node_order, bus1, bus2)
            element_name_list.append(dss.cktelement_name())
            num_cond_list.append(num_cond)
            num_phases_list.append(dss.cktelement_num_phases())
            element_bus1_list.append(bus1)
            element_bus2_list.append(bus2)
            from_bus_list.append(from_bus)
            to_bus_list.append(to_bus)
            conn_list.append(conn)

            if ii == 'loads':
                if conn == 'delta':
                    if len(dss.cktelement_node_order()) == 1:
                        if dss.cktelement_node_order()[0] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(0);
                            Q2.append(0)
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 2:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 3:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(0);
                            Q2.append(0)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])

                    elif len(dss.cktelement_node_order()) == 2:
                        if dss.cktelement_node_order()[0] == 1 and dss.cktelement_node_order()[1] == 2:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(dss.cktelement_powers()[2]);
                            Q2.append(dss.cktelement_powers()[3])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 1 and dss.cktelement_node_order()[1] == 3:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(0);
                            Q2.append(0)
                            P3.append(dss.cktelement_powers()[2]);
                            Q3.append(dss.cktelement_powers()[3])

                        elif dss.cktelement_node_order()[0] == 2 and dss.cktelement_node_order()[1] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[2]);
                            Q1.append(dss.cktelement_powers()[3])
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 2 and dss.cktelement_node_order()[1] == 3:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(1)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(dss.cktelement_powers()[2]);
                            Q3.append(dss.cktelement_powers()[3])


                        elif dss.cktelement_node_order()[0] == 3 and dss.cktelement_node_order()[1] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])
                            P1.append(dss.cktelement_powers()[2]);
                            Q1.append(dss.cktelement_powers()[3])
                            P2.append(0);
                            Q2.append(0)

                        elif dss.cktelement_node_order()[0] == 3 and dss.cktelement_node_order()[1] == 2:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(1)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])
                            P2.append(dss.cktelement_powers()[2]);
                            Q2.append(dss.cktelement_powers()[3])
                            P1.append(0);
                            Q1.append(0)

                    elif len(dss.cktelement_node_order()) == 3:
                        element_terminal1_list.append(1)
                        element_terminal2_list.append(1)
                        element_terminal3_list.append(1)
                        P1.append(dss.cktelement_powers()[0])
                        Q1.append(dss.cktelement_powers()[1])
                        P2.append(dss.cktelement_powers()[2])
                        Q2.append(dss.cktelement_powers()[3])
                        P3.append(dss.cktelement_powers()[4])
                        Q3.append(dss.cktelement_powers()[5])

                elif conn == 'wye':

                    if len(dss.cktelement_node_order()) == 2:
                        if dss.cktelement_node_order()[0] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(0);
                            Q2.append(0)
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 2:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 3:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(0);
                            Q2.append(0)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])

                    elif len(dss.cktelement_node_order()) == 3:
                        if dss.cktelement_node_order()[0] == 1 and dss.cktelement_node_order()[1] == 2:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(dss.cktelement_powers()[2]);
                            Q2.append(dss.cktelement_powers()[3])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 1 and dss.cktelement_node_order()[1] == 3:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(0);
                            Q2.append(0)
                            P3.append(dss.cktelement_powers()[2]);
                            Q3.append(dss.cktelement_powers()[3])

                        elif dss.cktelement_node_order()[0] == 2 and dss.cktelement_node_order()[1] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P1.append(dss.cktelement_powers()[2]);
                            Q1.append(dss.cktelement_powers()[3])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 2 and dss.cktelement_node_order()[1] == 3:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(1)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(dss.cktelement_powers()[2]);
                            Q3.append(dss.cktelement_powers()[3])
                            P1.append(0);
                            Q1.append(0)

                        elif dss.cktelement_node_order()[0] == 3 and dss.cktelement_node_order()[1] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])
                            P1.append(dss.cktelement_powers()[2]);
                            Q1.append(dss.cktelement_powers()[3])
                            P2.append(0);
                            Q2.append(0)

                        elif dss.cktelement_node_order()[0] == 3 and dss.cktelement_node_order()[1] == 2:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(1)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])
                            P2.append(dss.cktelement_powers()[2]);
                            Q2.append(dss.cktelement_powers()[3])
                            P1.append(0);
                            Q1.append(0)

                    elif len(dss.cktelement_node_order()) == 4:
                        element_terminal1_list.append(1)
                        element_terminal2_list.append(1)
                        element_terminal3_list.append(1)
                        P1.append(dss.cktelement_powers()[0]);
                        Q1.append(dss.cktelement_powers()[1])
                        P2.append(dss.cktelement_powers()[2]);
                        Q2.append(dss.cktelement_powers()[3])
                        P3.append(dss.cktelement_powers()[4]);
                        Q3.append(dss.cktelement_powers()[5])


            else:
                if conn == '':
                    if dss.cktelement_num_phases() == 1:
                        if dss.cktelement_node_order()[0] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(0);
                            Q2.append(0)
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 2:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 3:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(0);
                            Q2.append(0)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])

                    elif dss.cktelement_num_phases() == 2:
                        if dss.cktelement_node_order()[0] == 1 and dss.cktelement_node_order()[1] == 2:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(dss.cktelement_powers()[2]);
                            Q2.append(dss.cktelement_powers()[3])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 1 and dss.cktelement_node_order()[1] == 3:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P1.append(dss.cktelement_powers()[0]);
                            Q1.append(dss.cktelement_powers()[1])
                            P2.append(0);
                            Q2.append(0)
                            P3.append(dss.cktelement_powers()[2]);
                            Q3.append(dss.cktelement_powers()[3])

                        elif dss.cktelement_node_order()[0] == 2 and dss.cktelement_node_order()[1] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(0)
                            P1.append(dss.cktelement_powers()[2]);
                            Q1.append(dss.cktelement_powers()[3])
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(0);
                            Q3.append(0)

                        elif dss.cktelement_node_order()[0] == 2 and dss.cktelement_node_order()[1] == 3:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(1)
                            P1.append(0);
                            Q1.append(0)
                            P2.append(dss.cktelement_powers()[0]);
                            Q2.append(dss.cktelement_powers()[1])
                            P3.append(dss.cktelement_powers()[2]);
                            Q3.append(dss.cktelement_powers()[3])


                        elif dss.cktelement_node_order()[0] == 3 and dss.cktelement_node_order()[1] == 1:
                            element_terminal1_list.append(1)
                            element_terminal2_list.append(0)
                            element_terminal3_list.append(1)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])
                            P1.append(dss.cktelement_powers()[2]);
                            Q1.append(dss.cktelement_powers()[3])
                            P2.append(0);
                            Q2.append(0)

                        elif dss.cktelement_node_order()[0] == 3 and dss.cktelement_node_order()[1] == 2:
                            element_terminal1_list.append(0)
                            element_terminal2_list.append(1)
                            element_terminal3_list.append(1)
                            P3.append(dss.cktelement_powers()[0]);
                            Q3.append(dss.cktelement_powers()[1])
                            P2.append(dss.cktelement_powers()[2]);
                            Q2.append(dss.cktelement_powers()[3])
                            P1.append(0);
                            Q1.append(0)

                    elif dss.cktelement_num_phases() == 3:
                        element_terminal1_list.append(1)
                        element_terminal2_list.append(1)
                        element_terminal3_list.append(1)
                        P1.append(dss.cktelement_powers()[0])
                        Q1.append(dss.cktelement_powers()[1])
                        P2.append(dss.cktelement_powers()[2])
                        Q2.append(dss.cktelement_powers()[3])
                        P3.append(dss.cktelement_powers()[4])
                        Q3.append(dss.cktelement_powers()[5])

                    elif dss.cktelement_num_phases() == 4:
                        element_terminal1_list.append(1)
                        element_terminal2_list.append(1)
                        element_terminal3_list.append(1)
                        P1.append(dss.cktelement_powers()[0])
                        Q1.append(dss.cktelement_powers()[1])
                        P2.append(dss.cktelement_powers()[2])
                        Q2.append(dss.cktelement_powers()[3])
                        P3.append(dss.cktelement_powers()[4])
                        Q3.append(dss.cktelement_powers()[5])

            if ii == 'transformers':
                dss.transformers_next()
            elif ii == 'lines':
                dss.lines_next()
            elif ii == 'loads':
                dss.loads_next()
            elif ii == 'capacitors':
                dss.capacitors_next()
            elif ii == 'vsources':
                dss.vsources_next()

    element_PQ_list = list(
        zip(element_name_list, num_phases_list, num_cond_list, conn_list, from_bus_list, to_bus_list,
            element_bus1_list, element_bus2_list, element_terminal1_list, element_terminal2_list,
            element_terminal3_list,
            P1, Q1, P2, Q2, P3, Q3))

    DF_element_PQ = pd.DataFrame(element_PQ_list,
                                    columns=['element_name', 'num_ph', 'num_cond', 'conn', 'from_bus', 'to_bus',
                                            'bus1', 'bus2', 'ph_1', 'ph_2', 'ph_3',
                                            'P1(kW)', 'Q1(kvar)', 'P2(kW)', 'Q2(kvar)', 'P3(kW)', 'Q3(kvar)'])

    return DF_element_PQ

def from_to_bus(type_element, num_cond, node_order, bus1, bus2):
    if type_element == 'loads':
        if num_cond == 1:
            aux1 = bus1.find('.' + str(node_order[0]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]
            to_bus = ''
        elif num_cond == 2:
            aux1 = bus1.find('.' + str(node_order[0]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]
            to_bus = ''
        elif num_cond == 3:
            aux1 = bus1.find('.' + str(node_order[0]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]
            to_bus = ''
        elif num_cond == 4:
            aux1 = bus1.find('.' + str(node_order[0]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]
            to_bus = ''

    else:
        if num_cond == 1:
            aux1 = bus1.find('.' + str(node_order[0]))
            aux2 = bus2.find('.' + str(node_order[1]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]

            if aux2 == -1:
                to_bus = bus2
            else:
                to_bus = bus2[:aux2]
        elif num_cond == 2:
            aux1 = bus1.find('.' + str(node_order[0]))
            aux2 = bus2.find('.' + str(node_order[2]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]

            if aux2 == -1:
                to_bus = bus2
            else:
                to_bus = bus2[:aux2]

        elif num_cond == 3:
            aux1 = bus1.find('.' + str(node_order[0]))
            aux2 = bus2.find('.' + str(node_order[3]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]

            if aux2 == -1:
                to_bus = bus2
            else:
                to_bus = bus2[:aux2]
        elif num_cond == 4:
            aux1 = bus1.find('.' + str(node_order[0]))
            aux2 = bus2.find('.' + str(node_order[4]))
            if aux1 == -1:
                from_bus = bus1
            else:
                from_bus = bus1[:aux1]
            if aux2 == -1:
                to_bus = bus2
            else:
                to_bus = bus2[:aux2]

    return from_bus, to_bus
 
# Power injection measurements from opendss
def PQ_i(self) -> pd.DataFrame:
    df_PQi=element_powers_PQij(self, element=['vsources', 'transformers', 'lines', 'loads', 'capacitors'])
    PQi=element_PQij_PU(df_PQi)
    PQi = PQi[PQi['element_name'].str.contains('Vsource.source|Load')]
    PQi=PQi.drop(columns=['bus1','bus2','element_name','num_ph','num_cond','conn','to_bus'])
    PQi=PQi.rename(columns={'from_bus':'Bus'})

    modified_df = PQi.groupby('Bus').agg({
    'ph_1': 'max',  # Assuming a bus has phases 1, 2, and 3
    'ph_2': 'max',  # Aggregate to make sure we get the maximum value if any phase is present
    'ph_3': 'max',
    'P1(kW)': 'sum',  # Summing the P values for each phase
    'Q1(kvar)': 'sum',
    'P2(kW)': 'sum',
    'Q2(kvar)': 'sum',
    'P3(kW)': 'sum',
    'Q3(kvar)': 'sum'
    }).reset_index()

    bus_nodes_dict = dict(zip(buses, Bus_Nodes))

# Initialize an empty list to store rows for the final dataframe
    final_rows = []

# Loop through the buses and rearrange/add missing buses
    for bus in buses:
        # Check if the bus is present in the current dataframe
        if bus in modified_df['Bus'].values:
            # If the bus is present, add its row
            row = modified_df[modified_df['Bus'] == bus].iloc[0].to_dict()
            final_rows.append(row)
        else:
            # If the bus is missing, create a new row with zero power values and correct phases
            nodes = bus_nodes_dict[bus]
            new_row = {
                'Bus': bus,
                'ph_1': 1 if 1 in nodes else 0,
                'ph_2': 1 if 2 in nodes else 0,
                'ph_3': 1 if 3 in nodes else 0,
                'P1(kW)': 0.0,
                'Q1(kvar)': 0.0,
                'P2(kW)': 0.0,
                'Q2(kvar)': 0.0,
                'P3(kW)': 0.0,
                'Q3(kvar)': 0.0
            }
            final_rows.append(new_row)

    # Create a new dataframe from the rearranged/modified rows
        final_df = pd.DataFrame(final_rows)

    return final_df

# Voltage Magnitude from OpenDSS
def Vi() -> pd.DataFrame:
    df_Vi=Volt_Ang_node_no_PU()
    df_Vi=df_Vi.drop(columns=['Ang1(deg)','Ang2(deg)','Ang3(deg)'])
    return df_Vi

def Vi_pu() -> pd.DataFrame:
    df_Vi=Volt_Ang_node_PU()
    df_Vi=df_Vi.drop(columns=['Ang1(deg)','Ang2(deg)','Ang3(deg)'])
    return df_Vi

# Voltage angle from OpenDSS
def Angi() -> pd.DataFrame:
    df_Vi=Volt_Ang_node_PU()
    df_Vi=df_Vi.drop(columns=['V1(pu)','V2(pu)','V3(pu)'])
    return df_Vi

# Z measurement vector
def Z_meas(self,nvi,npQi):

    # Voltage Magnitude measurement (Type 1)
    V=Vi()
    V_sample=V.sample(n=nvi, random_state=0)
    z1=jnp.array(V_sample[['V1(kV)','V2(kV)','V3(kV)']])

    # Active Power Injection (Type 2)
    PQi=PQ_i(self)
    PQi=PQi.iloc[1:]
    PQ_sample=PQi.sample(n=npQi,random_state=0)
    indices=PQ_sample.index
    indices=jnp.array(indices)
    indices=indices-1
    z2=jnp.array(PQ_sample[['P1(kW)','P2(kW)','P3(kW)']])

    # Reactive Power Injection (Type 3)
    z3=jnp.array(PQ_sample[['Q1(kvar)','Q2(kvar)','Q3(kvar)']])

    zmeas=jnp.concatenate((z1, z2, z3), 0)

    return zmeas,indices


def Z_meas_pu(self,nvi,npQi):

    # Voltage Magnitude measurement (Type 1)
    V=Vi_pu()
    V_sample=V.sample(n=nvi, random_state=0)
    z1=jnp.array(V_sample[['V1(pu)','V2(pu)','V3(pu)']])

    # Active Power Injection (Type 2)
    PQi=PQ_i(self)
    PQi=PQi.iloc[1:]
    PQ_sample=PQi.sample(n=npQi,random_state=0)
    indices=PQ_sample.index
    indices=jnp.array(indices)
    indices=indices-1
    z2=jnp.array(PQ_sample[['P1(pu)','P2(pu)','P3(pu)']])

    # Reactive Power Injection (Type 3)
    z3=jnp.array(PQ_sample[['Q1(pu)','Q2(pu)','Q3(pu)']])

    zmeas=jnp.concatenate((z1, z2, z3), 0)

    return zmeas,indices

# def polar(r,theta):
#     real_part = r * jnp.cos(theta)
#     imaginary_part = r * jnp.sin(theta)
#     complex_number = real_part + 1j*imaginary_part
#     return complex_number

num_phases=3

Y_expand=expand_ybus_general(Y_matrix,nodes,expand_nodes) # (n_buses X n_phases) order

@jit
def flatten_indices(i, ph, offset=0):
    return i * num_phases + ph + offset

# Power injection equations
# @jit
# def P_i_ph(V, delta, i, ph, Y):
#     P = 0.0
#     for l in range(num_phases):
#         for j in range(n_buses):
#             V_i_ph = V[flatten_indices(i, ph)]
#             V_j_l = V[flatten_indices(j, l)]
#             delta_i_ph = delta[flatten_indices(i, ph)]
#             delta_j_l = delta[flatten_indices(j, l)]
#             G_ij_ph_l = Y.real[flatten_indices(i, ph), flatten_indices(j, l)]
#             B_ij_ph_l = Y.imag[flatten_indices(i, ph), flatten_indices(j, l)]
#             P += V_i_ph * V_j_l * (G_ij_ph_l * jnp.cos(delta_i_ph - delta_j_l) + B_ij_ph_l * jnp.sin(delta_i_ph - delta_j_l))*1e3
#     return P

# @jit
# def Q_i_ph(V, delta, i, ph, Y):
#     Q = 0.0
#     for l in range(num_phases):
#         for j in range(n_buses):
#             V_i_ph = V[flatten_indices(i, ph)]
#             V_j_l = V[flatten_indices(j, l)]
#             delta_i_ph = delta[flatten_indices(i, ph)]
#             delta_j_l = delta[flatten_indices(j, l)]
#             G_ij_ph_l = Y.real[flatten_indices(i, ph), flatten_indices(j, l)]
#             B_ij_ph_l = Y.imag[flatten_indices(i, ph), flatten_indices(j, l)]
#             Q += V_i_ph * V_j_l * (G_ij_ph_l * jnp.sin(delta_i_ph - delta_j_l) - B_ij_ph_l * jnp.cos(delta_i_ph - delta_j_l))*1e3
#     return Q

@partial(jit,static_argnums=(5,))
def P_i_ph(V, delta, ii, ph, Y,n_bus):
    V_i_ph = V[flatten_indices(ii, ph)]
    delta_i_ph = delta[flatten_indices(ii, ph)]

    # Reshape V and delta to have dimensions [n_buses, num_phases]
    V_j_l = V.reshape(n_bus,3)  # shape (n_buses, num_phases)
    delta_j_l = delta.reshape(n_bus,3)

    # Extract real and imaginary parts of Y
    G_ij_ph_l = Y.real[flatten_indices(ii, ph), :].reshape(n_bus, 3)
    B_ij_ph_l = Y.imag[flatten_indices(ii, ph), :].reshape(n_bus, 3)

    # Use einsum for vectorized operations
    P = jnp.sum(V_i_ph * V_j_l * (G_ij_ph_l * jnp.cos(delta_i_ph - delta_j_l) + B_ij_ph_l * jnp.sin(delta_i_ph - delta_j_l))) * 1e3
    return P

@partial(jit,static_argnums=(5,))
def Q_i_ph(V, delta, ii, ph, Y, n_bus):
    V_i_ph = V[flatten_indices(ii, ph)]
    delta_i_ph = delta[flatten_indices(ii, ph)]

    # Reshape V and delta to have dimensions [n_buses, num_phases]
    V_j_l = V.reshape(n_bus, 3)
    delta_j_l = delta.reshape(n_bus, 3)

    # Extract real and imaginary parts of Y
    G_ij_ph_l = Y.real[flatten_indices(ii, ph), :].reshape(n_bus, 3)
    B_ij_ph_l = Y.imag[flatten_indices(ii, ph), :].reshape(n_bus, 3)

    # Use einsum for vectorized operations
    Q = jnp.sum(V_i_ph * V_j_l * (G_ij_ph_l * jnp.sin(delta_i_ph - delta_j_l) - B_ij_ph_l * jnp.cos(delta_i_ph - delta_j_l))) * 1e3
    return Q

# @partial(jit,static_argnums=(3,))
# def PQ_i_ph(Volt, Ang, Y, n_bus): # don't take global variables
#     def calc_PQ(i):
#         P_vals = vmap(lambda ph: (P_i_ph(Volt, Ang, i, ph, Y, n_bus)))(jnp.arange(3))
#         Q_vals = vmap(lambda ph: (Q_i_ph(Volt, Ang, i, ph, Y, n_bus)))(jnp.arange(3)) # no. of phase = 3
#         return P_vals, Q_vals

#     P_list, Q_list = zip(*(calc_PQ(i) for i in range(1, n_bus)))

#     P = jnp.stack(P_list)
#     Q = jnp.stack(Q_list)

#     PQ = jnp.concatenate((P, Q), axis=0)
#     return PQ

# Optimized Combined PQ Calculation
@partial(jit, static_argnums=(3,4))
def PQ_i_ph(Volt, ang, Y, n_bus, num_phases=3):
    # Vectorized computation for each bus and phase
    def calc_PQ(ii):
        P_vals = vmap(lambda ph: P_i_ph(Volt, ang, ii, ph, Y, n_bus))(jnp.arange(num_phases))
        Q_vals = vmap(lambda ph: Q_i_ph(Volt, ang, ii, ph, Y, n_bus))(jnp.arange(num_phases))
        return P_vals, Q_vals

    # Use `vmap` to vectorize over the buses
    P_list, Q_list = vmap(calc_PQ)(jnp.arange(1, n_bus))

    # Stack results
    P = jnp.stack(P_list)
    Q = jnp.stack(Q_list)

    # Concatenate P and Q matrices
    PQ = jnp.concatenate((P, Q), axis=0)
    return PQ
    
# 3 phase Voltage & Angle
Voltage= jnp.array(Volt_Ang_node_no_PU()[['V1(kV)','V2(kV)','V3(kV)']]).flatten()
#Voltage_pu= jnp.array(Volt_Ang_node_PU()[['V1(pu)','V2(pu)','V3(pu)']]).flatten()
Angle= jnp.array(Volt_Ang_node_no_PU()[['Ang1(deg)','Ang2(deg)','Ang3(deg)']]).flatten()
Angle=jnp.radians(Angle)

def Z_est(Vsp, theta,Y,nvi, ind, n_bus):

    # Power Injection
    PQ=PQ_i_ph(Vsp,theta,Y, n_bus)     # Range Change
    n=len(PQ)
    
    # Active Power Injection (Type 2)
    Z2 = PQ[:int(n/2)][ind]
    # Reactive Power Injection (Type 3)
    Z3 = PQ[int(n/2):][ind]

    # Voltage Estimation (Type 1)
    V = Vi()
    V_sample=V.sample(n=nvi, random_state=0)
    Z1 = jnp.array(V_sample[['V1(kV)', 'V2(kV)', 'V3(kV)']])   # Change kV <=> pu

    zest=jnp.concatenate((Z1, Z2, Z3), 0)

    return zest

# @jit
# def gradient(Volt,Ang):
#     dPQ_dV = jit(jacobian(lambda V, theta: PQ_i_ph(V,theta), argnums=0))(Volt, Ang)
#     dPQ_dtheta = jit(jacobian(lambda V, theta: PQ_i_ph(V,theta), argnums=1))(Volt, Ang)
#     dPQ_dV=dPQ_dV.reshape((n_buses-1)*3*2,(n_buses*num_phases))
#     dPQ_dtheta=dPQ_dtheta.reshape((n_buses-1)*3*2,(n_buses*num_phases))

#     return dPQ_dV,dPQ_dtheta

total_meas=(n_vi*3)+(6*n_pQi)
rii=jnp.ones(total_meas)*0.01
weights=jnp.diag(1/rii)
Z_MEAS,indx=Z_meas(dss,n_vi,n_pQi)
n_bus_phase=n_buses*num_phases

@partial(jit,static_argnums=(5,))
def loss_func(Vsp,theta,Y,w,ind, n_bus):
    Z_EST=Z_est(Vsp,theta,Y,n_vi,ind,n_bus)
    residual=Z_MEAS-Z_EST
    residual=residual.flatten()
    loss_objective = jnp.dot(residual.T, (w@residual))
    return loss_objective

@partial(jit,static_argnums=(1,2))
def combined_loss_func(V_ang,n,n_bus,w,ind,Y):
    Vsp = V_ang[:n]
    theta = V_ang[n:]
    return loss_func(Vsp, theta,Y,w,ind, n_bus)

# IEEE 13 bus initialization
# Voltage=Voltage.at[Voltage!=0].set(2.401777)
# Voltage=Voltage.at[6:9].set(0.277)

# IEEE 37 bus initialization
# Voltage=Voltage.at[Voltage!=0].set(2.7)
# Voltage=Voltage.at[6:9].set(0.27)

# IEEE 123 bus initialization
Voltage=Voltage.at[Voltage!=0].set(2.40177)
Voltage=Voltage.at[-3:].set(0.277)
initial_guess = jnp.concatenate([Voltage,Angle])  
loss_grad=grad(combined_loss_func, argnums=0)

@partial(jit,static_argnums=(1,2,3,4))
def gradient_descent(initial_params,learning_rate, num_iterations,n,n_bus,w,ind,Y):
    def body_fn(i, params):
        grads = loss_grad(params/1e6,n,n_bus,w,ind,Y)  # Compute the gradient
        params=params - learning_rate * grads * 1e-7 # Update the parameters
        return params # Update the parameters
    
    optimal_params = lax.fori_loop(0, num_iterations, body_fn, initial_params*1e6)
    return optimal_params/1e6

start=time.time()
result= gradient_descent(initial_guess,1e-6,100,n_bus_phase,n_buses,weights,indx,Y_expand)     
end=time.time() 
#result = minimize(combined_loss_func,initial_guess, method='BFGS')  #scipy optimize
final_time=end-start
result=result[result!=0]
Vest=result[:n_nodes]
Aest=result[n_nodes:]
Aest=jnp.rad2deg(Aest)
baseKV=dss.bus_kv_base()
V=Vi() # Change pu <=> kV
V_actual=jnp.array(V[['V1(kV)', 'V2(kV)', 'V3(kV)']])
V_actual=V_actual[V_actual!=0]
Ang=Angi()
A_actual=jnp.array(Ang[['Ang1(deg)', 'Ang2(deg)', 'Ang3(deg)']])
A_actual=A_actual[A_actual!=0]
#A_actual=jnp.deg2rad(A_actual)
V_act=V_actual.tolist()
V_cal=Vest.tolist()
A_cal=Aest.tolist()
A_act=A_actual.tolist()
Volt_Ang_frame=pd.DataFrame({'Nodes':nodes,'V_Actual':V_act,'V_Estimated':V_cal,'Ang_Actual':A_act,'Ang_Estimated':A_cal})
Volt_Ang_frame.to_excel(r'C:\Users\rdutta24\SURI_Project\OpenPy-DSSE\openpy_dsse\volt_ang_frame.xlsx', index=False)


# Note: 1.The step size alpha = 1e-4 for 10% error in measurement
#       2. Step size = 1e-5 for 1% error in measurement
#       3. For 123 bus, adjustment parameter= 1e6 and step size= 1e-6
#       4. For 13 and 37 bus, adjustment parameter = 1e3 and step size = 1e-4

# Volt_Ang_frame.loc[:5,['V_Actual','V_Estimated']]=Volt_Ang_frame.loc[:5,['V_Actual','V_Estimated']]/baseKV
# Volt_Ang_frame.loc[9:,['V_Actual','V_Estimated']]=Volt_Ang_frame.loc[9:,['V_Actual','V_Estimated']]/baseKV
# Volt_Ang_frame.loc[6:8,['V_Actual','V_Estimated']]=Volt_Ang_frame.loc[6:8,['V_Actual','V_Estimated']]/(baseKV/8.67)
###  Rough ####
 
def get_bus_volt_angle(self):
    bus_name = self.Circuit.AllBusNames()
    volt_bus = []
    angle_bus = []

    for bus in bus_name:
        self.Circuit.SetActiveBus(bus)
        volt_angle = self.Bus.VMagAngle()
        volt_bus.extend(volt_angle[0::2])
        angle_bus.extend(volt_angle[1::2])

    volt_bus = jnp.array(volt_bus)
    angle_bus = jnp.radians(jnp.array(angle_bus))

    return volt_bus, angle_bus

# Example usage
volt_bus, angle_bus = get_bus_volt_angle(dss)
print(1)




# Rough Code Snippets
Voltage=Voltage[Voltage!=0]

# e=jnp.array([0,1]).reshape(2,1)
# Y2=jnp.matmul(e,jnp.matmul(e.T,Y_matrix[3:5,3:5]))
# PP=(Y2+jnp.transpose(jnp.conjugate(Y2)))/2
# Vnode=polar(Voltage,Angle)
# v2=Vnode[3:5].reshape(2,1)
# P_2_a=jnp.matmul(PP,jnp.matmul(v2,jnp.transpose(jnp.conjugate(v2))))

# a = [node_complex_power_injection(node_name, Vnode, Y_mat_sp, Y_order) for node_name in Y_order]
# a=jnp.array(a)

def PQ_i_ph(Volt,Ang,Y):
    P=jnp.zeros((n_buses-1,num_phases))
    Q=jnp.zeros((n_buses-1,num_phases))
    for i in range(1,n_buses):
        for ph in range(num_phases):
            P = P.at[i-1, ph].set(P_i_ph(Volt, Ang, i, ph, Y))
            Q = Q.at[i-1, ph].set(Q_i_ph(Volt, Ang, i, ph, Y))
    PQ=jnp.concatenate((P,Q),0)
    return PQ



solver = GradientDescent(fun=combined_loss_func, stepsize=0.01)

# Run the optimization
sol = solver.run(initial_guess)

# Extract optimal Vsp and theta from the solution
optimal_params = sol.params