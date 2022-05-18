#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>


double get_task_ready_time(int *deployed_server, int *execution_order, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id);
double get_task_finish_time(int *deployed_server, int *execution_order, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id);
double get_task_ready_time_no_exec_order(int *deployed_server, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id);
double get_task_finish_time_no_exec_order(int *deployed_server, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id);
double get_transmission_time(double amount, int sender, int receiver, double *B_dd, double B_edge_up, double B_edge_down, double B_cloud_up, double B_cloud_down, double memory_bandwidth, int num_servers, int edge_id, int cloud_id);
bool mapToDict(std::map<std::pair<int, int>, double> &srcMap, PyObject *destDict);
bool dictToMap(PyObject *srcDict, std::map<std::pair<int, int>, double> &destMap);
bool dictToMap(PyObject *srcDict, std::map<int, std::vector<int>> &destMap);
std::vector<int> listTupleToVector(PyObject* incoming);
bool is_done(double *array, int size);


PyObject* get_completion_time(PyObject* self, PyObject* args) {
    Py_Initialize();
    import_array();

    // initialize
    int num_partitions;
    PyObject *py_deployed_server, *py_execution_order, *py_node_weight, *py_edge_weight, *py_partition_predecessor, *py_partition_successor;
    if (!PyArg_ParseTuple(args, "iOOOOOO", &num_partitions, &py_deployed_server, &py_execution_order, &py_node_weight, &py_edge_weight, &py_partition_predecessor, &py_partition_successor)) {
        std::cout << "PyArg_ParseTuple Error" << std::endl;
        return NULL;
    }
    int *deployed_server = (int*)PyArray_DATA(py_deployed_server);
    int *execution_order = (int*)PyArray_DATA(py_execution_order);
    double *node_weight = (double*)PyArray_DATA(py_node_weight);
    std::map<std::pair<int, int>, double> edge_weight;
    if (!dictToMap(py_edge_weight, edge_weight)) {
        std::cout << "dictToMap Error" << std::endl;
        return NULL;
    }
    std::map<int, std::vector<int>> partition_predecessor;
    if (!dictToMap(py_partition_predecessor, partition_predecessor)) {
        std::cout << "listToVector Error" << std::endl;
        return NULL;
    }
    std::map<int, std::vector<int>> partition_successor;
    if (!dictToMap(py_partition_successor, partition_successor)) {
        std::cout << "listToVector Error" << std::endl;
        return NULL;
    }
    Py_DECREF(&py_deployed_server);
    Py_DECREF(&py_execution_order);
    Py_DECREF(&py_node_weight);
    Py_DECREF(&py_edge_weight);
    Py_DECREF(&py_partition_predecessor);
    Py_DECREF(&py_partition_successor);
    double *ready_time = new double[num_partitions]();
    double *finish_time = new double[num_partitions]();

    // completion time calculation
    for (int c_id = 0; c_id < num_partitions; c_id++) {
        if (partition_predecessor[c_id].size() == 0) {
            ready_time[c_id] = get_task_ready_time(deployed_server, execution_order, node_weight, edge_weight, partition_predecessor, partition_successor, ready_time, finish_time, num_partitions, c_id);
        }
    }

    for (int n = 0; n < num_partitions; n++) {
        int first_order = 2147483647;
        int target_c_id = 0;

        for (int c_id = 0; c_id < num_partitions; c_id++) {
            if (ready_time[c_id] > 0 && finish_time[c_id] == 0 && first_order > execution_order[c_id]) {
                first_order = execution_order[c_id];
                target_c_id = c_id;
            }
        }

        finish_time[target_c_id] = get_task_finish_time(deployed_server, execution_order, node_weight, edge_weight, partition_predecessor, partition_successor, ready_time, finish_time, num_partitions, target_c_id);
        for (int succ_id : partition_successor[target_c_id]) {
            bool prepared = true;
            for (int pred_id : partition_predecessor[succ_id]) {
                if (finish_time[pred_id] == 0) {
                    prepared = false;
                }
            }
            if (prepared) {
                ready_time[succ_id] = get_task_ready_time(deployed_server, execution_order, node_weight, edge_weight, partition_predecessor, partition_successor, ready_time, finish_time, num_partitions, succ_id);
            }
        }
    }

    npy_intp m = num_partitions;
    return PyArray_SimpleNewFromData(1, &m, NPY_DOUBLE, (void *)finish_time);
}

double get_task_ready_time(int *deployed_server, int *execution_order, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id) {
    double TR_n = 0, TF_p = 0, T_tr = 0;
    if (partition_predecessor[c_id].size() == 0) {
        TR_n = edge_weight[std::make_pair(c_id, c_id)];
    }
    else {
        for (int pred_id : partition_predecessor[c_id]) {
            if (finish_time[pred_id] > 0) {
                TF_p = finish_time[pred_id];
            }
            else {
                std::cout << "Error: predecessor not finish, #" << c_id << std::endl;
                return -1;
            }
            T_tr = edge_weight[std::make_pair(pred_id, c_id)];
            TR_n = std::max(TF_p + T_tr, TR_n);
        }
    }
    ready_time[c_id] = TR_n;
    return ready_time[c_id];
}

double get_task_finish_time(int *deployed_server, int *execution_order, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id) {
    double TR_n = 0, T_cp = 0;
    if (ready_time[c_id] == 0) {
        std::cout << "Error: partition not ready, #" << c_id << std::endl;
        return -1;
    }
    TR_n = ready_time[c_id];

    for (int p_id = 0; p_id < num_partitions; p_id++) {
        if (deployed_server[c_id] == deployed_server[p_id]) {
            TR_n = std::max(TR_n, finish_time[p_id]);
        }
    }
    T_cp = node_weight[c_id];
    finish_time[c_id] = TR_n + T_cp;
    return finish_time[c_id];
}



PyObject* get_completion_time_no_exec_order(PyObject* self, PyObject* args) {
    Py_Initialize();
    import_array();

    // initialize
    int num_partitions;
    PyObject *py_deployed_server, *py_node_weight, *py_edge_weight, *py_partition_predecessor, *py_partition_successor;
    if (!PyArg_ParseTuple(args, "iOOOOO", &num_partitions, &py_deployed_server, &py_node_weight, &py_edge_weight, &py_partition_predecessor, &py_partition_successor)) {
        std::cout << "PyArg_ParseTuple Error" << std::endl;
        return NULL;
    }
    int *deployed_server = (int*)PyArray_DATA(py_deployed_server);
    double *node_weight = (double*)PyArray_DATA(py_node_weight);
    std::map<std::pair<int, int>, double> edge_weight;
    if (!dictToMap(py_edge_weight, edge_weight)) {
        std::cout << "dictToMap Error" << std::endl;
        return NULL;
    }
    std::map<int, std::vector<int>> partition_predecessor;
    if (!dictToMap(py_partition_predecessor, partition_predecessor)) {
        std::cout << "listToVector Error" << std::endl;
        return NULL;
    }
    std::map<int, std::vector<int>> partition_successor;
    if (!dictToMap(py_partition_successor, partition_successor)) {
        std::cout << "listToVector Error" << std::endl;
        return NULL;
    }
    Py_DECREF(&py_deployed_server);
    Py_DECREF(&py_node_weight);
    Py_DECREF(&py_edge_weight);
    Py_DECREF(&py_partition_predecessor);
    Py_DECREF(&py_partition_successor);
    double *ready_time = new double[num_partitions]();
    double *finish_time = new double[num_partitions]();

    // completion time calculation
    for (int c_id = 0; c_id < num_partitions; c_id++)
        if (partition_predecessor[c_id].size() == 0)
            ready_time[c_id] = get_task_ready_time_no_exec_order(deployed_server, node_weight, edge_weight, partition_predecessor, partition_successor, ready_time, finish_time, num_partitions, c_id);

    for (int n = 0; n < num_partitions; n++) {
        int target_c_id = 0;

        for (int c_id = 0; c_id < num_partitions; c_id++) {
            if (ready_time[c_id] > 0 && finish_time[c_id] == 0) {
                target_c_id = c_id;
            }
        }

        finish_time[target_c_id] = get_task_finish_time_no_exec_order(deployed_server, node_weight, edge_weight, partition_predecessor, partition_successor, ready_time, finish_time, num_partitions, target_c_id);
        for (int succ_id : partition_successor[target_c_id]) {
            bool prepared = true;
            for (int pred_id : partition_predecessor[succ_id]) {
                if (finish_time[pred_id] == 0) {
                    prepared = false;
                }
            }
            if (prepared) {
                ready_time[succ_id] = get_task_ready_time_no_exec_order(deployed_server, node_weight, edge_weight, partition_predecessor, partition_successor, ready_time, finish_time, num_partitions, succ_id);
            }
        }
    }

    npy_intp m = num_partitions;
    return PyArray_SimpleNewFromData(1, &m, NPY_DOUBLE, (void *)finish_time);
}

double get_task_ready_time_no_exec_order(int *deployed_server, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id) {
    double TR_n = 0, TF_p = 0, T_tr = 0;
    if (partition_predecessor[c_id].size() == 0) {
        TR_n = edge_weight[std::make_pair(c_id, c_id)];
    }
    else {
        for (int pred_id : partition_predecessor[c_id]) {
            if (finish_time[pred_id] > 0) {
                TF_p = finish_time[pred_id];
            }
            else {
                std::cout << "Error: predecessor not finish, #" << c_id << std::endl;
                return -1;
            }
            T_tr = edge_weight[std::make_pair(pred_id, c_id)];
            TR_n = std::max(TF_p + T_tr, TR_n);
        }
    }
    ready_time[c_id] = TR_n;
    return ready_time[c_id];
}

double get_task_finish_time_no_exec_order(int *deployed_server, double *node_weight,
                            std::map<std::pair<int, int>, double> &edge_weight,
                            std::map<int, std::vector<int>> &partition_predecessor,
                            std::map<int, std::vector<int>> &partition_successor,
                            double *ready_time, double *finish_time, int num_partitions, int c_id) {
    double TR_n = 0, T_cp = 0;
    if (ready_time[c_id] == 0) {
        std::cout << "Error: partition not ready, #" << c_id << std::endl;
        return -1;
    }
    TR_n = ready_time[c_id];
    T_cp = node_weight[c_id];
    finish_time[c_id] = TR_n + T_cp;
    return finish_time[c_id];
}



PyObject* get_edge_weight(PyObject* self, PyObject* args) {
    Py_Initialize();
    import_array();

    // initialize
    int num_servers, edge_id, cloud_id;
    double B_edge_up, B_edge_down, B_cloud_up, B_cloud_down, memory_bandwidth;
    PyObject *py_deployed_server, *py_request_device, *py_input_data_size, *py_B_dd;
    if (!PyArg_ParseTuple(args, "iiidddddOOOO", &num_servers, &edge_id, &cloud_id, &B_edge_up, &B_edge_down, &B_cloud_up, &B_cloud_down, &memory_bandwidth, &py_deployed_server, &py_request_device, &py_input_data_size, &py_B_dd)) {
        std::cout << "PyArg_ParseTuple Error" << std::endl;
        return NULL;
    }
    int *deployed_server = (int*)PyArray_DATA(py_deployed_server);
    double *request_device = (double*)PyArray_DATA(py_request_device);
    double *B_dd = (double*)PyArray_DATA(py_B_dd);
    std::map<std::pair<int, int>, double> input_data_size;
    if (!dictToMap(py_input_data_size, input_data_size)) {
        std::cout << "dictToMap Error" << std::endl;
        return NULL;
    }
    Py_DECREF(&py_deployed_server);
    Py_DECREF(&py_input_data_size);
    Py_DECREF(&py_B_dd);

    // completion time calculation
    std::map<std::pair<int, int>, double> edge_weight;
    auto iter = input_data_size.begin();
    while (iter != input_data_size.end()) {
        if (iter->first.first == iter->first.second) {
            edge_weight[iter->first] = get_transmission_time(iter->second, request_device[iter->first.first], deployed_server[iter->first.first], B_dd, B_edge_up, B_edge_down, B_cloud_up, B_cloud_down, memory_bandwidth, num_servers, edge_id, cloud_id);
        }
        else {
            edge_weight[iter->first] = get_transmission_time(iter->second, deployed_server[iter->first.first], deployed_server[iter->first.second], B_dd, B_edge_up, B_edge_down, B_cloud_up, B_cloud_down, memory_bandwidth, num_servers, edge_id, cloud_id);
        }
        ++iter;
    }
    
    PyObject *py_edge_weight = PyDict_New();
    mapToDict(edge_weight, py_edge_weight);
    return py_edge_weight;
}

double get_transmission_time(double amount, int sender, int receiver, double *B_dd, double B_edge_up, double B_edge_down, double B_cloud_up, double B_cloud_down, double memory_bandwidth, int num_servers, int edge_id, int cloud_id) {
    if (sender == receiver) {
        return amount / memory_bandwidth;
    }
    else if (sender == cloud_id) {
        return amount / B_cloud_down;
    }
    else if (receiver == cloud_id) {
        return amount / B_cloud_up;
    }
    else if (sender == edge_id) {
        return amount / B_edge_down;
    }
    else if (receiver == edge_id) {
        return amount / B_edge_up;
    }
    else {
        return amount / B_dd[sender * num_servers + receiver];
    }
}



bool mapToDict(std::map<std::pair<int, int>, double> &srcMap, PyObject *destDict) {
    if(PyDict_Check(destDict)) {
        auto iter = srcMap.begin();
        while (iter != srcMap.end()) {
            PyObject *key = PyTuple_New(2);
            PyTuple_SetItem(key, 0, PyLong_FromLong(iter->first.first));
            PyTuple_SetItem(key, 1, PyLong_FromLong(iter->first.second));
            PyDict_SetItem(destDict, key, PyFloat_FromDouble(iter->second));
            ++iter;
        }
        return true;
    }
    else { // not a dict return with failure
        std::cout << "PyDict_Check Error" << std::endl;
        return false;
    }
}

bool dictToMap(PyObject *srcDict, std::map<std::pair<int, int>, double> &destMap) {
    destMap.clear();
    if(PyDict_Check(srcDict)) {
        Py_ssize_t numItems = PyDict_Size(srcDict);
        Py_ssize_t iteratorPPOS = 0;
        PyObject *currentKey, *currentVal;
        int first, second;
        while(PyDict_Next(srcDict, &iteratorPPOS, &currentKey, &currentVal)) {
            if (!PyArg_ParseTuple(currentKey, "ii", &first, &second)) {
                std::cout << "PyArg_ParseTuple Error" << std::endl;
                return false;
            }
            // might be worth checking PyFloat_Check...
            destMap[std::make_pair(first, second)] = PyFloat_AsDouble(currentVal);
        }
        return (numItems == destMap.size());
    }
    else { // not a dict return with failure
        std::cout << "PyDict_Check Error" << std::endl;
        return false;
    }
}

bool dictToMap(PyObject *srcDict, std::map<int, std::vector<int>> &destMap) {
    destMap.clear();
    if(PyDict_Check(srcDict)) {
        Py_ssize_t numItems = PyDict_Size(srcDict);
        Py_ssize_t iteratorPPOS = 0;
        PyObject *currentKey, *currentVal;
        while(PyDict_Next(srcDict, &iteratorPPOS, &currentKey, &currentVal)) {
            // might be worth checking PyFloat_Check...
            destMap[PyLong_AsLong(currentKey)] = listTupleToVector(currentVal);
        }
        return (numItems == destMap.size());
    }
    else { // not a dict return with failure
        std::cout << "PyDict_Check Error" << std::endl;
        return false;
    }
}

std::vector<int> listTupleToVector(PyObject* incoming) {
	std::vector<int> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back(PyLong_AsLong(value));
		}
	}
    else if (PyList_Check(incoming)) {
        for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
            PyObject *value = PyList_GetItem(incoming, i);
            data.push_back(PyLong_AsLong(value));
        }
    }
    else {
        throw std::logic_error("Passed PyObject pointer was not a list or tuple!");
    }
	return data;
}

bool is_done(double *array, int size) {
    for (int i = 0; i < size; i++) {
        if (array[i] == 0) {
            return false;
        }
    }
    return true;
}

static PyMethodDef dag_completion_time_methods[] = {
    // The first property is the name exposed to Python, fast_tanh, the second is the C++
    // function name that contains the implementation.
    { "get_completion_time", (PyCFunction)get_completion_time, METH_VARARGS, "calculate DAG completion time" },
    { "get_completion_time_no_exec_order", (PyCFunction)get_completion_time_no_exec_order, METH_VARARGS, "calculate DAG completion time" },
    { "get_edge_weight", (PyCFunction)get_edge_weight, METH_VARARGS, "calculate DAG edge weight" },
 
    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};
 
static PyModuleDef dag_completion_time_module = {
    PyModuleDef_HEAD_INIT,
    "dag_completion_time",                        // Module name to use with Python import statements
    "Provides DAG completion time calculation, but faster",  // Module description
    0,                                      // Module memory option
    dag_completion_time_methods                   // Structure that defines the methods of the module
};
 
PyMODINIT_FUNC PyInit_dag_completion_time() {
    return PyModule_Create(&dag_completion_time_module);
}