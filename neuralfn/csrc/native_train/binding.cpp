#include <Python.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#error "neuralfn._native_train currently targets POSIX fork/exec environments."
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

PyObject* get_optional_item(PyObject* mapping, const char* key) {
    PyObject* value = PyMapping_GetItemString(mapping, key);
    if (value == nullptr && PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyErr_Clear();
        return nullptr;
    }
    return value;
}

bool unicode_to_string(PyObject* value, const char* field, std::string* out) {
    if (!PyUnicode_Check(value)) {
        PyErr_Format(PyExc_TypeError, "%s entries must be strings", field);
        return false;
    }
    Py_ssize_t size = 0;
    const char* chars = PyUnicode_AsUTF8AndSize(value, &size);
    if (chars == nullptr) {
        return false;
    }
    out->assign(chars, static_cast<std::size_t>(size));
    return true;
}

bool string_list_from_config(PyObject* config, const char* key, std::vector<std::string>* out) {
    PyObject* value = get_optional_item(config, key);
    if (value == nullptr) {
        return !PyErr_Occurred();
    }

    PyObject* seq = PySequence_Fast(value, "native train argv must be a sequence");
    Py_DECREF(value);
    if (seq == nullptr) {
        return false;
    }

    const Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    out->clear();
    out->reserve(static_cast<std::size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < size; ++i) {
        std::string item;
        if (!unicode_to_string(items[i], key, &item)) {
            Py_DECREF(seq);
            return false;
        }
        out->push_back(std::move(item));
    }
    Py_DECREF(seq);
    return true;
}

bool optional_string_from_config(PyObject* config, const char* key, std::string* out, bool* present = nullptr) {
    PyObject* value = get_optional_item(config, key);
    if (value == nullptr) {
        if (present != nullptr) {
            *present = false;
        }
        return !PyErr_Occurred();
    }
    if (present != nullptr) {
        *present = true;
    }
    const bool ok = unicode_to_string(value, key, out);
    Py_DECREF(value);
    return ok;
}

bool command_from_config(PyObject* config, std::vector<std::string>* command, std::string* error) {
    std::string train_data;
    std::string val_data;
    bool has_train_data = false;
    bool has_val_data = false;
    if (!optional_string_from_config(config, "train_data", &train_data, &has_train_data)) {
        return false;
    }
    if (!optional_string_from_config(config, "val_data", &val_data, &has_val_data)) {
        return false;
    }

    const bool alias_only_gpt_config =
        (has_train_data || has_val_data) && (train_data.empty() || val_data.empty());
    const char* first_key = alias_only_gpt_config ? "compiled_cli_argv" : "argv";
    const char* second_key = alias_only_gpt_config ? "argv" : "compiled_cli_argv";

    if (!string_list_from_config(config, first_key, command)) {
        return false;
    }
    if (!command->empty()) {
        return true;
    }
    if (!string_list_from_config(config, second_key, command)) {
        return false;
    }
    if (!command->empty()) {
        return true;
    }
    if (!string_list_from_config(config, "launcher_argv", command)) {
        return false;
    }
    if (!command->empty()) {
        return true;
    }

    *error = alias_only_gpt_config
        ? "native train alias-only config requires a non-empty compiled_cli_argv, argv, or launcher_argv list"
        : "native train config requires a non-empty argv, compiled_cli_argv, or launcher_argv list";
    return true;
}

int run_exec_and_wait(const std::vector<std::string>& command) {
    pid_t pid = fork();
    if (pid < 0) {
        PyErr_Format(PyExc_OSError, "fork failed: %s", std::strerror(errno));
        return -1;
    }

    if (pid == 0) {
        std::vector<char*> exec_args;
        exec_args.reserve(command.size() + 1);
        for (const std::string& item : command) {
            exec_args.push_back(const_cast<char*>(item.c_str()));
        }
        exec_args.push_back(nullptr);
        execvp(command[0].c_str(), exec_args.data());
        _exit(errno == ENOENT ? 127 : 126);
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        PyErr_Format(PyExc_OSError, "waitpid failed: %s", std::strerror(errno));
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 126;
}

PyObject* run_train(PyObject*, PyObject* args) {
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(args, "O!:run_train", &PyDict_Type, &config)) {
        return nullptr;
    }

    std::vector<std::string> command;
    std::string command_error;
    if (!command_from_config(config, &command, &command_error)) {
        return nullptr;
    }
    if (command.empty()) {
        PyErr_SetString(PyExc_ValueError, command_error.c_str());
        return nullptr;
    }

    std::string visible_devices;
    if (!optional_string_from_config(config, "cuda_visible_devices", &visible_devices)) {
        return nullptr;
    }
    if (!visible_devices.empty() && std::getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
        setenv("CUDA_VISIBLE_DEVICES", visible_devices.c_str(), 0);
    }

    std::string max_connections;
    if (!optional_string_from_config(config, "cuda_device_max_connections", &max_connections)) {
        return nullptr;
    }
    if (!max_connections.empty() && std::getenv("CUDA_DEVICE_MAX_CONNECTIONS") == nullptr) {
        setenv("CUDA_DEVICE_MAX_CONNECTIONS", max_connections.c_str(), 0);
    }
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "LAZY", 0);
    }

    const int return_code = run_exec_and_wait(command);
    if (return_code < 0) {
        return nullptr;
    }
    return PyLong_FromLong(return_code);
}

PyMethodDef methods[] = {
    {"run_train", run_train, METH_VARARGS, "Run the unified native NeuralFn trainer from a config dict."},
    {"run_native_train", run_train, METH_VARARGS, "Alias for run_train."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native_train",
    "Unified native training frontend binding for NeuralFn.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__native_train() {
    return PyModule_Create(&module);
}
