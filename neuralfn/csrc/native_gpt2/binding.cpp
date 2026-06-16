#include <Python.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#error "neuralfn._native_gpt/_native_gpt2 currently targets POSIX fork/exec environments."
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef NFN_NATIVE_GPT_PY_MODULE_NAME
#define NFN_NATIVE_GPT_PY_MODULE_NAME "_native_gpt2"
#endif

#ifndef NFN_NATIVE_GPT_PY_MODULE_DOC
#define NFN_NATIVE_GPT_PY_MODULE_DOC "Native GPT CUDA trainer binding for NeuralFn."
#endif

#ifndef NFN_NATIVE_GPT_PY_INIT
#define NFN_NATIVE_GPT_PY_INIT PyInit__native_gpt2
#endif

extern "C" char** environ;

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

    PyObject* seq = PySequence_Fast(value, "native GPT-2 argv must be a sequence");
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

bool optional_string_from_config(PyObject* config, const char* key, std::string* out) {
    PyObject* value = get_optional_item(config, key);
    if (value == nullptr) {
        return !PyErr_Occurred();
    }
    const bool ok = unicode_to_string(value, key, out);
    Py_DECREF(value);
    return ok;
}

int run_exec_and_wait(const std::vector<std::string>& command) {
    std::vector<char*> exec_args;
    exec_args.reserve(command.size() + 1);
    for (const std::string& item : command) {
        exec_args.push_back(const_cast<char*>(item.c_str()));
    }
    exec_args.push_back(nullptr);

    pid_t pid = 0;
    const int spawn_status = posix_spawnp(&pid, command[0].c_str(), nullptr, nullptr, exec_args.data(), environ);
    if (spawn_status != 0) {
        PyErr_Format(PyExc_OSError, "posix_spawnp failed for %s: %s", command[0].c_str(), std::strerror(spawn_status));
        return -1;
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

PyObject* run_gpt(PyObject*, PyObject* args) {
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(args, "O!:run_gpt", &PyDict_Type, &config)) {
        return nullptr;
    }

    std::string train_data;
    std::string val_data;
    if (!optional_string_from_config(config, "train_data", &train_data)) {
        return nullptr;
    }
    if (!optional_string_from_config(config, "val_data", &val_data)) {
        return nullptr;
    }

    std::vector<std::string> command;
    const bool use_compiled_cli = train_data.empty() || val_data.empty();
    if (use_compiled_cli) {
        if (!string_list_from_config(config, "compiled_cli_argv", &command)) {
            return nullptr;
        }
    }
    if (command.empty()) {
        if (!string_list_from_config(config, "argv", &command)) {
            return nullptr;
        }
    }
    if (command.empty()) {
        if (!string_list_from_config(config, "launcher_argv", &command)) {
            return nullptr;
        }
    }
    if (command.empty()) {
        PyErr_SetString(
            PyExc_ValueError,
            use_compiled_cli
                ? "native GPT alias-only config requires a non-empty compiled_cli_argv list"
                : "native GPT config requires a non-empty argv list");
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
    {"run_gpt", run_gpt, METH_VARARGS, "Run the native GPT CUDA trainer from a NeuralFn config dict."},
    {"run_gpt2", run_gpt, METH_VARARGS, "Compatibility alias for run_gpt."},
    {"run_train", run_gpt, METH_VARARGS, "Alias for run_gpt."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    NFN_NATIVE_GPT_PY_MODULE_NAME,
    NFN_NATIVE_GPT_PY_MODULE_DOC,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC NFN_NATIVE_GPT_PY_INIT() {
    return PyModule_Create(&module);
}
