#include <Python.h>

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#error "neuralfn._native_train currently targets POSIX spawn/exec environments."
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
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

bool optional_bool_from_config(PyObject* config, const char* key, bool default_value, bool* out) {
    PyObject* value = get_optional_item(config, key);
    if (value == nullptr) {
        if (PyErr_Occurred()) {
            return false;
        }
        *out = default_value;
        return true;
    }
    const int truthy = PyObject_IsTrue(value);
    Py_DECREF(value);
    if (truthy < 0) {
        return false;
    }
    *out = truthy != 0;
    return true;
}

bool is_forbidden_native_launcher(const std::string& command) {
    const std::string executable = std::filesystem::path(command).filename().string();
    std::string lower;
    lower.reserve(executable.size());
    for (char ch : executable) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    static const char* forbidden_names[] = {
        "python", "python3", "python3.11", "python3.12", "python3.13",
        "pypy", "pypy3", "bash", "sh", "zsh", "fish",
    };
    for (const char* name : forbidden_names) {
        if (lower == name) {
            return true;
        }
    }
    static const char* forbidden_suffixes[] = {".py", ".sh", ".bash", ".zsh"};
    for (const char* suffix : forbidden_suffixes) {
        const std::size_t suffix_len = std::strlen(suffix);
        if (lower.size() >= suffix_len &&
            lower.compare(lower.size() - suffix_len, suffix_len, suffix) == 0) {
            return true;
        }
    }
    return false;
}

bool env_is_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0';
}

void setenv_default_if_empty(const char* name, const std::string& value) {
    if (!value.empty() && env_is_empty(name)) {
        setenv(name, value.c_str(), 1);
    }
}

bool validate_native_command_from_config(
    PyObject* config,
    std::vector<std::string>* command,
    std::string* error) {
    bool strict_native_command = true;
    if (!optional_bool_from_config(config, "strict_native_command", true, &strict_native_command)) {
        return false;
    }
    if (strict_native_command && !command->empty() && is_forbidden_native_launcher((*command)[0])) {
        *error = "native train binding requires a compiled C++ command; got launcher " + (*command)[0] +
            ". Set strict_native_command=False only for diagnostics.";
        command->clear();
    }
    return true;
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
        return validate_native_command_from_config(config, command, error);
    }
    if (!string_list_from_config(config, second_key, command)) {
        return false;
    }
    if (!command->empty()) {
        return validate_native_command_from_config(config, command, error);
    }
    if (!string_list_from_config(config, "launcher_argv", command)) {
        return false;
    }
    if (!command->empty()) {
        return validate_native_command_from_config(config, command, error);
    }

    *error = alias_only_gpt_config
        ? "native train alias-only config requires a non-empty compiled_cli_argv, argv, or launcher_argv list"
        : "native train config requires a non-empty argv, compiled_cli_argv, or launcher_argv list";
    return true;
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
    setenv_default_if_empty("CUDA_VISIBLE_DEVICES", visible_devices);

    std::string max_connections;
    if (!optional_string_from_config(config, "cuda_device_max_connections", &max_connections)) {
        return nullptr;
    }
    setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", max_connections);
    setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");

    const int return_code = run_exec_and_wait(command);
    if (return_code < 0) {
        return nullptr;
    }
    return PyLong_FromLong(return_code);
}

PyObject* resolve_command(PyObject*, PyObject* args) {
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(args, "O!:resolve_command", &PyDict_Type, &config)) {
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

    PyObject* list = PyList_New(static_cast<Py_ssize_t>(command.size()));
    if (list == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(command.size()); ++i) {
        PyObject* item = PyUnicode_FromStringAndSize(command[static_cast<std::size_t>(i)].data(),
                                                     static_cast<Py_ssize_t>(command[static_cast<std::size_t>(i)].size()));
        if (item == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

PyMethodDef methods[] = {
    {"run_train", run_train, METH_VARARGS, "Run the unified native NeuralFn trainer from a config dict."},
    {"run_native_train", run_train, METH_VARARGS, "Alias for run_train."},
    {"resolve_command", resolve_command, METH_VARARGS, "Resolve the native train command argv from a config dict."},
    {"resolve_native_train_command", resolve_command, METH_VARARGS, "Alias for resolve_command."},
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
