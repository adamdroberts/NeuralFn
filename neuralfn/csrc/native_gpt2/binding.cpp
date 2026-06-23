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

bool command_from_config(PyObject* config, std::vector<std::string>* command, std::string* error) {
    std::string train_data;
    std::string val_data;
    if (!optional_string_from_config(config, "train_data", &train_data)) {
        return false;
    }
    if (!optional_string_from_config(config, "val_data", &val_data)) {
        return false;
    }

    command->clear();
    const bool use_compiled_cli = train_data.empty() || val_data.empty();
    if (use_compiled_cli) {
        if (!string_list_from_config(config, "compiled_cli_argv", command)) {
            return false;
        }
        if (!command->empty()) {
            return true;
        }
    }

    if (!string_list_from_config(config, "argv", command)) {
        return false;
    }
    if (!command->empty()) {
        return true;
    }

    if (!use_compiled_cli) {
        if (!string_list_from_config(config, "compiled_cli_argv", command)) {
            return false;
        }
        if (!command->empty()) {
            return true;
        }
    }

    if (!string_list_from_config(config, "launcher_argv", command)) {
        return false;
    }
    if (!command->empty()) {
        return true;
    }

    *error = use_compiled_cli
        ? "native GPT alias-only config requires a non-empty compiled_cli_argv, argv, or launcher_argv list"
        : "native GPT config requires a non-empty argv, compiled_cli_argv, or launcher_argv list";
    return true;
}

PyObject* command_to_list(const std::vector<std::string>& command) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(command.size()));
    if (list == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(command.size()); ++i) {
        const std::string& item = command[static_cast<std::size_t>(i)];
        PyObject* value = PyUnicode_FromStringAndSize(item.c_str(), static_cast<Py_ssize_t>(item.size()));
        if (value == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, value);
    }
    return list;
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

bool apply_cuda_env_from_config(PyObject* config) {
    std::string visible_devices;
    if (!optional_string_from_config(config, "cuda_visible_devices", &visible_devices)) {
        return false;
    }
    if (!visible_devices.empty() && std::getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
        setenv("CUDA_VISIBLE_DEVICES", visible_devices.c_str(), 0);
    }

    std::string max_connections;
    if (!optional_string_from_config(config, "cuda_device_max_connections", &max_connections)) {
        return false;
    }
    if (!max_connections.empty() && std::getenv("CUDA_DEVICE_MAX_CONNECTIONS") == nullptr) {
        setenv("CUDA_DEVICE_MAX_CONNECTIONS", max_connections.c_str(), 0);
    }
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "LAZY", 0);
    }
    return true;
}

bool run_exec_capture_stdout(const std::vector<std::string>& command, int* return_code, std::string* stdout_text) {
    int stdout_pipe[2] = {-1, -1};
    if (pipe(stdout_pipe) != 0) {
        PyErr_Format(PyExc_OSError, "pipe failed: %s", std::strerror(errno));
        return false;
    }

    posix_spawn_file_actions_t actions;
    int action_status = posix_spawn_file_actions_init(&actions);
    if (action_status != 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        PyErr_Format(PyExc_OSError, "posix_spawn_file_actions_init failed: %s", std::strerror(action_status));
        return false;
    }
    action_status = posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_addclose(&actions, stdout_pipe[0]);
    }
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_addclose(&actions, stdout_pipe[1]);
    }
    if (action_status != 0) {
        posix_spawn_file_actions_destroy(&actions);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        PyErr_Format(PyExc_OSError, "posix_spawn_file_actions setup failed: %s", std::strerror(action_status));
        return false;
    }

    std::vector<char*> exec_args;
    exec_args.reserve(command.size() + 1);
    for (const std::string& item : command) {
        exec_args.push_back(const_cast<char*>(item.c_str()));
    }
    exec_args.push_back(nullptr);

    pid_t pid = 0;
    const int spawn_status = posix_spawnp(&pid, command[0].c_str(), &actions, nullptr, exec_args.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stdout_pipe[1]);
    if (spawn_status != 0) {
        close(stdout_pipe[0]);
        PyErr_Format(PyExc_OSError, "posix_spawnp failed for %s: %s", command[0].c_str(), std::strerror(spawn_status));
        return false;
    }

    stdout_text->clear();
    char buffer[8192];
    for (;;) {
        const ssize_t count = read(stdout_pipe[0], buffer, sizeof(buffer));
        if (count > 0) {
            stdout_text->append(buffer, static_cast<std::size_t>(count));
            continue;
        }
        if (count == 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        close(stdout_pipe[0]);
        PyErr_Format(PyExc_OSError, "read stdout pipe failed: %s", std::strerror(errno));
        return false;
    }
    close(stdout_pipe[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        PyErr_Format(PyExc_OSError, "waitpid failed: %s", std::strerror(errno));
        return false;
    }

    if (WIFEXITED(status)) {
        *return_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        *return_code = 128 + WTERMSIG(status);
    } else {
        *return_code = 126;
    }
    return true;
}

PyObject* run_gpt(PyObject*, PyObject* args) {
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(args, "O!:run_gpt", &PyDict_Type, &config)) {
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

    if (!apply_cuda_env_from_config(config)) {
        return nullptr;
    }

    const int return_code = run_exec_and_wait(command);

    if (return_code < 0) {
        return nullptr;
    }
    return PyLong_FromLong(return_code);
}

PyObject* run_gpt_capture(PyObject*, PyObject* args) {
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(args, "O!:run_gpt_capture", &PyDict_Type, &config)) {
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
    if (!apply_cuda_env_from_config(config)) {
        return nullptr;
    }

    int return_code = 0;
    std::string stdout_text;
    if (!run_exec_capture_stdout(command, &return_code, &stdout_text)) {
        return nullptr;
    }
    PyObject* stdout_obj = PyUnicode_FromStringAndSize(
        stdout_text.c_str(),
        static_cast<Py_ssize_t>(stdout_text.size()));
    if (stdout_obj == nullptr) {
        return nullptr;
    }
    PyObject* result = Py_BuildValue("{s:i,s:O,s:s}", "returncode", return_code, "stdout", stdout_obj, "stderr", "");
    Py_DECREF(stdout_obj);
    return result;
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
    return command_to_list(command);
}

PyMethodDef methods[] = {
    {"run_gpt", run_gpt, METH_VARARGS, "Run the native GPT CUDA trainer from a NeuralFn config dict."},
    {"run_gpt2", run_gpt, METH_VARARGS, "Compatibility alias for run_gpt."},
    {"run_train", run_gpt, METH_VARARGS, "Alias for run_gpt."},
    {"run_gpt_capture", run_gpt_capture, METH_VARARGS, "Run the native GPT command and capture stdout."},
    {"run_gpt2_capture", run_gpt_capture, METH_VARARGS, "Compatibility alias for run_gpt_capture."},
    {"run_infer", run_gpt_capture, METH_VARARGS, "Run a native GPT inference command and capture stdout."},
    {"resolve_command", resolve_command, METH_VARARGS, "Return the native GPT command argv without running it."},
    {"resolve_native_gpt_command", resolve_command, METH_VARARGS, "Alias for resolve_command."},
    {"resolve_native_gpt2_command", resolve_command, METH_VARARGS, "Compatibility alias for resolve_command."},
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
