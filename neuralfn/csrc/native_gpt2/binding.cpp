#include <Python.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#error "neuralfn._native_gpt/_native_gpt2 currently targets POSIX fork/exec environments."
#else
#include <spawn.h>
#include <sys/select.h>
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

bool is_env_launcher(const std::string& command) {
    const std::string executable = std::filesystem::path(command).filename().string();
    std::string lower;
    lower.reserve(executable.size());
    for (char ch : executable) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return lower == "env";
}

std::string forbidden_native_launcher_from_command(const std::vector<std::string>& command) {
    if (command.empty()) {
        return "";
    }
    if (is_forbidden_native_launcher(command[0])) {
        return command[0];
    }
    if (is_env_launcher(command[0])) {
        return command[0];
    }
    return "";
}

bool validate_native_command_from_config(
    PyObject* config,
    std::vector<std::string>* command,
    std::string* error) {
    bool strict_native_command = true;
    if (!optional_bool_from_config(config, "strict_native_command", true, &strict_native_command)) {
        return false;
    }
    const std::string forbidden_launcher =
        strict_native_command ? forbidden_native_launcher_from_command(*command) : "";
    if (!forbidden_launcher.empty()) {
        *error = "native GPT binding requires a compiled C++ command; got launcher " + forbidden_launcher +
            ". Set strict_native_command=False only for diagnostics.";
        command->clear();
    }
    return true;
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
            return validate_native_command_from_config(config, command, error);
        }
    }

    if (!string_list_from_config(config, "argv", command)) {
        return false;
    }
    if (!command->empty()) {
        return validate_native_command_from_config(config, command, error);
    }

    if (!use_compiled_cli) {
        if (!string_list_from_config(config, "compiled_cli_argv", command)) {
            return false;
        }
        if (!command->empty()) {
            return validate_native_command_from_config(config, command, error);
        }
    }

    if (!string_list_from_config(config, "launcher_argv", command)) {
        return false;
    }
    if (!command->empty()) {
        return validate_native_command_from_config(config, command, error);
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

bool env_is_empty(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0';
}

void setenv_default_if_empty(const char* name, const std::string& value) {
    if (!value.empty() && env_is_empty(name)) {
        setenv(name, value.c_str(), 1);
    }
}

bool apply_cuda_env_from_config(PyObject* config) {
    std::string visible_devices;
    if (!optional_string_from_config(config, "cuda_visible_devices", &visible_devices)) {
        return false;
    }
    setenv_default_if_empty("CUDA_VISIBLE_DEVICES", visible_devices);

    std::string max_connections;
    if (!optional_string_from_config(config, "cuda_device_max_connections", &max_connections)) {
        return false;
    }
    setenv_default_if_empty("CUDA_DEVICE_MAX_CONNECTIONS", max_connections);
    setenv_default_if_empty("CUDA_MODULE_LOADING", "LAZY");
    return true;
}

bool drain_child_output_pipes(int stdout_fd, int stderr_fd, std::string* stdout_text, std::string* stderr_text) {
    stdout_text->clear();
    stderr_text->clear();
    bool stdout_open = stdout_fd >= 0;
    bool stderr_open = stderr_fd >= 0;
    char buffer[8192];
    while (stdout_open || stderr_open) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        int max_fd = -1;
        if (stdout_open) {
            FD_SET(stdout_fd, &read_fds);
            max_fd = std::max(max_fd, stdout_fd);
        }
        if (stderr_open) {
            FD_SET(stderr_fd, &read_fds);
            max_fd = std::max(max_fd, stderr_fd);
        }
        const int ready = select(max_fd + 1, &read_fds, nullptr, nullptr, nullptr);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            PyErr_Format(PyExc_OSError, "select on child output pipes failed: %s", std::strerror(errno));
            return false;
        }
        auto drain_one = [&](int fd, bool* open, std::string* text, const char* label) -> bool {
            if (!*open || !FD_ISSET(fd, &read_fds)) {
                return true;
            }
            const ssize_t count = read(fd, buffer, sizeof(buffer));
            if (count > 0) {
                text->append(buffer, static_cast<std::size_t>(count));
                return true;
            }
            if (count == 0) {
                close(fd);
                *open = false;
                return true;
            }
            if (errno == EINTR) {
                return true;
            }
            PyErr_Format(PyExc_OSError, "read %s pipe failed: %s", label, std::strerror(errno));
            return false;
        };
        if (!drain_one(stdout_fd, &stdout_open, stdout_text, "stdout")) {
            return false;
        }
        if (!drain_one(stderr_fd, &stderr_open, stderr_text, "stderr")) {
            return false;
        }
    }
    return true;
}

bool run_exec_capture_output(
    const std::vector<std::string>& command,
    int* return_code,
    std::string* stdout_text,
    std::string* stderr_text) {
    int stdout_pipe[2] = {-1, -1};
    if (pipe(stdout_pipe) != 0) {
        PyErr_Format(PyExc_OSError, "pipe failed: %s", std::strerror(errno));
        return false;
    }
    int stderr_pipe[2] = {-1, -1};
    if (pipe(stderr_pipe) != 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        PyErr_Format(PyExc_OSError, "pipe failed: %s", std::strerror(errno));
        return false;
    }

    posix_spawn_file_actions_t actions;
    int action_status = posix_spawn_file_actions_init(&actions);
    if (action_status != 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        PyErr_Format(PyExc_OSError, "posix_spawn_file_actions_init failed: %s", std::strerror(action_status));
        return false;
    }
    action_status = posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_adddup2(&actions, stderr_pipe[1], STDERR_FILENO);
    }
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_addclose(&actions, stdout_pipe[0]);
    }
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_addclose(&actions, stdout_pipe[1]);
    }
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_addclose(&actions, stderr_pipe[0]);
    }
    if (action_status == 0) {
        action_status = posix_spawn_file_actions_addclose(&actions, stderr_pipe[1]);
    }
    if (action_status != 0) {
        posix_spawn_file_actions_destroy(&actions);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
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
    close(stderr_pipe[1]);
    if (spawn_status != 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        PyErr_Format(PyExc_OSError, "posix_spawnp failed for %s: %s", command[0].c_str(), std::strerror(spawn_status));
        return false;
    }

    if (!drain_child_output_pipes(stdout_pipe[0], stderr_pipe[0], stdout_text, stderr_text)) {
        return false;
    }

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
    std::string stderr_text;
    if (!run_exec_capture_output(command, &return_code, &stdout_text, &stderr_text)) {
        return nullptr;
    }
    PyObject* stdout_obj = PyUnicode_FromStringAndSize(
        stdout_text.c_str(),
        static_cast<Py_ssize_t>(stdout_text.size()));
    if (stdout_obj == nullptr) {
        return nullptr;
    }
    PyObject* stderr_obj = PyUnicode_FromStringAndSize(
        stderr_text.c_str(),
        static_cast<Py_ssize_t>(stderr_text.size()));
    if (stderr_obj == nullptr) {
        Py_DECREF(stdout_obj);
        return nullptr;
    }
    PyObject* result = Py_BuildValue(
        "{s:i,s:O,s:O}",
        "returncode",
        return_code,
        "stdout",
        stdout_obj,
        "stderr",
        stderr_obj);
    Py_DECREF(stdout_obj);
    Py_DECREF(stderr_obj);
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
