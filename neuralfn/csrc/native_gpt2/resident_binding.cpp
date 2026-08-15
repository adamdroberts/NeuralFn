#include <Python.h>

#include "resident_dense.h"
#include "resident_glimmer.h"
#include "resident_glimmer_assistant.h"
#include "resident_llama.h"
#include "resident_moe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using neuralfn::resident_dense::DecodeResult;
using neuralfn::resident_dense::DenseInferenceConfig;
using neuralfn::resident_dense::DenseModel;
using neuralfn::resident_dense::DenseSession;
using neuralfn::resident_dense::GenerationConfig;
using neuralfn::resident_dense::KVCacheMode;
using neuralfn::resident_dense::MlpActivation;
using neuralfn::resident_dense::ResidentCancellationError;
using neuralfn::resident_dense::TurboQuantProfile;
using neuralfn::resident_dense::TurboQuantCodec;
using neuralfn::resident_dense::TurboQuantTables;
using neuralfn::resident_dense::TileTurboQuantConfig;
using neuralfn::resident_dense::TileTurboQuantModelStats;
using neuralfn::resident_llama::LlamaInferenceConfig;
using neuralfn::resident_llama::LlamaModel;
using neuralfn::resident_llama::LlamaSession;
using neuralfn::resident_glimmer::GlimmerInferenceConfig;
using neuralfn::resident_glimmer::GlimmerModel;
using neuralfn::resident_glimmer::GlimmerSession;
using neuralfn::resident_glimmer::WeightContainer;
using GlimmerAssistantConfig = neuralfn::resident_glimmer_assistant::Config;
using GlimmerAssistantModel = neuralfn::resident_glimmer_assistant::Model;
using neuralfn::resident_moe::MoeInferenceConfig;
using neuralfn::resident_moe::MoeModel;
using neuralfn::resident_moe::MoeSession;

constexpr const char* kModelCapsuleName = "neuralfn.resident_dense.model.v1";
constexpr const char* kSessionCapsuleName = "neuralfn.resident_dense.session.v1";
std::atomic<std::uint64_t> next_model_identity{1};

struct ModelHandle {
    enum class Kind {
        Dense,
        Llama,
        Glimmer,
        Moe,
    };

    Kind kind = Kind::Dense;
    std::uint64_t identity = 0;
    std::shared_ptr<DenseModel> dense;
    std::shared_ptr<LlamaModel> llama;
    std::shared_ptr<GlimmerModel> glimmer;
    std::shared_ptr<GlimmerAssistantModel> glimmer_assistant;
    std::shared_ptr<MoeModel> moe;

    bool has_value() const noexcept {
        if (kind == Kind::Dense) {
            return static_cast<bool>(dense);
        }
        if (kind == Kind::Llama) {
            return static_cast<bool>(llama);
        }
        return kind == Kind::Glimmer ? static_cast<bool>(glimmer) : static_cast<bool>(moe);
    }

    void close() noexcept {
        if (dense) {
            dense->close();
            dense.reset();
        }
        if (llama) {
            llama->close();
            llama.reset();
        }
        if (glimmer) {
            if (glimmer_assistant) {
                glimmer_assistant->close();
                glimmer_assistant.reset();
            }
            glimmer->close();
            glimmer.reset();
        }
        if (moe) {
            moe->close();
            moe.reset();
        }
    }
};

struct SessionHandle {
    ModelHandle::Kind kind = ModelHandle::Kind::Dense;
    std::uint64_t model_identity = 0;
    std::shared_ptr<DenseSession> dense;
    std::shared_ptr<LlamaSession> llama;
    std::shared_ptr<GlimmerSession> glimmer;
    std::shared_ptr<MoeSession> moe;

    bool has_value() const noexcept {
        if (kind == ModelHandle::Kind::Dense) {
            return static_cast<bool>(dense);
        }
        if (kind == ModelHandle::Kind::Llama) {
            return static_cast<bool>(llama);
        }
        return kind == ModelHandle::Kind::Glimmer
            ? static_cast<bool>(glimmer)
            : static_cast<bool>(moe);
    }

    void close() noexcept {
        if (dense) {
            dense->close();
            dense.reset();
        }
        if (llama) {
            llama->close();
            llama.reset();
        }
        if (glimmer) {
            glimmer->close();
            glimmer.reset();
        }
        if (moe) {
            moe->close();
            moe.reset();
        }
    }
};

struct PyObjectDeleter {
    void operator()(PyObject* value) const noexcept {
        Py_XDECREF(value);
    }
};

using OwnedPyObject = std::unique_ptr<PyObject, PyObjectDeleter>;

PyObject* optional_item(PyObject* mapping, const char* key) {
    PyObject* value = PyMapping_GetItemString(mapping, key);
    if (value == nullptr && PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyErr_Clear();
        return nullptr;
    }
    return value;
}

bool unicode_value(PyObject* value, const char* field, std::string* output) {
    if (!PyUnicode_Check(value)) {
        PyErr_Format(PyExc_TypeError, "%s must be a string", field);
        return false;
    }
    Py_ssize_t size = 0;
    const char* raw = PyUnicode_AsUTF8AndSize(value, &size);
    if (raw == nullptr) {
        return false;
    }
    output->assign(raw, static_cast<std::size_t>(size));
    return true;
}

bool mapping_string(PyObject* mapping, const char* key, bool required, std::string* output) {
    PyObject* value = optional_item(mapping, key);
    if (value == nullptr) {
        if (PyErr_Occurred()) {
            return false;
        }
        if (required) {
            PyErr_Format(PyExc_ValueError, "resident manifest is missing %s", key);
            return false;
        }
        return true;
    }
    const bool result = unicode_value(value, key, output);
    Py_DECREF(value);
    return result;
}

bool mapping_optional_string_or_none(
    PyObject* mapping,
    const char* key,
    std::string* output) {
    PyObject* value = optional_item(mapping, key);
    if (value == nullptr) {
        return !PyErr_Occurred();
    }
    if (value == Py_None) {
        output->clear();
        Py_DECREF(value);
        return true;
    }
    const bool result = unicode_value(value, key, output);
    Py_DECREF(value);
    return result;
}

bool py_int64(PyObject* value, const char* field, std::int64_t* output) {
    if (PyBool_Check(value) || !PyLong_Check(value)) {
        PyErr_Format(PyExc_TypeError, "%s must be an integer", field);
        return false;
    }
    const long long parsed = PyLong_AsLongLong(value);
    if (parsed == -1 && PyErr_Occurred()) {
        return false;
    }
    *output = static_cast<std::int64_t>(parsed);
    return true;
}

bool mapping_int64(PyObject* mapping, const char* key, bool required, std::int64_t default_value, std::int64_t* output) {
    PyObject* value = optional_item(mapping, key);
    if (value == nullptr) {
        if (PyErr_Occurred()) {
            return false;
        }
        if (required) {
            PyErr_Format(PyExc_ValueError, "resident config is missing %s", key);
            return false;
        }
        *output = default_value;
        return true;
    }
    const bool result = py_int64(value, key, output);
    Py_DECREF(value);
    return result;
}

bool mapping_bool(PyObject* mapping, const char* key, bool required, bool default_value, bool* output) {
    PyObject* value = optional_item(mapping, key);
    if (value == nullptr) {
        if (PyErr_Occurred()) {
            return false;
        }
        if (required) {
            PyErr_Format(PyExc_ValueError, "resident config is missing %s", key);
            return false;
        }
        *output = default_value;
        return true;
    }
    if (!PyBool_Check(value)) {
        Py_DECREF(value);
        PyErr_Format(PyExc_TypeError, "%s must be a boolean", key);
        return false;
    }
    *output = value == Py_True;
    Py_DECREF(value);
    return true;
}

bool mapping_double(PyObject* mapping, const char* key, bool required, double default_value, double* output) {
    PyObject* value = optional_item(mapping, key);
    if (value == nullptr) {
        if (PyErr_Occurred()) {
            return false;
        }
        if (required) {
            PyErr_Format(PyExc_ValueError, "resident config is missing %s", key);
            return false;
        }
        *output = default_value;
        return true;
    }
    if (PyBool_Check(value)) {
        Py_DECREF(value);
        PyErr_Format(PyExc_TypeError, "%s must be a number", key);
        return false;
    }
    const double parsed = PyFloat_AsDouble(value);
    Py_DECREF(value);
    if (parsed == -1.0 && PyErr_Occurred()) {
        return false;
    }
    *output = parsed;
    return true;
}

bool token_vector(PyObject* value, const char* field, std::vector<std::int64_t>* output) {
    PyObject* sequence = PySequence_Fast(value, "token ids must be a sequence");
    if (sequence == nullptr) {
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    output->clear();
    output->reserve(static_cast<std::size_t>(count));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < count; ++index) {
        std::int64_t token = 0;
        if (!py_int64(items[index], field, &token)) {
            Py_DECREF(sequence);
            return false;
        }
        output->push_back(token);
    }
    Py_DECREF(sequence);
    return true;
}

bool string_vector(PyObject* value, const char* field, std::vector<std::string>* output) {
    OwnedPyObject sequence(PySequence_Fast(value, "value must be a sequence"));
    if (!sequence) {
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence.get());
    output->clear();
    output->reserve(static_cast<std::size_t>(count));
    PyObject** items = PySequence_Fast_ITEMS(sequence.get());
    for (Py_ssize_t index = 0; index < count; ++index) {
        std::string parsed;
        if (!unicode_value(items[index], field, &parsed)) {
            return false;
        }
        output->push_back(std::move(parsed));
    }
    return true;
}

bool double_vector(PyObject* value, const char* field, std::vector<double>* output) {
    PyObject* sequence = PySequence_Fast(value, "TurboQuant table must be a sequence");
    if (sequence == nullptr) {
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    output->clear();
    output->reserve(static_cast<std::size_t>(count));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t index = 0; index < count; ++index) {
        if (PyBool_Check(items[index])) {
            Py_DECREF(sequence);
            PyErr_Format(PyExc_TypeError, "%s[%zd] must be a finite number", field, index);
            return false;
        }
        const double parsed = PyFloat_AsDouble(items[index]);
        if ((parsed == -1.0 && PyErr_Occurred()) || !std::isfinite(parsed)) {
            Py_DECREF(sequence);
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_ValueError, "%s[%zd] must be finite", field, index);
            }
            return false;
        }
        output->push_back(parsed);
    }
    Py_DECREF(sequence);
    return true;
}

bool float_vector(PyObject* value, const char* field, std::vector<float>* output) {
    std::vector<double> parsed;
    if (!double_vector(value, field, &parsed)) {
        return false;
    }
    output->clear();
    output->reserve(parsed.size());
    for (std::size_t index = 0; index < parsed.size(); ++index) {
        const float converted = static_cast<float>(parsed[index]);
        if (!std::isfinite(converted)) {
            PyErr_Format(
                PyExc_ValueError, "%s[%zu] is not representable as finite float32", field, index);
            return false;
        }
        output->push_back(converted);
    }
    return true;
}

bool turboquant_tables(PyObject* cache, TurboQuantTables* output) {
    PyObject* tables = optional_item(cache, "tables");
    if (tables == nullptr || !PyMapping_Check(tables)) {
        Py_XDECREF(tables);
        PyErr_SetString(
            PyExc_ValueError,
            "TurboQuant resident cache requires deterministic codec tables");
        return false;
    }
    std::string profile;
    if (!mapping_string(cache, "turboquant_profile", true, &profile) ||
        !mapping_int64(tables, "dimension", true, 0, &output->dimension)) {
        Py_DECREF(tables);
        return false;
    }
    if (profile == "mse-3.5") {
        output->profile = TurboQuantProfile::Mse35;
    } else if (profile == "qjl-3.5") {
        output->profile = TurboQuantProfile::Qjl35;
    } else {
        Py_DECREF(tables);
        PyErr_SetString(PyExc_ValueError, "turboquant_profile must be mse-3.5 or qjl-3.5");
        return false;
    }

    PyObject* rotation = optional_item(tables, "rotation");
    PyObject* projection = optional_item(tables, "qjl_projection");
    PyObject* value_widths = optional_item(tables, "value_bit_widths");
    PyObject* key_widths = optional_item(tables, "key_bit_widths");
    PyObject* centroids = optional_item(tables, "centroids");
    if (rotation == nullptr || projection == nullptr || value_widths == nullptr ||
        key_widths == nullptr || centroids == nullptr) {
        Py_XDECREF(rotation);
        Py_XDECREF(projection);
        Py_XDECREF(value_widths);
        Py_XDECREF(key_widths);
        Py_XDECREF(centroids);
        Py_DECREF(tables);
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "TurboQuant codec tables are incomplete");
        }
        return false;
    }
    const bool vectors_valid =
        double_vector(rotation, "rotation", &output->rotation) &&
        double_vector(projection, "qjl_projection", &output->qjl_projection) &&
        token_vector(value_widths, "value_bit_widths", &output->value_bit_widths) &&
        token_vector(key_widths, "key_bit_widths", &output->key_bit_widths);
    Py_DECREF(rotation);
    Py_DECREF(projection);
    Py_DECREF(value_widths);
    Py_DECREF(key_widths);
    if (!vectors_valid) {
        Py_DECREF(centroids);
        Py_DECREF(tables);
        return false;
    }

    PyObject* centroid_sequence = PySequence_Fast(
        centroids, "TurboQuant centroids must be a sequence indexed by bit width");
    Py_DECREF(centroids);
    if (centroid_sequence == nullptr) {
        Py_DECREF(tables);
        return false;
    }
    const Py_ssize_t centroid_count = PySequence_Fast_GET_SIZE(centroid_sequence);
    output->centroids.assign(static_cast<std::size_t>(centroid_count), {});
    PyObject** centroid_items = PySequence_Fast_ITEMS(centroid_sequence);
    for (Py_ssize_t index = 0; index < centroid_count; ++index) {
        if (!double_vector(
                centroid_items[index], "centroids", &output->centroids[static_cast<std::size_t>(index)])) {
            Py_DECREF(centroid_sequence);
            Py_DECREF(tables);
            return false;
        }
    }
    Py_DECREF(centroid_sequence);
    Py_DECREF(tables);
    return true;
}

ModelHandle* model_handle(PyObject* capsule, bool require_value = true) {
    auto* handle = static_cast<ModelHandle*>(PyCapsule_GetPointer(capsule, kModelCapsuleName));
    if (handle == nullptr) {
        return nullptr;
    }
    if (require_value && !handle->has_value()) {
        PyErr_SetString(PyExc_RuntimeError, "resident inference model handle is closed");
        return nullptr;
    }
    return handle;
}

SessionHandle* session_handle(PyObject* capsule, bool require_value = true) {
    auto* handle = static_cast<SessionHandle*>(PyCapsule_GetPointer(capsule, kSessionCapsuleName));
    if (handle == nullptr) {
        return nullptr;
    }
    if (require_value && !handle->has_value()) {
        PyErr_SetString(PyExc_RuntimeError, "resident inference session handle is closed");
        return nullptr;
    }
    return handle;
}

bool owned_session(PyObject* model_capsule, PyObject* session_capsule, ModelHandle** model, SessionHandle** session) {
    *model = model_handle(model_capsule);
    *session = session_handle(session_capsule);
    if (*model == nullptr || *session == nullptr) {
        return false;
    }
    const bool owned = (*model)->identity != 0 &&
        (*model)->identity == (*session)->model_identity &&
        (*model)->kind == (*session)->kind &&
        ((*model)->kind == ModelHandle::Kind::Dense
            ? (*model)->dense && (*session)->dense &&
                (*session)->dense->model().get() == (*model)->dense.get()
            : ((*model)->kind == ModelHandle::Kind::Llama
                ? (*model)->llama && (*session)->llama &&
                    (*session)->llama->model().get() == (*model)->llama.get()
                : ((*model)->kind == ModelHandle::Kind::Glimmer
                    ? (*model)->glimmer && (*session)->glimmer &&
                        (*session)->glimmer->model().get() == (*model)->glimmer.get()
                    : (*model)->moe && (*session)->moe &&
                        (*session)->moe->model().get() == (*model)->moe.get())));
    if (!owned) {
        PyErr_SetString(PyExc_ValueError, "resident inference session does not belong to this model");
        return false;
    }
    return true;
}

void destroy_model_capsule(PyObject* capsule) {
    auto* handle = static_cast<ModelHandle*>(PyCapsule_GetPointer(capsule, kModelCapsuleName));
    if (handle != nullptr) {
        handle->close();
        delete handle;
    } else {
        PyErr_Clear();
    }
}

void destroy_session_capsule(PyObject* capsule) {
    auto* handle = static_cast<SessionHandle*>(PyCapsule_GetPointer(capsule, kSessionCapsuleName));
    if (handle != nullptr) {
        handle->close();
        delete handle;
    } else {
        PyErr_Clear();
    }
}

PyObject* return_cpp_error(const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
}

bool path_is_within(const std::filesystem::path& root, const std::filesystem::path& candidate) {
    auto root_it = root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != root.end(); ++root_it, ++candidate_it) {
        if (candidate_it == candidate.end() || *root_it != *candidate_it) {
            return false;
        }
    }
    return true;
}

std::string sha256_file(const std::filesystem::path& path) {
    OwnedPyObject hashlib(PyImport_ImportModule("hashlib"));
    OwnedPyObject constructor(
        hashlib ? PyObject_GetAttrString(hashlib.get(), "sha256") : nullptr);
    OwnedPyObject digest(
        constructor && PyCallable_Check(constructor.get())
            ? PyObject_CallNoArgs(constructor.get())
            : nullptr);
    OwnedPyObject update(
        digest ? PyObject_GetAttrString(digest.get(), "update") : nullptr);
    if (!digest || !update || !PyCallable_Check(update.get())) {
        throw std::runtime_error("resident binding could not initialize SHA-256 verification");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("resident binding could not open checkpoint for SHA-256 verification");
    }
    std::array<char, 64 * 1024> chunk{};
    while (input) {
        input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            OwnedPyObject payload(PyBytes_FromStringAndSize(
                chunk.data(), static_cast<Py_ssize_t>(count)));
            OwnedPyObject ignored(
                payload ? PyObject_CallOneArg(update.get(), payload.get()) : nullptr);
            if (!ignored) {
                throw std::runtime_error(
                    "resident binding failed while hashing the checkpoint");
            }
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("resident binding failed while reading the checkpoint");
    }
    OwnedPyObject hexdigest_method(PyObject_GetAttrString(digest.get(), "hexdigest"));
    OwnedPyObject hexdigest(
        hexdigest_method && PyCallable_Check(hexdigest_method.get())
            ? PyObject_CallNoArgs(hexdigest_method.get())
            : nullptr);
    std::string result;
    if (!hexdigest || !unicode_value(hexdigest.get(), "checkpoint SHA-256", &result)) {
        throw std::runtime_error("resident binding could not finish SHA-256 verification");
    }
    return result;
}

void require_checkpoint_sha256(
    const std::filesystem::path& path,
    std::string expected,
    const char* contract) {
    std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (sha256_file(path) != expected) {
        throw std::runtime_error(
            std::string(contract) + " checkpoint SHA-256 does not match its manifest fingerprint");
    }
}

bool is_sha256_hex(const std::string& value) {
    return value.size() == 64 && std::all_of(
        value.begin(), value.end(), [](unsigned char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f') ||
                (character >= 'A' && character <= 'F');
        });
}

DenseInferenceConfig validate_dense_model_contract(PyObject* model) {
    PyObject* spec = optional_item(model, "template_spec");
    if (spec == nullptr || !PyMapping_Check(spec)) {
        Py_XDECREF(spec);
        throw std::runtime_error("resident dense manifest model.template_spec must be an object");
    }
    PyObject* block = optional_item(spec, "block_spec");
    if (block == nullptr || !PyMapping_Check(block)) {
        Py_XDECREF(block);
        Py_DECREF(spec);
        throw std::runtime_error("resident dense manifest template_spec.block_spec must be an object");
    }
    const std::vector<std::pair<const char*, const char*>> string_fields = {
        {"norm_type", "layernorm"},
        {"mlp_type", "gelu"},
        {"pos_encoding", "absolute"},
        {"attention_variant", "dense"},
        {"residual_type", "add"},
        {"compression", "none"},
    };
    for (const auto& [field, expected] : string_fields) {
        std::string actual;
        if (!mapping_string(block, field, true, &actual)) {
            Py_DECREF(block);
            Py_DECREF(spec);
            throw std::runtime_error("resident dense manifest has an invalid block contract");
        }
        if (actual != expected) {
            Py_DECREF(block);
            Py_DECREF(spec);
            throw std::runtime_error(
                std::string("resident dense v5 binding requires block_spec.") + field +
                "='" + expected + "'");
        }
    }
    std::string activation_mode;
    if (!mapping_string(block, "activation_mode", true, &activation_mode) ||
        (activation_mode != "single" && activation_mode != "moa")) {
        Py_DECREF(block);
        Py_DECREF(spec);
        throw std::runtime_error(
            "resident dense v5 binding requires block_spec.activation_mode='single' or 'moa'");
    }
    const bool moa_mode = activation_mode == "moa";
    std::int64_t moa_interval = 0;
    if (moa_mode) {
        OwnedPyObject raw_candidates(optional_item(block, "moa_activations"));
        std::vector<std::string> candidates;
        if (!raw_candidates ||
            !string_vector(raw_candidates.get(), "block_spec.moa_activations", &candidates) ||
            candidates != std::vector<std::string>{"gelu", "relu", "silu", "relu2"} ||
            !mapping_int64(block, "moa_interval", true, 0, &moa_interval) ||
            moa_interval <= 0) {
            Py_DECREF(block);
            Py_DECREF(spec);
            throw std::runtime_error(
                "resident MoA requires candidates [gelu,relu,silu,relu2] and a positive interval");
        }
    }
    bool linear_bias = false;
    bool use_qk_norm = true;
    double dropout = 0.0;
    if (!mapping_bool(block, "linear_bias", true, false, &linear_bias) ||
        !mapping_bool(block, "use_qk_norm", true, true, &use_qk_norm) ||
        !mapping_double(block, "dropout_p", true, 0.0, &dropout)) {
        Py_DECREF(block);
        Py_DECREF(spec);
        throw std::runtime_error("resident dense manifest has invalid block flags");
    }
    if (!linear_bias || dropout != 0.0) {
        Py_DECREF(block);
        Py_DECREF(spec);
        throw std::runtime_error(
            "resident dense v5 binding requires biased linear layers and dropout_p=0");
    }
    std::int64_t num_heads = 0;
    if (!mapping_int64(block, "num_heads", true, 0, &num_heads) || num_heads <= 0) {
        Py_DECREF(block);
        Py_DECREF(spec);
        throw std::runtime_error("resident dense manifest has invalid num_heads");
    }
    PyObject* num_kv_heads = optional_item(block, "num_kv_heads");
    if (num_kv_heads != nullptr && num_kv_heads != Py_None) {
        std::int64_t parsed = 0;
        const bool valid = py_int64(num_kv_heads, "num_kv_heads", &parsed);
        Py_DECREF(num_kv_heads);
        if (!valid || parsed != num_heads) {
            Py_DECREF(block);
            Py_DECREF(spec);
            throw std::runtime_error("resident dense v5 binding requires MHA, not grouped-query attention");
        }
    } else {
        Py_XDECREF(num_kv_heads);
    }
    Py_DECREF(block);

    bool tie_embeddings = false;
    double logit_softcap = 0.0;
    if (!mapping_bool(spec, "tie_embeddings", true, false, &tie_embeddings) ||
        !mapping_double(spec, "logit_softcap", true, 0.0, &logit_softcap)) {
        Py_DECREF(spec);
        throw std::runtime_error("resident dense manifest has invalid output-head contract");
    }
    Py_DECREF(spec);
    if (!tie_embeddings || !std::isfinite(logit_softcap) || logit_softcap < 0.0) {
        throw std::runtime_error(
            "resident dense v5 binding requires tied embeddings and a finite non-negative logit_softcap");
    }
    if (moa_mode && (use_qk_norm || logit_softcap != 0.0)) {
        throw std::runtime_error(
            "resident MoA requires the reviewed dense attention path without QK norm or logit softcap");
    }
    DenseInferenceConfig config;
    config.use_qk_norm = use_qk_norm;
    config.qk_norm_eps = 1.0e-6;
    config.logit_softcap = logit_softcap;
    config.moa_mode = moa_mode;
    config.moa_interval = moa_interval;
    config.mlp_activation = moa_mode
        ? MlpActivation::GeluTanh
        : MlpActivation::GeluExact;
    return config;
}

void apply_dense_checkpoint_semantics(
    PyObject* manifest,
    PyObject* checkpoint,
    DenseInferenceConfig* config) {
    if (config == nullptr) {
        throw std::runtime_error("resident dense checkpoint semantics output is null");
    }
    OwnedPyObject moa(optional_item(checkpoint, "moa"));
    if (!config->moa_mode) {
        if (moa) {
            throw std::runtime_error(
                "resident single-activation dense graph cannot consume a MoA checkpoint contract");
        }
        config->mlp_activation = MlpActivation::GeluExact;
        return;
    }
    if (!moa || !PyMapping_Check(moa.get())) {
        throw std::runtime_error(
            "resident MoA requires a strict source-bound checkpoint.moa contract");
    }
    std::string schema;
    std::string preset;
    std::string selected_activation;
    std::string source_graph_sha256;
    std::string metadata_sha256;
    std::int64_t version = 0;
    std::int64_t interval = 0;
    if (!mapping_string(moa.get(), "schema", true, &schema) ||
        !mapping_int64(moa.get(), "version", true, 0, &version) ||
        !mapping_string(moa.get(), "preset", true, &preset) ||
        !mapping_string(moa.get(), "selected_activation", true, &selected_activation) ||
        !mapping_int64(moa.get(), "interval", true, 0, &interval) ||
        !mapping_string(moa.get(), "source_graph_sha256", true, &source_graph_sha256) ||
        !mapping_string(moa.get(), "metadata_sha256", true, &metadata_sha256) ||
        schema != "neuralfn.native_dense_moa.inference_checkpoint" ||
        version != 1 || preset != "gpt2_moa" || interval != config->moa_interval ||
        !is_sha256_hex(source_graph_sha256) || !is_sha256_hex(metadata_sha256)) {
        throw std::runtime_error("resident MoA checkpoint contract is invalid");
    }
    OwnedPyObject raw_candidates(optional_item(moa.get(), "candidate_activations"));
    std::vector<std::string> candidates;
    if (!raw_candidates ||
        !string_vector(raw_candidates.get(), "checkpoint.moa.candidate_activations", &candidates) ||
        candidates != std::vector<std::string>{"gelu", "relu", "silu", "relu2"}) {
        throw std::runtime_error("resident MoA checkpoint candidate set is not canonical");
    }
    OwnedPyObject source_graph(optional_item(manifest, "source_graph"));
    std::string manifest_graph_sha256;
    if (!source_graph || !PyMapping_Check(source_graph.get()) ||
        !mapping_string(source_graph.get(), "sha256", true, &manifest_graph_sha256) ||
        manifest_graph_sha256 != source_graph_sha256) {
        throw std::runtime_error(
            "resident MoA checkpoint source graph SHA-256 does not match its manifest");
    }
    if (selected_activation == "gelu") {
        config->mlp_activation = MlpActivation::GeluTanh;
    } else if (selected_activation == "relu") {
        config->mlp_activation = MlpActivation::Relu;
    } else if (selected_activation == "silu") {
        config->mlp_activation = MlpActivation::Silu;
    } else if (selected_activation == "relu2") {
        config->mlp_activation = MlpActivation::ReluSquared;
    } else {
        throw std::runtime_error("resident MoA checkpoint selected activation is unsupported");
    }
}

struct DenseTopologyNode {
    std::string path;
    std::string instance_id;
    std::string operation;
    double transform_parameter = 0.0;
};

struct DenseTopologyEdge {
    std::string src_node;
    std::int64_t src_port = 0;
    std::string dst_node;
    std::int64_t dst_port = 0;
};

struct DenseTopologyGraph {
    std::string path;
    std::vector<DenseTopologyNode> nodes;
    std::vector<DenseTopologyEdge> edges;
};

const DenseTopologyGraph* unique_topology_graph(
    const std::vector<DenseTopologyGraph>& graphs,
    const std::string& path) {
    const DenseTopologyGraph* match = nullptr;
    for (const auto& graph : graphs) {
        if (graph.path != path) {
            continue;
        }
        if (match != nullptr) {
            throw std::runtime_error(
                "resident dense topology contains a duplicate active graph path: " + path);
        }
        match = &graph;
    }
    if (match == nullptr) {
        throw std::runtime_error(
            "resident dense topology is missing the canonical active graph: " + path);
    }
    return match;
}

const DenseTopologyNode* unique_operation_node(
    const DenseTopologyGraph& graph,
    const std::string& operation) {
    const DenseTopologyNode* match = nullptr;
    for (const auto& node : graph.nodes) {
        if (node.operation != operation) {
            continue;
        }
        if (match != nullptr) {
            throw std::runtime_error(
                "resident dense topology requires one " + operation + " node in " + graph.path);
        }
        match = &node;
    }
    if (match == nullptr || match->path.empty()) {
        throw std::runtime_error(
            "resident dense topology requires one path-addressable " + operation +
            " node in " + graph.path);
    }
    return match;
}

const DenseTopologyNode* unique_instance_operation_node(
    const DenseTopologyGraph& graph,
    const std::string& instance_id,
    const std::string& operation) {
    const DenseTopologyNode* match = nullptr;
    for (const auto& node : graph.nodes) {
        if (node.instance_id != instance_id || node.operation != operation) {
            continue;
        }
        if (match != nullptr) {
            throw std::runtime_error(
                "resident dense topology contains a duplicate " + instance_id + " node in " +
                graph.path);
        }
        match = &node;
    }
    if (match == nullptr || match->path.empty()) {
        throw std::runtime_error(
            "resident dense topology requires " + instance_id + " (" + operation + ") in " +
            graph.path);
    }
    return match;
}

void require_unique_transform_edge(
    const DenseTopologyGraph& graph,
    const DenseTopologyNode& source,
    std::int64_t source_port,
    const DenseTopologyNode& destination,
    std::int64_t destination_port) {
    std::size_t exact = 0;
    std::size_t source_connections = 0;
    std::size_t destination_connections = 0;
    for (const auto& edge : graph.edges) {
        if (edge.src_node == source.path && edge.src_port == source_port) {
            ++source_connections;
        }
        if (edge.dst_node == destination.path && edge.dst_port == destination_port) {
            ++destination_connections;
        }
        if (edge.src_node == source.path && edge.src_port == source_port &&
            edge.dst_node == destination.path && edge.dst_port == destination_port) {
            ++exact;
        }
    }
    if (exact != 1 || source_connections != 1 || destination_connections != 1) {
        throw std::runtime_error(
            "resident dense parameter-free transform dataflow is not canonical in " + graph.path);
    }
}

void validate_dense_topology_contract(
    PyObject* manifest,
    PyObject* model,
    const DenseInferenceConfig& config) {
    OwnedPyObject spec(optional_item(model, "template_spec"));
    std::int64_t num_layers = 0;
    if (!spec || !PyMapping_Check(spec.get()) ||
        !mapping_int64(spec.get(), "num_layers", true, 0, &num_layers) ||
        num_layers <= 0) {
        throw std::runtime_error("resident dense manifest has invalid num_layers");
    }

    OwnedPyObject topology(optional_item(manifest, "topology"));
    if (!topology || !PyMapping_Check(topology.get())) {
        throw std::runtime_error("resident dense manifest topology must be an object");
    }
    OwnedPyObject raw_graphs(optional_item(topology.get(), "graphs"));
    OwnedPyObject graphs(raw_graphs
        ? PySequence_Fast(raw_graphs.get(), "resident dense topology.graphs must be an array")
        : nullptr);
    if (!graphs) {
        throw std::runtime_error("resident dense manifest topology.graphs must be an array");
    }

    std::vector<DenseTopologyGraph> active_graphs;
    std::int64_t qk_norm_nodes = 0;
    std::int64_t softcap_nodes = 0;
    const Py_ssize_t graph_count = PySequence_Fast_GET_SIZE(graphs.get());
    PyObject** graph_items = PySequence_Fast_ITEMS(graphs.get());
    for (Py_ssize_t graph_index = 0; graph_index < graph_count; ++graph_index) {
        PyObject* graph = graph_items[graph_index];
        if (!PyMapping_Check(graph)) {
            throw std::runtime_error("resident dense topology graph entries must be objects");
        }
        std::string path;
        if (!mapping_string(graph, "path", true, &path)) {
            throw std::runtime_error("resident dense topology graph has an invalid path");
        }
        if (path != "root" && path.rfind("root/", 0) != 0) {
            continue;
        }
        DenseTopologyGraph parsed_graph;
        parsed_graph.path = path;
        OwnedPyObject raw_nodes(optional_item(graph, "nodes"));
        OwnedPyObject nodes(raw_nodes
            ? PySequence_Fast(raw_nodes.get(), "resident dense topology nodes must be an array")
            : nullptr);
        if (!nodes) {
            throw std::runtime_error("resident dense topology graph nodes must be an array");
        }
        const Py_ssize_t node_count = PySequence_Fast_GET_SIZE(nodes.get());
        PyObject** node_items = PySequence_Fast_ITEMS(nodes.get());
        for (Py_ssize_t node_index = 0; node_index < node_count; ++node_index) {
            PyObject* node = node_items[node_index];
            if (!PyMapping_Check(node)) {
                throw std::runtime_error("resident dense topology node entries must be objects");
            }
            std::string operation;
            if (!mapping_string(node, "operation", true, &operation)) {
                throw std::runtime_error("resident dense topology node has an invalid operation");
            }
            DenseTopologyNode parsed_node;
            parsed_node.operation = operation;
            if (!mapping_string(node, "path", false, &parsed_node.path) ||
                !mapping_string(node, "instance_id", false, &parsed_node.instance_id)) {
                throw std::runtime_error("resident dense topology node has invalid identity fields");
            }
            if (operation == "qk_norm") {
                OwnedPyObject module_config(optional_item(node, "module_config"));
                if (!module_config || !PyMapping_Check(module_config.get())) {
                    throw std::runtime_error(
                        "resident dense QK topology node requires module_config");
                }
                double epsilon = 0.0;
                if (!mapping_double(module_config.get(), "eps", true, 0.0, &epsilon) ||
                    epsilon != config.qk_norm_eps) {
                    throw std::runtime_error(
                        "resident dense qk_norm topology requires eps=1e-6");
                }
                parsed_node.transform_parameter = epsilon;
                ++qk_norm_nodes;
            } else if (operation == "logit_softcap") {
                OwnedPyObject module_config(optional_item(node, "module_config"));
                if (!module_config || !PyMapping_Check(module_config.get())) {
                    throw std::runtime_error(
                        "resident dense softcap topology node requires module_config");
                }
                double softcap = 0.0;
                if (!mapping_double(module_config.get(), "softcap", true, 0.0, &softcap) ||
                    softcap != config.logit_softcap) {
                    throw std::runtime_error(
                        "resident dense logit_softcap topology does not match template_spec");
                }
                parsed_node.transform_parameter = softcap;
                ++softcap_nodes;
            }
            parsed_graph.nodes.push_back(std::move(parsed_node));
        }

        OwnedPyObject raw_edges(optional_item(graph, "edges"));
        if (raw_edges) {
            OwnedPyObject edges(PySequence_Fast(
                raw_edges.get(), "resident dense topology graph edges must be an array"));
            if (!edges) {
                throw std::runtime_error("resident dense topology graph edges must be an array");
            }
            const Py_ssize_t edge_count = PySequence_Fast_GET_SIZE(edges.get());
            PyObject** edge_items = PySequence_Fast_ITEMS(edges.get());
            parsed_graph.edges.reserve(static_cast<std::size_t>(edge_count));
            for (Py_ssize_t edge_index = 0; edge_index < edge_count; ++edge_index) {
                PyObject* edge = edge_items[edge_index];
                if (!PyMapping_Check(edge)) {
                    throw std::runtime_error(
                        "resident dense topology edge entries must be objects");
                }
                DenseTopologyEdge parsed_edge;
                if (!mapping_string(edge, "src_node", true, &parsed_edge.src_node) ||
                    !mapping_int64(edge, "src_port", true, 0, &parsed_edge.src_port) ||
                    !mapping_string(edge, "dst_node", true, &parsed_edge.dst_node) ||
                    !mapping_int64(edge, "dst_port", true, 0, &parsed_edge.dst_port) ||
                    parsed_edge.src_port < 0 || parsed_edge.dst_port < 0) {
                    throw std::runtime_error(
                        "resident dense topology edge has invalid endpoints or ports");
                }
                parsed_graph.edges.push_back(std::move(parsed_edge));
            }
        } else if (PyErr_Occurred()) {
            throw std::runtime_error("resident dense topology graph has invalid edges");
        }
        active_graphs.push_back(std::move(parsed_graph));
    }
    const std::int64_t expected_qk_norm_nodes = config.use_qk_norm ? num_layers : 0;
    const std::int64_t expected_softcap_nodes = config.logit_softcap > 0.0 ? 1 : 0;
    if (qk_norm_nodes != expected_qk_norm_nodes || softcap_nodes != expected_softcap_nodes) {
        throw std::runtime_error(
            "resident dense active topology does not match its QK normalization/softcap contract");
    }

    if (config.use_qk_norm) {
        for (std::int64_t layer = 0; layer < num_layers; ++layer) {
            const std::string path =
                "root/nodes/model/subgraph/nodes/block_" + std::to_string(layer) +
                "/subgraph/nodes/attention/subgraph";
            const auto* graph = unique_topology_graph(active_graphs, path);
            const auto* q_heads = unique_instance_operation_node(
                *graph, "q_heads", "reshape_heads");
            const auto* k_heads = unique_instance_operation_node(
                *graph, "k_heads", "reshape_heads");
            const auto* qk_norm = unique_operation_node(*graph, "qk_norm");
            const auto* sdpa = unique_operation_node(
                *graph, "scaled_dot_product_attention");
            require_unique_transform_edge(*graph, *q_heads, 0, *qk_norm, 0);
            require_unique_transform_edge(*graph, *k_heads, 0, *qk_norm, 1);
            require_unique_transform_edge(*graph, *qk_norm, 0, *sdpa, 0);
            require_unique_transform_edge(*graph, *qk_norm, 1, *sdpa, 1);
        }
    }

    if (config.logit_softcap > 0.0) {
        const auto* graph = unique_topology_graph(
            active_graphs, "root/nodes/model/subgraph");
        const auto* tied_head = unique_operation_node(*graph, "tied_lm_head");
        const auto* softcap = unique_operation_node(*graph, "logit_softcap");
        const auto* consumer = unique_operation_node(*graph, "token_cross_entropy");
        require_unique_transform_edge(*graph, *tied_head, 0, *softcap, 0);
        require_unique_transform_edge(*graph, *softcap, 0, *consumer, 0);
    }
}

std::filesystem::path checkpoint_from_manifest(
    const std::string& artifact_root,
    PyObject* manifest,
    DenseInferenceConfig* inference_config) {
    std::string schema;
    std::int64_t version = 0;
    if (!mapping_string(manifest, "schema", true, &schema) ||
        !mapping_int64(manifest, "version", true, 0, &version)) {
        throw std::runtime_error("resident manifest has an invalid schema declaration");
    }
    if (schema != "neuralfn.native_execution_manifest" || version != 1) {
        throw std::runtime_error("resident dense binding requires Native Execution Manifest version 1");
    }
    PyObject* capabilities = optional_item(manifest, "capabilities");
    if (capabilities == nullptr || !PyMapping_Check(capabilities)) {
        Py_XDECREF(capabilities);
        throw std::runtime_error("resident manifest capabilities must be an object");
    }
    bool native_inference = false;
    bool resident_inference = false;
    const bool capabilities_valid =
        mapping_bool(capabilities, "native_inference", true, false, &native_inference) &&
        mapping_bool(capabilities, "resident_inference", true, false, &resident_inference);
    Py_DECREF(capabilities);
    if (!capabilities_valid) {
        throw std::runtime_error("resident manifest has invalid inference capabilities");
    }
    if (!native_inference || !resident_inference) {
        throw std::runtime_error("resident manifest does not prove native resident inference");
    }
    PyObject* kernel_abi = optional_item(manifest, "kernel_abi");
    if (kernel_abi == nullptr || !PyMapping_Check(kernel_abi)) {
        Py_XDECREF(kernel_abi);
        throw std::runtime_error("resident manifest kernel_abi must be an object");
    }
    PyObject* resident_abi = optional_item(kernel_abi, "resident_inference");
    Py_DECREF(kernel_abi);
    if (resident_abi == nullptr || !PyMapping_Check(resident_abi)) {
        Py_XDECREF(resident_abi);
        throw std::runtime_error("resident manifest must declare the resident inference ABI");
    }
    std::int64_t resident_version = 0;
    std::string resident_status;
    const bool resident_abi_valid =
        mapping_int64(resident_abi, "version", true, 0, &resident_version) &&
        mapping_string(resident_abi, "status", true, &resident_status);
    Py_DECREF(resident_abi);
    if (!resident_abi_valid) {
        throw std::runtime_error("resident manifest has an invalid resident inference ABI");
    }
    if (resident_version != neuralfn::resident_dense::kResidentInferenceAbiVersion ||
        resident_status != "ready") {
        throw std::runtime_error("resident manifest ABI is not ready at version 1");
    }

    PyObject* model = optional_item(manifest, "model");
    if (model == nullptr || !PyMapping_Check(model)) {
        Py_XDECREF(model);
        throw std::runtime_error("resident manifest model must be an object");
    }
    std::string family;
    std::string family_class;
    if (!mapping_string(model, "family", true, &family) ||
        !mapping_string(model, "family_class", true, &family_class)) {
        Py_DECREF(model);
        throw std::runtime_error("resident manifest has invalid model identity");
    }
    const bool supported_family =
        family == "gpt" || family == "gpt2" || family == "gpt3" ||
        family == "nanogpt" || family == "gpt2-evo";
    if (!supported_family || family_class != "autoregressive_transformer") {
        Py_DECREF(model);
        throw std::runtime_error("resident dense v5 binding only supports proved dense GPT-family manifests");
    }
    DenseInferenceConfig parsed_config;
    try {
        parsed_config = validate_dense_model_contract(model);
        validate_dense_topology_contract(manifest, model, parsed_config);
    } catch (...) {
        Py_DECREF(model);
        throw;
    }
    Py_DECREF(model);
    if (inference_config == nullptr) {
        throw std::runtime_error("resident dense inference config output is null");
    }

    PyObject* checkpoint = optional_item(manifest, "checkpoint");
    if (checkpoint == nullptr || !PyMapping_Check(checkpoint)) {
        Py_XDECREF(checkpoint);
        throw std::runtime_error("resident manifest checkpoint must be an object");
    }
    std::string relative_path;
    std::string checkpoint_format;
    std::string target_sha256;
    std::int64_t target_nbytes = -1;
    if (!mapping_string(checkpoint, "format", true, &checkpoint_format)) {
        Py_DECREF(checkpoint);
        throw std::runtime_error("resident manifest checkpoint format must be a string");
    }
    if (checkpoint_format != "neuralfn.native_dense_gpt.v5") {
        Py_DECREF(checkpoint);
        throw std::runtime_error("resident dense binding requires checkpoint format neuralfn.native_dense_gpt.v5");
    }
    if (!mapping_int64(checkpoint, "target_nbytes", true, -1, &target_nbytes) ||
        !mapping_string(checkpoint, "target_sha256", true, &target_sha256)) {
        Py_DECREF(checkpoint);
        throw std::runtime_error("resident dense manifest has an invalid checkpoint fingerprint");
    }
    if (target_nbytes < 0 || !is_sha256_hex(target_sha256)) {
        Py_DECREF(checkpoint);
        throw std::runtime_error("resident dense manifest has an invalid checkpoint fingerprint");
    }
    if (!mapping_string(checkpoint, "artifact_path", false, &relative_path)) {
        Py_DECREF(checkpoint);
        throw std::runtime_error("resident manifest checkpoint artifact_path must be a string");
    }
    if (relative_path.empty() && !mapping_string(checkpoint, "resident_checkpoint", false, &relative_path)) {
        Py_DECREF(checkpoint);
        throw std::runtime_error("resident manifest checkpoint resident_checkpoint must be a string");
    }
    try {
        apply_dense_checkpoint_semantics(manifest, checkpoint, &parsed_config);
    } catch (...) {
        Py_DECREF(checkpoint);
        throw;
    }
    Py_DECREF(checkpoint);
    *inference_config = parsed_config;
    if (relative_path.empty()) {
        throw std::runtime_error(
            "resident dense manifest must declare checkpoint.artifact_path relative to the artifact root");
    }
    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("resident checkpoint artifact_path must be relative");
    }
    std::error_code error;
    const std::filesystem::path root = std::filesystem::weakly_canonical(artifact_root, error);
    if (error) {
        throw std::runtime_error("failed to resolve resident artifact root: " + error.message());
    }
    if (!std::filesystem::is_directory(root, error) || error) {
        throw std::runtime_error("resident artifact root is not a directory");
    }
    const std::filesystem::path candidate = std::filesystem::weakly_canonical(root / requested, error);
    if (error || !path_is_within(root, candidate)) {
        throw std::runtime_error("resident checkpoint artifact_path escapes the artifact root");
    }
    if (!std::filesystem::is_regular_file(candidate, error) || error) {
        throw std::runtime_error("resident checkpoint artifact_path is not a regular file");
    }
    const std::uintmax_t candidate_size = std::filesystem::file_size(candidate, error);
    if (error || candidate_size > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()) ||
        static_cast<std::int64_t>(candidate_size) != target_nbytes) {
        throw std::runtime_error("resident checkpoint length does not match its manifest fingerprint");
    }
    require_checkpoint_sha256(candidate, target_sha256, "resident dense");
    return candidate;
}

std::string checkpoint_format_from_manifest(PyObject* manifest) {
    OwnedPyObject checkpoint(optional_item(manifest, "checkpoint"));
    if (!checkpoint || !PyMapping_Check(checkpoint.get())) {
        throw std::runtime_error("resident manifest checkpoint must be an object");
    }
    std::string format;
    if (!mapping_string(checkpoint.get(), "format", true, &format)) {
        throw std::runtime_error("resident manifest checkpoint format must be a string");
    }
    return format;
}

void require_mapping_string_value(
    PyObject* mapping,
    const char* field,
    const char* expected,
    const char* contract) {
    std::string actual;
    if (!mapping_string(mapping, field, true, &actual) || actual != expected) {
        throw std::runtime_error(
            std::string(contract) + " requires " + field + "='" + expected + "'");
    }
}

void require_proof_boolean(PyObject* proof, const char* field, bool expected) {
    OwnedPyObject value(PyObject_GetAttrString(proof, field));
    if (!value || !PyBool_Check(value.get()) || (value.get() == Py_True) != expected) {
        throw std::runtime_error(
            std::string("resident LLaMA registry proof requires ") + field +
            (expected ? "=true" : "=false"));
    }
}

void validate_llama_registry_proof(PyObject* manifest) {
    OwnedPyObject model(optional_item(manifest, "model"));
    OwnedPyObject topology(optional_item(manifest, "topology"));
    if (!model || !topology || !PyMapping_Check(model.get()) ||
        !PyMapping_Check(topology.get())) {
        throw std::runtime_error(
            "resident LLaMA manifest requires model and topology objects");
    }
    OwnedPyObject registry(PyImport_ImportModule("neuralfn.native_registry"));
    OwnedPyObject capability_proof(
        registry ? PyObject_GetAttrString(registry.get(), "capability_proof_for") : nullptr);
    if (!capability_proof || !PyCallable_Check(capability_proof.get())) {
        throw std::runtime_error(
            "resident LLaMA binding could not load the dependency-light capability registry");
    }
    OwnedPyObject proof(PyObject_CallFunctionObjArgs(
        capability_proof.get(), model.get(), topology.get(), nullptr));
    if (!proof) {
        throw std::runtime_error(
            "resident LLaMA binding could not recompute the exact topology proof");
    }
    OwnedPyObject family(PyObject_GetAttrString(proof.get(), "model_family"));
    std::string family_name;
    if (!family || !unicode_value(family.get(), "proof.model_family", &family_name) ||
        family_name != "llama") {
        throw std::runtime_error(
            "resident LLaMA registry proof has the wrong model family");
    }
    for (const char* field : {
             "native_ir_lowering",
             "architecture_persistence_proven",
             "native_forward_proven",
             "resident_inference_proven",
             "lossless_cache_proven",
             "serving_proven",
         }) {
        require_proof_boolean(proof.get(), field, true);
    }
    require_proof_boolean(proof.get(), "turboquant_cache_proven", false);
}

std::int64_t checked_contract_mul(
    std::int64_t left,
    std::int64_t right,
    const char* field) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
        throw std::runtime_error(
            std::string("resident LLaMA tensor layout overflows at ") + field);
    }
    return left * right;
}

struct ExpectedLlamaTensor {
    std::string name;
    std::vector<std::int64_t> shape;
};

void validate_llama_tensor_table(
    PyObject* manifest,
    const LlamaInferenceConfig& config,
    std::int64_t target_nbytes) {
    const std::int64_t kv_width = checked_contract_mul(
        config.num_kv_heads, config.head_dim, "KV width");
    std::vector<ExpectedLlamaTensor> expected = {
        {"token_embedding.weight", {config.padded_vocab_size, config.model_dim}},
        {"final_norm.weight", {config.model_dim}},
        {"lm_head.weight", {config.padded_vocab_size, config.model_dim}},
    };
    expected.reserve(static_cast<std::size_t>(3 + 8 * config.num_layers));
    for (std::int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm.weight", {config.model_dim}});
        expected.push_back({prefix + "q_proj.weight", {config.model_dim, config.model_dim}});
        expected.push_back({prefix + "k_proj.weight", {kv_width, config.model_dim}});
        expected.push_back({prefix + "v_proj.weight", {kv_width, config.model_dim}});
        expected.push_back({prefix + "attention_out.weight", {config.model_dim, config.model_dim}});
        expected.push_back({prefix + "ffn_norm.weight", {config.model_dim}});
        expected.push_back(
            {prefix + "ffn_gate_up.weight", {2, config.hidden_dim, config.model_dim}});
        expected.push_back({prefix + "ffn_down.weight", {config.model_dim, config.hidden_dim}});
    }

    OwnedPyObject raw_tensors(optional_item(manifest, "tensors"));
    OwnedPyObject tensors(
        raw_tensors ? PySequence_Fast(
            raw_tensors.get(), "resident LLaMA manifest tensors must be an array") : nullptr);
    if (!tensors || PySequence_Fast_GET_SIZE(tensors.get()) !=
            static_cast<Py_ssize_t>(expected.size())) {
        throw std::runtime_error(
            "resident LLaMA manifest tensor table does not match the canonical layout");
    }
    std::int64_t expected_offset = 0;
    PyObject** rows = PySequence_Fast_ITEMS(tensors.get());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        PyObject* row = rows[index];
        if (!PyMapping_Check(row)) {
            throw std::runtime_error("resident LLaMA tensor entries must be objects");
        }
        std::string name;
        std::string source_name;
        std::string dtype;
        std::string byte_order;
        std::string role;
        std::string tensor_sha256;
        std::int64_t offset = -1;
        std::int64_t nbytes = -1;
        if (!mapping_string(row, "name", true, &name) ||
            !mapping_string(row, "source_name", true, &source_name) ||
            !mapping_string(row, "dtype", true, &dtype) ||
            !mapping_string(row, "byte_order", true, &byte_order) ||
            !mapping_string(row, "role", true, &role) ||
            !mapping_string(row, "sha256", true, &tensor_sha256) ||
            !mapping_int64(row, "offset", true, -1, &offset) ||
            !mapping_int64(row, "nbytes", true, -1, &nbytes)) {
            throw std::runtime_error("resident LLaMA tensor entry is invalid");
        }
        const auto& expected_tensor = expected[index];
        if (name != expected_tensor.name || source_name != name || dtype != "float32" ||
            byte_order != "little" || role != "parameter" || offset != expected_offset ||
            tensor_sha256.size() != 64 ||
            !std::all_of(
                tensor_sha256.begin(), tensor_sha256.end(), [](unsigned char character) {
                    return (character >= '0' && character <= '9') ||
                        (character >= 'a' && character <= 'f');
                })) {
            throw std::runtime_error(
                "resident LLaMA tensor identity, encoding, offset, or checksum is not canonical");
        }
        OwnedPyObject raw_shape(optional_item(row, "shape"));
        OwnedPyObject shape(
            raw_shape ? PySequence_Fast(
                raw_shape.get(), "resident LLaMA tensor shape must be an array") : nullptr);
        if (!shape || PySequence_Fast_GET_SIZE(shape.get()) !=
                static_cast<Py_ssize_t>(expected_tensor.shape.size())) {
            throw std::runtime_error("resident LLaMA tensor shape is not canonical");
        }
        std::int64_t elements = 1;
        PyObject** dimensions = PySequence_Fast_ITEMS(shape.get());
        for (std::size_t dimension = 0; dimension < expected_tensor.shape.size(); ++dimension) {
            std::int64_t actual = 0;
            if (!py_int64(dimensions[dimension], "tensor.shape", &actual) ||
                actual != expected_tensor.shape[dimension]) {
                throw std::runtime_error("resident LLaMA tensor shape is not canonical");
            }
            elements = checked_contract_mul(elements, actual, "tensor shape");
        }
        const std::int64_t expected_nbytes = checked_contract_mul(
            elements, static_cast<std::int64_t>(sizeof(float)), "tensor bytes");
        if (nbytes != expected_nbytes ||
            expected_offset > std::numeric_limits<std::int64_t>::max() - nbytes) {
            throw std::runtime_error("resident LLaMA tensor byte extent is not canonical");
        }
        expected_offset += nbytes;
    }
    if (expected_offset != target_nbytes) {
        throw std::runtime_error(
            "resident LLaMA tensor table length does not match the checkpoint fingerprint");
    }
}

std::filesystem::path llama_checkpoint_from_manifest(
    const std::string& artifact_root,
    PyObject* manifest,
    LlamaInferenceConfig* output_config) {
    if (output_config == nullptr) {
        throw std::runtime_error("resident LLaMA inference config output is null");
    }
    std::string schema;
    std::int64_t version = 0;
    if (!mapping_string(manifest, "schema", true, &schema) ||
        !mapping_int64(manifest, "version", true, 0, &version) ||
        schema != "neuralfn.native_execution_manifest" || version != 1) {
        throw std::runtime_error(
            "resident LLaMA binding requires Native Execution Manifest version 1");
    }

    OwnedPyObject capabilities(optional_item(manifest, "capabilities"));
    bool native_inference = false;
    bool resident_inference = false;
    bool lossless_cache = false;
    bool turboquant_cache = true;
    if (!capabilities || !PyMapping_Check(capabilities.get()) ||
        !mapping_bool(capabilities.get(), "native_inference", true, false, &native_inference) ||
        !mapping_bool(capabilities.get(), "resident_inference", true, false, &resident_inference) ||
        !mapping_bool(capabilities.get(), "lossless_kv_cache", true, false, &lossless_cache) ||
        !mapping_bool(capabilities.get(), "turboquant_kv_cache", true, false, &turboquant_cache) ||
        !native_inference || !resident_inference || !lossless_cache || turboquant_cache) {
        throw std::runtime_error(
            "canonical LLaMA manifest must prove resident lossless inference and keep TurboQuant disabled");
    }

    OwnedPyObject kernel_abi(optional_item(manifest, "kernel_abi"));
    OwnedPyObject resident_abi(
        kernel_abi && PyMapping_Check(kernel_abi.get())
            ? optional_item(kernel_abi.get(), "resident_inference")
            : nullptr);
    std::int64_t resident_version = 0;
    std::string resident_status;
    if (!resident_abi || !PyMapping_Check(resident_abi.get()) ||
        !mapping_int64(resident_abi.get(), "version", true, 0, &resident_version) ||
        !mapping_string(resident_abi.get(), "status", true, &resident_status) ||
        resident_version != neuralfn::resident_dense::kResidentInferenceAbiVersion ||
        resident_status != "ready") {
        throw std::runtime_error("resident LLaMA manifest ABI is not ready at version 1");
    }

    OwnedPyObject model(optional_item(manifest, "model"));
    std::string family;
    std::string family_class;
    if (!model || !PyMapping_Check(model.get()) ||
        !mapping_string(model.get(), "family", true, &family) ||
        !mapping_string(model.get(), "family_class", true, &family_class) ||
        family != "llama" || family_class != "autoregressive_transformer") {
        throw std::runtime_error(
            "resident LLaMA binding requires the canonical llama autoregressive family");
    }
    OwnedPyObject spec(optional_item(model.get(), "template_spec"));
    OwnedPyObject block(
        spec && PyMapping_Check(spec.get())
            ? optional_item(spec.get(), "block_spec")
            : nullptr);
    OwnedPyObject templ(
        spec && PyMapping_Check(spec.get())
            ? optional_item(spec.get(), "template")
            : nullptr);
    if (!spec || !block || !templ || !PyMapping_Check(spec.get()) ||
        !PyMapping_Check(block.get()) || !PyMapping_Check(templ.get())) {
        throw std::runtime_error(
            "resident LLaMA manifest requires template_spec, block_spec, and template objects");
    }
    validate_llama_registry_proof(manifest);
    for (const auto& [field, expected] : std::vector<std::pair<const char*, const char*>>{
             {"norm_type", "rmsnorm"},
             {"mlp_type", "swiglu"},
             {"pos_encoding", "rope"},
             {"attention_variant", "dense"},
             {"attention_backend", "sdpa"},
             {"residual_type", "add"},
             {"compression", "none"},
             {"adapter_type", "none"},
             {"activation_mode", "single"},
         }) {
        require_mapping_string_value(block.get(), field, expected, "resident canonical LLaMA");
    }
    for (const auto& [field, expected] : std::vector<std::pair<const char*, const char*>>{
             {"backbone", "llama"},
             {"objective", "ar"},
             {"adapter", "none"},
             {"compression", "none"},
             {"sparsity", "dense"},
             {"router_mode", "none"},
         }) {
        require_mapping_string_value(templ.get(), field, expected, "resident canonical LLaMA");
    }
    std::string runtime;
    if (!mapping_string(templ.get(), "runtime", true, &runtime) ||
        (runtime != "eager" && runtime != "compile")) {
        throw std::runtime_error(
            "resident canonical LLaMA requires runtime='eager' or reviewed alias runtime='compile'");
    }
    bool linear_bias = true;
    bool use_qk_norm = true;
    bool causal = false;
    bool tie_embeddings = true;
    double dropout = -1.0;
    double logit_softcap = -1.0;
    if (!mapping_bool(block.get(), "linear_bias", true, true, &linear_bias) || linear_bias ||
        !mapping_bool(block.get(), "use_qk_norm", true, true, &use_qk_norm) || use_qk_norm ||
        !mapping_bool(block.get(), "is_causal", true, false, &causal) || !causal ||
        !mapping_double(block.get(), "dropout_p", true, -1.0, &dropout) || dropout != 0.0 ||
        !mapping_bool(spec.get(), "tie_embeddings", true, true, &tie_embeddings) || tie_embeddings ||
        !mapping_double(spec.get(), "logit_softcap", true, -1.0, &logit_softcap) ||
        logit_softcap != 0.0) {
        throw std::runtime_error(
            "resident canonical LLaMA requires causal biasless dropout-free untied semantics");
    }
    OwnedPyObject rope_scaling(optional_item(block.get(), "rope_scaling"));
    if (!rope_scaling || rope_scaling.get() != Py_None) {
        throw std::runtime_error("resident canonical LLaMA requires unscaled RoPE");
    }

    OwnedPyObject checkpoint(optional_item(manifest, "checkpoint"));
    std::string checkpoint_format;
    std::string relative_path;
    std::string target_sha256;
    std::int64_t target_nbytes = -1;
    if (!checkpoint || !PyMapping_Check(checkpoint.get()) ||
        !mapping_string(checkpoint.get(), "format", true, &checkpoint_format) ||
        checkpoint_format != "neuralfn.native_family_llama.f32.v1" ||
        !mapping_string(checkpoint.get(), "artifact_path", true, &relative_path) ||
        !mapping_string(checkpoint.get(), "target_sha256", true, &target_sha256) ||
        !mapping_int64(checkpoint.get(), "target_nbytes", true, -1, &target_nbytes)) {
        throw std::runtime_error(
            "resident LLaMA binding requires a fingerprinted native family float32 checkpoint");
    }
    const bool sha_is_hex = target_sha256.size() == 64 && std::all_of(
        target_sha256.begin(), target_sha256.end(), [](unsigned char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f') ||
                (character >= 'A' && character <= 'F');
        });
    if (relative_path.empty() || target_nbytes < 0 || !sha_is_hex) {
        throw std::runtime_error("resident LLaMA checkpoint fingerprint is invalid");
    }
    std::transform(
        target_sha256.begin(),
        target_sha256.end(),
        target_sha256.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    OwnedPyObject geometry(optional_item(checkpoint.get(), "geometry"));
    if (!geometry || !PyMapping_Check(geometry.get())) {
        throw std::runtime_error("resident LLaMA checkpoint geometry must be an object");
    }
    LlamaInferenceConfig config;
    if (!mapping_int64(geometry.get(), "max_seq_len", true, 0, &config.max_seq_len) ||
        !mapping_int64(geometry.get(), "vocab_size", true, 0, &config.vocab_size) ||
        !mapping_int64(geometry.get(), "padded_vocab_size", true, 0, &config.padded_vocab_size) ||
        !mapping_int64(geometry.get(), "num_layers", true, 0, &config.num_layers) ||
        !mapping_int64(geometry.get(), "model_dim", true, 0, &config.model_dim) ||
        !mapping_int64(geometry.get(), "hidden_dim", true, 0, &config.hidden_dim) ||
        !mapping_int64(geometry.get(), "num_heads", true, 0, &config.num_heads) ||
        !mapping_int64(geometry.get(), "num_kv_heads", true, 0, &config.num_kv_heads) ||
        !mapping_int64(geometry.get(), "head_dim", true, 0, &config.head_dim) ||
        !mapping_double(geometry.get(), "rope_theta", true, 0.0, &config.rope_theta) ||
        !mapping_double(
            geometry.get(),
            "rope_scaling_factor",
            true,
            0.0,
            &config.rope_scaling_factor) ||
        !mapping_double(geometry.get(), "rms_norm_eps", true, 0.0, &config.rms_norm_eps)) {
        throw std::runtime_error("resident LLaMA checkpoint geometry is invalid");
    }

    OwnedPyObject semantics(optional_item(checkpoint.get(), "semantics"));
    if (!semantics || !PyMapping_Check(semantics.get())) {
        throw std::runtime_error("resident LLaMA checkpoint semantics must be an object");
    }
    for (const auto& [field, expected] : std::vector<std::pair<const char*, const char*>>{
             {"norm_type", "rmsnorm"},
             {"mlp_type", "swiglu"},
             {"pos_encoding", "rope"},
             {"attention_variant", "dense"},
             {"residual_type", "add"},
         }) {
        require_mapping_string_value(
            semantics.get(), field, expected, "resident LLaMA checkpoint semantics");
    }
    bool checkpoint_bias = true;
    bool checkpoint_tied = true;
    double checkpoint_dropout = -1.0;
    if (!mapping_bool(semantics.get(), "linear_bias", true, true, &checkpoint_bias) ||
        checkpoint_bias ||
        !mapping_bool(semantics.get(), "tie_embeddings", true, true, &checkpoint_tied) ||
        checkpoint_tied ||
        !mapping_double(
            semantics.get(), "dropout_p", true, -1.0, &checkpoint_dropout) ||
        checkpoint_dropout != 0.0) {
        throw std::runtime_error(
            "resident LLaMA checkpoint semantics must be biasless, untied, and dropout-free");
    }

    std::int64_t spec_model_dim = 0;
    std::int64_t spec_layers = 0;
    std::int64_t spec_vocab = 0;
    std::int64_t spec_heads = 0;
    std::int64_t spec_kv_heads = 0;
    std::int64_t multiple_of = 0;
    double spec_rope_theta = 0.0;
    double mlp_multiplier = 0.0;
    if (!mapping_int64(spec.get(), "model_dim", true, 0, &spec_model_dim) ||
        !mapping_int64(spec.get(), "num_layers", true, 0, &spec_layers) ||
        !mapping_int64(spec.get(), "vocab_size", true, 0, &spec_vocab) ||
        !mapping_int64(block.get(), "num_heads", true, 0, &spec_heads) ||
        !mapping_int64(block.get(), "num_kv_heads", true, 0, &spec_kv_heads) ||
        !mapping_int64(block.get(), "multiple_of", true, 0, &multiple_of) ||
        !mapping_double(block.get(), "rope_theta", true, 0.0, &spec_rope_theta) ||
        !mapping_double(block.get(), "mlp_multiplier", true, 0.0, &mlp_multiplier) ||
        spec_model_dim != config.model_dim || spec_layers != config.num_layers ||
        spec_vocab != config.vocab_size || spec_heads != config.num_heads ||
        spec_kv_heads != config.num_kv_heads || spec_rope_theta != config.rope_theta) {
        throw std::runtime_error(
            "resident LLaMA manifest model geometry does not match its checkpoint contract");
    }
    const double raw_hidden = static_cast<double>(spec_model_dim) * mlp_multiplier;
    if (multiple_of <= 0 || !std::isfinite(raw_hidden) || raw_hidden < 0.0 ||
        raw_hidden > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("resident LLaMA manifest has invalid SwiGLU width metadata");
    }
    const std::int64_t unaligned_hidden = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(raw_hidden));
    if (unaligned_hidden > std::numeric_limits<std::int64_t>::max() - (multiple_of - 1)) {
        throw std::runtime_error("resident LLaMA manifest SwiGLU width overflows");
    }
    const std::int64_t graph_hidden = checked_contract_mul(
        (unaligned_hidden + multiple_of - 1) / multiple_of,
        multiple_of,
        "aligned SwiGLU width");
    if (graph_hidden != config.hidden_dim) {
        throw std::runtime_error(
            "resident LLaMA graph SwiGLU width does not match its checkpoint geometry");
    }

    OwnedPyObject context(optional_item(manifest, "context_limits"));
    std::int64_t max_context_tokens = 0;
    if (!context || !PyMapping_Check(context.get()) ||
        !mapping_int64(
            context.get(), "max_context_tokens", true, 0, &max_context_tokens) ||
        max_context_tokens != config.max_seq_len) {
        throw std::runtime_error(
            "resident LLaMA manifest context limit does not match its checkpoint contract");
    }

    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("resident LLaMA checkpoint artifact_path must be relative");
    }
    std::error_code filesystem_error;
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(artifact_root, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error)) {
        throw std::runtime_error("resident LLaMA artifact root is not a directory");
    }
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(root / requested, filesystem_error);
    if (filesystem_error || !path_is_within(root, candidate) ||
        !std::filesystem::is_regular_file(candidate, filesystem_error)) {
        throw std::runtime_error(
            "resident LLaMA checkpoint artifact_path is invalid or escapes the artifact root");
    }
    const std::uintmax_t candidate_size =
        std::filesystem::file_size(candidate, filesystem_error);
    if (filesystem_error ||
        candidate_size > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()) ||
        static_cast<std::int64_t>(candidate_size) != target_nbytes) {
        throw std::runtime_error(
            "resident LLaMA checkpoint length does not match its manifest fingerprint");
    }
    validate_llama_tensor_table(manifest, config, target_nbytes);
    require_checkpoint_sha256(candidate, target_sha256, "resident LLaMA");
    config.checkpoint_sha256 = target_sha256;
    *output_config = config;
    return candidate;
}

std::filesystem::path glimmer_checkpoint_from_manifest(
    const std::string& artifact_root,
    PyObject* manifest,
    GlimmerInferenceConfig* output_config) {
    if (output_config == nullptr) {
        throw std::runtime_error("resident Glimmer inference config output is null");
    }
    std::string schema;
    std::int64_t version = 0;
    if (!mapping_string(manifest, "schema", true, &schema) ||
        !mapping_int64(manifest, "version", true, 0, &version) ||
        schema != "neuralfn.native_execution_manifest" || version != 1) {
        throw std::runtime_error(
            "resident Glimmer binding requires Native Execution Manifest version 1");
    }

    OwnedPyObject capabilities(optional_item(manifest, "capabilities"));
    bool native_inference = false;
    bool resident_inference = false;
    bool lossless_cache = false;
    bool turboquant_cache = true;
    if (!capabilities || !PyMapping_Check(capabilities.get()) ||
        !mapping_bool(capabilities.get(), "native_inference", true, false, &native_inference) ||
        !mapping_bool(capabilities.get(), "resident_inference", true, false, &resident_inference) ||
        !mapping_bool(capabilities.get(), "lossless_kv_cache", true, false, &lossless_cache) ||
        !mapping_bool(capabilities.get(), "turboquant_kv_cache", true, false, &turboquant_cache) ||
        !native_inference || !resident_inference || !lossless_cache || turboquant_cache) {
        throw std::runtime_error(
            "Muse Glimmer manifest must prove native resident hybrid-cache inference and disable TurboQuant");
    }

    OwnedPyObject kernel_abi(optional_item(manifest, "kernel_abi"));
    OwnedPyObject resident_abi(
        kernel_abi && PyMapping_Check(kernel_abi.get())
            ? optional_item(kernel_abi.get(), "resident_inference")
            : nullptr);
    std::int64_t resident_version = 0;
    std::string resident_status;
    if (!resident_abi || !PyMapping_Check(resident_abi.get()) ||
        !mapping_int64(resident_abi.get(), "version", true, 0, &resident_version) ||
        !mapping_string(resident_abi.get(), "status", true, &resident_status) ||
        resident_version != neuralfn::resident_dense::kResidentInferenceAbiVersion ||
        resident_status != "ready") {
        throw std::runtime_error("resident Glimmer manifest ABI is not ready at version 1");
    }

    OwnedPyObject model(optional_item(manifest, "model"));
    std::string family;
    std::string family_class;
    if (!model || !PyMapping_Check(model.get()) ||
        !mapping_string(model.get(), "family", true, &family) ||
        !mapping_string(model.get(), "family_class", true, &family_class) ||
        family != "muse_glimmer" || family_class != "autoregressive_transformer") {
        throw std::runtime_error(
            "resident Glimmer binding requires the muse_glimmer autoregressive family");
    }
    OwnedPyObject spec(optional_item(model.get(), "template_spec"));
    OwnedPyObject block(
        spec && PyMapping_Check(spec.get()) ? optional_item(spec.get(), "block_spec") : nullptr);
    OwnedPyObject templ(
        spec && PyMapping_Check(spec.get()) ? optional_item(spec.get(), "template") : nullptr);
    if (!spec || !block || !templ || !PyMapping_Check(spec.get()) ||
        !PyMapping_Check(block.get()) || !PyMapping_Check(templ.get())) {
        throw std::runtime_error(
            "resident Glimmer manifest requires template_spec, block_spec, and template objects");
    }
    require_mapping_string_value(block.get(), "family", "muse_glimmer", "resident Glimmer");
    require_mapping_string_value(block.get(), "norm_type", "rmsnorm", "resident Glimmer");
    require_mapping_string_value(block.get(), "mlp_type", "swiglu", "resident Glimmer");
    require_mapping_string_value(block.get(), "norm_layout", "sandwich", "resident Glimmer");
    require_mapping_string_value(
        block.get(), "qk_norm_kind", "weightless_rms", "resident Glimmer");
    require_mapping_string_value(block.get(), "attention_gate", "sigmoid", "resident Glimmer");
    require_mapping_string_value(templ.get(), "backbone", "muse_glimmer", "resident Glimmer");
    require_mapping_string_value(templ.get(), "objective", "ar", "resident Glimmer");
    require_mapping_string_value(templ.get(), "adapter", "none", "resident Glimmer");

    GlimmerInferenceConfig config;
    std::int64_t attention_inner_dim = 0;
    std::int64_t attention_gate_dim = 0;
    std::int64_t spec_context = 0;
    bool tied = true;
    bool causal = false;
    bool linear_bias = true;
    bool qk_norm = false;
    bool centered_norm = false;
    double dropout = -1.0;
    if (!mapping_int64(spec.get(), "model_dim", true, 0, &config.model_dim) ||
        !mapping_int64(spec.get(), "num_layers", true, 0, &config.num_layers) ||
        !mapping_int64(spec.get(), "vocab_size", true, 0, &config.vocab_size) ||
        !mapping_int64(spec.get(), "max_position_embeddings", true, 0, &spec_context) ||
        !mapping_bool(spec.get(), "tie_embeddings", true, true, &tied) || tied ||
        !mapping_double(spec.get(), "output_multiplier", true, 0.0, &config.output_multiplier) ||
        !mapping_double(spec.get(), "logit_softcap", true, 0.0, &config.logit_softcap) ||
        !mapping_int64(block.get(), "num_heads", true, 0, &config.num_heads) ||
        !mapping_int64(block.get(), "num_kv_heads", true, 0, &config.num_kv_heads) ||
        !mapping_int64(block.get(), "head_dim", true, 0, &config.head_dim) ||
        !mapping_int64(block.get(), "attention_inner_dim", true, 0, &attention_inner_dim) ||
        !mapping_int64(block.get(), "intermediate_size", true, 0, &config.intermediate_dim) ||
        !mapping_int64(block.get(), "attention_gate_dim", true, 0, &attention_gate_dim) ||
        !mapping_int64(block.get(), "window_size", true, 0, &config.sliding_window) ||
        !mapping_double(block.get(), "rope_theta", true, 0.0, &config.rope_theta) ||
        !mapping_double(block.get(), "norm_eps", true, 0.0, &config.norm_eps) ||
        !mapping_double(block.get(), "post_norm_eps", true, 0.0, &config.post_norm_eps) ||
        !mapping_double(block.get(), "q_scale_factor", true, 0.0, &config.q_scale_factor) ||
        !mapping_bool(block.get(), "is_causal", true, false, &causal) || !causal ||
        !mapping_bool(block.get(), "linear_bias", true, true, &linear_bias) || linear_bias ||
        !mapping_bool(block.get(), "use_qk_norm", true, false, &qk_norm) || !qk_norm ||
        !mapping_bool(block.get(), "centered_rms_norm", true, false, &centered_norm) ||
        !centered_norm ||
        !mapping_double(block.get(), "dropout_p", true, -1.0, &dropout) || dropout != 0.0 ||
        spec_context != 131072 || config.model_dim != 6656 || config.num_layers != 52 ||
        config.vocab_size != 202048 || config.num_heads != 32 || config.num_kv_heads != 2 ||
        config.head_dim != 128 || attention_inner_dim != 4096 ||
        attention_gate_dim != 4096 || config.intermediate_dim != 19968 ||
        config.sliding_window != 2048 || config.rope_theta != 500000.0 ||
        config.norm_eps != 1.0e-5 || config.post_norm_eps != 1.0e-8 ||
        config.q_scale_factor != 3.87 || config.output_multiplier != 0.19611613513818404 ||
        config.logit_softcap != 20.0) {
        throw std::runtime_error("resident Glimmer template geometry/semantics are not canonical");
    }
    config.max_seq_len = spec_context;

    OwnedPyObject raw_pattern(optional_item(block.get(), "layer_attention_pattern"));
    OwnedPyObject pattern(
        raw_pattern ? PySequence_Fast(
            raw_pattern.get(), "resident Glimmer layer_attention_pattern must be an array") : nullptr);
    if (!pattern || PySequence_Fast_GET_SIZE(pattern.get()) != 4) {
        throw std::runtime_error("resident Glimmer requires the exact three-local/one-global schedule");
    }
    PyObject** pattern_rows = PySequence_Fast_ITEMS(pattern.get());
    for (int index = 0; index < 4; ++index) {
        PyObject* row = pattern_rows[index];
        std::string kind;
        std::string pos_encoding;
        double theta = 0.0;
        if (!PyMapping_Check(row) || !mapping_string(row, "kind", true, &kind) ||
            !mapping_string(row, "pos_encoding", true, &pos_encoding) ||
            !mapping_double(row, "rope_theta", true, 0.0, &theta) ||
            theta != 500000.0 ||
            (index < 3 && (kind != "local" || pos_encoding != "rope")) ||
            (index == 3 && (kind != "full" || pos_encoding != "none"))) {
            throw std::runtime_error("resident Glimmer layer attention schedule is not canonical");
        }
        OwnedPyObject window(optional_item(row, "window_size"));
        if (index < 3) {
            std::int64_t parsed_window = 0;
            if (!window || !py_int64(window.get(), "window_size", &parsed_window) ||
                parsed_window != 2048) {
                throw std::runtime_error("resident Glimmer local layer window must be 2048");
            }
        } else if (!window || window.get() != Py_None) {
            throw std::runtime_error("resident Glimmer global layer must not declare a window");
        }
    }

    OwnedPyObject context(optional_item(manifest, "context_limits"));
    std::int64_t max_context_tokens = 0;
    if (!context || !PyMapping_Check(context.get()) ||
        !mapping_int64(context.get(), "max_context_tokens", true, 0, &max_context_tokens) ||
        max_context_tokens != config.max_seq_len) {
        throw std::runtime_error("resident Glimmer context limit is not 131072 tokens");
    }
    OwnedPyObject stop_tokens_raw(optional_item(manifest, "stop_tokens"));
    OwnedPyObject stop_tokens(
        stop_tokens_raw ? PySequence_Fast(
            stop_tokens_raw.get(), "resident Glimmer stop_tokens must be an array") : nullptr);
    if (!stop_tokens || PySequence_Fast_GET_SIZE(stop_tokens.get()) != 2) {
        throw std::runtime_error("resident Glimmer requires EOS/EOT stop tokens");
    }
    std::int64_t first_stop = 0;
    std::int64_t second_stop = 0;
    PyObject** stop_rows = PySequence_Fast_ITEMS(stop_tokens.get());
    if (!py_int64(stop_rows[0], "stop_tokens", &first_stop) ||
        !py_int64(stop_rows[1], "stop_tokens", &second_stop) ||
        first_stop != 200001 || second_stop != 200008) {
        throw std::runtime_error("resident Glimmer stop tokens must be [200001, 200008]");
    }

    OwnedPyObject checkpoint(optional_item(manifest, "checkpoint"));
    std::string format;
    std::string relative_path;
    std::string target_sha256;
    std::int64_t target_nbytes = -1;
    if (!checkpoint || !PyMapping_Check(checkpoint.get()) ||
        !mapping_string(checkpoint.get(), "format", true, &format) ||
        !mapping_string(checkpoint.get(), "artifact_path", true, &relative_path) ||
        !mapping_string(checkpoint.get(), "target_sha256", true, &target_sha256) ||
        !mapping_int64(checkpoint.get(), "target_nbytes", true, -1, &target_nbytes) ||
        relative_path.empty() || !is_sha256_hex(target_sha256)) {
        throw std::runtime_error("resident Glimmer checkpoint fingerprint is invalid");
    }
    std::transform(target_sha256.begin(), target_sha256.end(), target_sha256.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (format == "neuralfn.native_family_muse_glimmer.bf16.v1") {
        config.container = WeightContainer::NativeBf16;
        if (target_nbytes != 55'709'561'856LL && target_nbytes != 59'553'253'376LL) {
            throw std::runtime_error("resident Glimmer BF16 checkpoint has a noncanonical byte extent");
        }
    } else if (format == "neuralfn.native_family_muse_glimmer.gguf.kquant.v1") {
        config.container = WeightContainer::GgufKQuant;
        const bool compact = target_nbytes == 16'756'683'904LL &&
            target_sha256 == "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e" &&
            relative_path == "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf";
        const bool dynamic = target_nbytes == 19'653'960'832LL &&
            target_sha256 == "ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38" &&
            relative_path == "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf";
        if (!compact && !dynamic) {
            throw std::runtime_error("resident Glimmer K-Quant checkpoint is not a canonical official profile");
        }
    } else {
        throw std::runtime_error("resident Glimmer binding does not implement checkpoint format " + format);
    }

    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("resident Glimmer checkpoint artifact_path must be relative");
    }
    std::error_code filesystem_error;
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(artifact_root, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error)) {
        throw std::runtime_error("resident Glimmer artifact root is not a directory");
    }
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(root / requested, filesystem_error);
    if (filesystem_error || !path_is_within(root, candidate) ||
        !std::filesystem::is_regular_file(candidate, filesystem_error)) {
        throw std::runtime_error(
            "resident Glimmer checkpoint artifact_path is invalid or escapes the artifact root");
    }
    const std::uintmax_t actual_size = std::filesystem::file_size(candidate, filesystem_error);
    if (filesystem_error || actual_size != static_cast<std::uintmax_t>(target_nbytes)) {
        throw std::runtime_error("resident Glimmer checkpoint length does not match its fingerprint");
    }
    require_checkpoint_sha256(candidate, target_sha256, "resident Glimmer");
    config.checkpoint_sha256 = target_sha256;
    config.checkpoint_sha256_preverified = true;
    *output_config = config;
    return candidate;
}

std::filesystem::path glimmer_assistant_checkpoint_from_descriptor(
    const std::string& artifact_root,
    PyObject* descriptor,
    const GlimmerModel& target,
    GlimmerAssistantConfig* output_config) {
    if (output_config == nullptr || descriptor == nullptr || !PyMapping_Check(descriptor)) {
        throw std::runtime_error("DFlash companion descriptor/config is invalid");
    }
    std::string format;
    std::string relative_path;
    std::string target_sha256;
    std::int64_t target_nbytes = 0;
    if (!mapping_string(descriptor, "format", true, &format) ||
        !mapping_string(descriptor, "artifact_path", true, &relative_path) ||
        !mapping_string(descriptor, "target_sha256", true, &target_sha256) ||
        !mapping_int64(descriptor, "target_nbytes", true, 0, &target_nbytes) ||
        !is_sha256_hex(target_sha256)) {
        throw std::runtime_error("DFlash companion executable checkpoint fields are not canonical");
    }
    const bool bf16 =
        format == "neuralfn.native_family_muse_glimmer_dflash.bf16.v1" &&
        target_nbytes == 5'111'970'304LL && relative_path == "muse-glimmer-dflash.bf16";
    const bool kquant =
        format == "neuralfn.native_family_muse_glimmer_dflash.gguf.kquant.v1" &&
        target_nbytes == 1'631'208'128LL &&
        relative_path == "dflash-Muse-Glimmer-30B-Q4_K_M.gguf";
    if (!bf16 && !kquant) {
        throw std::runtime_error("DFlash companion format/path/extent is not canonical");
    }
    std::transform(target_sha256.begin(), target_sha256.end(), target_sha256.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    OwnedPyObject compatibility(optional_item(descriptor, "target_compatibility"));
    OwnedPyObject allowed(
        compatibility && PyMapping_Check(compatibility.get())
            ? optional_item(compatibility.get(), "allowed_target_checkpoint_sha256")
            : nullptr);
    OwnedPyObject target_layers(
        compatibility && PyMapping_Check(compatibility.get())
            ? optional_item(compatibility.get(), "target_layer_ids_zero_based")
            : nullptr);
    std::vector<std::int64_t> parsed_layers;
    std::int64_t block_size = 0;
    std::int64_t mask_token_id = 0;
    bool shared_lm_head = false;
    if (!compatibility || !PyMapping_Check(compatibility.get()) || !allowed ||
        !target_layers || !token_vector(target_layers.get(), "target_layer_ids_zero_based", &parsed_layers) ||
        parsed_layers != std::vector<std::int64_t>({1, 13, 25, 37, 49}) ||
        !mapping_int64(compatibility.get(), "block_size", true, 0, &block_size) ||
        block_size != 16 ||
        !mapping_int64(compatibility.get(), "mask_token_id", true, 0, &mask_token_id) ||
        mask_token_id != 201818 ||
        !mapping_bool(compatibility.get(), "shared_lm_head", true, false, &shared_lm_head) ||
        !shared_lm_head) {
        throw std::runtime_error("DFlash target compatibility contract is not canonical");
    }
    OwnedPyObject allowed_sequence(PySequence_Fast(
        allowed.get(), "allowed_target_checkpoint_sha256 must be an array"));
    if (!allowed_sequence || PySequence_Fast_GET_SIZE(allowed_sequence.get()) < 1) {
        throw std::runtime_error("DFlash companion has no compatible target digest");
    }
    bool target_allowed = false;
    PyObject** allowed_rows = PySequence_Fast_ITEMS(allowed_sequence.get());
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(allowed_sequence.get()); ++index) {
        std::string digest;
        if (!unicode_value(allowed_rows[index], "allowed target digest", &digest) ||
            !is_sha256_hex(digest)) {
            throw std::runtime_error("DFlash compatible target digest is invalid");
        }
        std::transform(digest.begin(), digest.end(), digest.begin(),
            [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        target_allowed = target_allowed || digest == target.checkpoint_sha256();
    }
    if (!target_allowed) {
        throw std::runtime_error("DFlash companion is not bound to the loaded target checkpoint digest");
    }

    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("DFlash artifact_path must be relative");
    }
    std::error_code filesystem_error;
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(artifact_root, filesystem_error);
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(root / requested, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error) ||
        !path_is_within(root, candidate) ||
        !std::filesystem::is_regular_file(candidate, filesystem_error) ||
        std::filesystem::file_size(candidate, filesystem_error) !=
            static_cast<std::uintmax_t>(target_nbytes)) {
        throw std::runtime_error("DFlash companion path/extent is invalid or escapes its artifact root");
    }
    require_checkpoint_sha256(candidate, target_sha256, "DFlash companion");
    GlimmerAssistantConfig config;
    config.container = kquant
        ? neuralfn::resident_glimmer_assistant::WeightContainer::GgufKQuant
        : neuralfn::resident_glimmer_assistant::WeightContainer::NativeBf16;
    config.checkpoint_sha256 = target_sha256;
    config.checkpoint_sha256_preverified = true;
    *output_config = std::move(config);
    return candidate;
}

std::filesystem::path glimmer_vision_checkpoint_from_descriptor(
    const std::string& artifact_root,
    PyObject* descriptor,
    const GlimmerModel& target) {
    if (descriptor == nullptr || !PyMapping_Check(descriptor)) {
        throw std::runtime_error("mmproj companion descriptor is invalid");
    }
    std::string format;
    std::string relative_path;
    std::string target_sha256;
    std::int64_t target_nbytes = 0;
    if (!mapping_string(descriptor, "format", true, &format) ||
        !mapping_string(descriptor, "artifact_path", true, &relative_path) ||
        !mapping_string(descriptor, "target_sha256", true, &target_sha256) ||
        !mapping_int64(descriptor, "target_nbytes", true, 0, &target_nbytes) ||
        format != "neuralfn.native_family_muse_glimmer_mmproj.gguf.kquant.v1" ||
        relative_path != "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf" ||
        target_nbytes != 1'400'328'928LL ||
        target_sha256 != "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6") {
        throw std::runtime_error("mmproj companion executable fields are not canonical");
    }
    OwnedPyObject compatibility(optional_item(descriptor, "target_compatibility"));
    OwnedPyObject allowed(
        compatibility && PyMapping_Check(compatibility.get())
            ? optional_item(compatibility.get(), "allowed_target_checkpoint_sha256")
            : nullptr);
    OwnedPyObject media_ids(
        compatibility && PyMapping_Check(compatibility.get())
            ? optional_item(compatibility.get(), "media_token_ids")
            : nullptr);
    std::string target_config_sha;
    std::string processor_sha;
    std::string tokenizer_sha;
    std::string template_sha;
    std::string temporal_reduction;
    std::int64_t projection_width = 0;
    std::int64_t patch_size = 0;
    std::int64_t temporal_patch_size = 0;
    std::int64_t merge_size = 0;
    std::int64_t packed_patch_width = 0;
    std::int64_t image_token = 0;
    std::int64_t video_token = 0;
    std::int64_t patch_token = 0;
    if (!compatibility || !PyMapping_Check(compatibility.get()) || !allowed ||
        !media_ids || !PyMapping_Check(media_ids.get()) ||
        !mapping_string(compatibility.get(), "target_config_sha256", true, &target_config_sha) ||
        target_config_sha != "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14" ||
        !mapping_string(compatibility.get(), "processor_config_sha256", true, &processor_sha) ||
        processor_sha != "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77" ||
        !mapping_string(compatibility.get(), "tokenizer_sha256", true, &tokenizer_sha) ||
        tokenizer_sha != "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d" ||
        !mapping_string(compatibility.get(), "chat_template_sha256", true, &template_sha) ||
        template_sha != "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678" ||
        !mapping_string(compatibility.get(), "temporal_patch_reduction", true, &temporal_reduction) ||
        temporal_reduction != "sum" ||
        !mapping_int64(compatibility.get(), "projection_width", true, 0, &projection_width) ||
        projection_width != 6656 ||
        !mapping_int64(compatibility.get(), "patch_size", true, 0, &patch_size) ||
        patch_size != 14 ||
        !mapping_int64(compatibility.get(), "temporal_patch_size", true, 0, &temporal_patch_size) ||
        temporal_patch_size != 2 ||
        !mapping_int64(compatibility.get(), "merge_size", true, 0, &merge_size) ||
        merge_size != 2 ||
        !mapping_int64(compatibility.get(), "packed_patch_width", true, 0, &packed_patch_width) ||
        packed_patch_width != 588 ||
        !mapping_int64(media_ids.get(), "image", true, 0, &image_token) ||
        image_token != 200090 ||
        !mapping_int64(media_ids.get(), "video", true, 0, &video_token) ||
        video_token != 200091 ||
        !mapping_int64(media_ids.get(), "patch", true, 0, &patch_token) ||
        patch_token != 200092) {
        throw std::runtime_error("mmproj target/processor compatibility contract is not canonical");
    }
    OwnedPyObject allowed_sequence(PySequence_Fast(
        allowed.get(), "allowed_target_checkpoint_sha256 must be an array"));
    if (!allowed_sequence || PySequence_Fast_GET_SIZE(allowed_sequence.get()) < 1) {
        throw std::runtime_error("mmproj companion has no compatible target digest");
    }
    bool target_allowed = false;
    PyObject** allowed_rows = PySequence_Fast_ITEMS(allowed_sequence.get());
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(allowed_sequence.get()); ++index) {
        std::string digest;
        if (!unicode_value(allowed_rows[index], "allowed target digest", &digest) ||
            !is_sha256_hex(digest)) {
            throw std::runtime_error("mmproj compatible target digest is invalid");
        }
        std::transform(digest.begin(), digest.end(), digest.begin(),
            [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        target_allowed = target_allowed || digest == target.checkpoint_sha256();
    }
    if (!target_allowed) {
        throw std::runtime_error("mmproj companion is not bound to the loaded target digest");
    }

    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("mmproj artifact_path must be relative");
    }
    std::error_code filesystem_error;
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(artifact_root, filesystem_error);
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(root / requested, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error) ||
        !path_is_within(root, candidate) ||
        !std::filesystem::is_regular_file(candidate, filesystem_error) ||
        std::filesystem::file_size(candidate, filesystem_error) !=
            static_cast<std::uintmax_t>(target_nbytes)) {
        throw std::runtime_error("mmproj path/extent is invalid or escapes its artifact root");
    }
    require_checkpoint_sha256(candidate, target_sha256, "mmproj companion");
    return candidate;
}

std::filesystem::path glimmer_lora_checkpoint_from_descriptor(
    const std::string& artifact_root,
    PyObject* descriptor,
    const GlimmerModel& target,
    std::int64_t* output_rank,
    double* output_alpha,
    std::uint32_t* output_target_mask) {
    if (descriptor == nullptr || !PyMapping_Check(descriptor) || output_rank == nullptr ||
        output_alpha == nullptr || output_target_mask == nullptr) {
        throw std::runtime_error("Muse Glimmer LoRA companion descriptor is invalid");
    }
    std::string format;
    std::string relative_path;
    std::string target_sha256;
    std::int64_t target_nbytes = 0;
    std::int64_t resident_weight_bytes = 0;
    std::int64_t rank = 0;
    double alpha = 0.0;
    double scaling = 0.0;
    double dropout = -1.0;
    if (!mapping_string(descriptor, "format", true, &format) ||
        format != "neuralfn.native_muse_glimmer_lora.bf16.v1" ||
        !mapping_string(descriptor, "artifact_path", true, &relative_path) ||
        relative_path != "muse-glimmer-lora.bf16" ||
        !mapping_string(descriptor, "target_sha256", true, &target_sha256) ||
        !is_sha256_hex(target_sha256) ||
        !mapping_int64(descriptor, "target_nbytes", true, 0, &target_nbytes) ||
        !mapping_int64(descriptor, "resident_weight_bytes", true, 0, &resident_weight_bytes) ||
        target_nbytes <= 0 || resident_weight_bytes != target_nbytes ||
        !mapping_int64(descriptor, "rank", true, 0, &rank) || rank <= 0 ||
        rank > target.model_dim() ||
        !mapping_double(descriptor, "alpha", true, 0.0, &alpha) || !(alpha > 0.0) ||
        !mapping_double(descriptor, "scaling", true, 0.0, &scaling) ||
        scaling != alpha / static_cast<double>(rank) ||
        !mapping_double(descriptor, "dropout", true, -1.0, &dropout) ||
        dropout < 0.0 || dropout >= 1.0) {
        throw std::runtime_error("Muse Glimmer LoRA executable fields are not canonical");
    }
    OwnedPyObject compatibility(optional_item(descriptor, "target_compatibility"));
    OwnedPyObject allowed(
        compatibility && PyMapping_Check(compatibility.get())
            ? optional_item(compatibility.get(), "allowed_target_checkpoint_sha256")
            : nullptr);
    OwnedPyObject targets(optional_item(descriptor, "targets"));
    std::string base_precision;
    std::string topology_sha;
    std::string graph_sha;
    std::string tokenizer_sha;
    std::string template_sha;
    if (!compatibility || !PyMapping_Check(compatibility.get()) || !allowed || !targets ||
        !mapping_string(compatibility.get(), "base_weight_precision", true, &base_precision) ||
        base_precision != "bf16" ||
        !mapping_string(descriptor, "graph_topology_sha256", true, &topology_sha) ||
        topology_sha != "4e3c741890fefc43e1027c61ffcfe9ec09c8f6448ee27dd548ab3fad3c172a56" ||
        !mapping_string(descriptor, "graph_fingerprint", true, &graph_sha) ||
        !is_sha256_hex(graph_sha) ||
        !mapping_string(compatibility.get(), "tokenizer_sha256", true, &tokenizer_sha) ||
        tokenizer_sha != "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d" ||
        !mapping_string(compatibility.get(), "chat_template_sha256", true, &template_sha) ||
        template_sha != "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678") {
        throw std::runtime_error("Muse Glimmer LoRA target compatibility contract is invalid");
    }
    OwnedPyObject allowed_sequence(PySequence_Fast(
        allowed.get(), "allowed_target_checkpoint_sha256 must be an array"));
    if (!allowed_sequence || PySequence_Fast_GET_SIZE(allowed_sequence.get()) != 1) {
        throw std::runtime_error("Muse Glimmer LoRA must bind exactly one base checkpoint digest");
    }
    std::string base_digest;
    if (!unicode_value(
            PySequence_Fast_ITEMS(allowed_sequence.get())[0], "LoRA base digest", &base_digest) ||
        !is_sha256_hex(base_digest)) {
        throw std::runtime_error("Muse Glimmer LoRA base checkpoint digest is invalid");
    }
    std::transform(base_digest.begin(), base_digest.end(), base_digest.begin(),
        [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (base_digest != target.checkpoint_sha256()) {
        throw std::runtime_error("Muse Glimmer LoRA is bound to a different base checkpoint");
    }
    OwnedPyObject target_sequence(PySequence_Fast(
        targets.get(), "Muse Glimmer LoRA targets must be an array"));
    if (!target_sequence || PySequence_Fast_GET_SIZE(target_sequence.get()) < 1 ||
        PySequence_Fast_GET_SIZE(target_sequence.get()) > 8) {
        throw std::runtime_error("Muse Glimmer LoRA target list is invalid");
    }
    const std::unordered_map<std::string, std::uint32_t> target_bits{
        {"q_proj", 1U << 0U}, {"k_proj", 1U << 1U}, {"v_proj", 1U << 2U},
        {"o_proj", 1U << 3U}, {"attn_gate_proj", 1U << 4U},
        {"gate_proj", 1U << 5U}, {"up_proj", 1U << 6U},
        {"down_proj", 1U << 7U},
    };
    std::uint32_t target_mask = 0;
    PyObject** target_rows = PySequence_Fast_ITEMS(target_sequence.get());
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(target_sequence.get()); ++index) {
        std::string name;
        if (!unicode_value(target_rows[index], "Muse Glimmer LoRA target", &name)) {
            throw std::runtime_error("Muse Glimmer LoRA target is not a string");
        }
        const auto bit = target_bits.find(name);
        if (bit == target_bits.end() || (target_mask & bit->second) != 0) {
            throw std::runtime_error("Muse Glimmer LoRA target is unsupported or duplicated");
        }
        target_mask |= bit->second;
    }
    auto site_parameters = [&](std::int64_t rows, std::int64_t cols) {
        return rank * (rows + cols);
    };
    const std::int64_t d = target.model_dim();
    const std::int64_t q = target.num_heads() * target.head_dim();
    const std::int64_t kv = target.num_kv_heads() * target.head_dim();
    const std::int64_t f = target.intermediate_dim();
    std::int64_t per_layer = 0;
    if (target_mask & (1U << 0U)) per_layer += site_parameters(q, d);
    if (target_mask & (1U << 1U)) per_layer += site_parameters(kv, d);
    if (target_mask & (1U << 2U)) per_layer += site_parameters(kv, d);
    if (target_mask & (1U << 3U)) per_layer += site_parameters(d, q);
    if (target_mask & (1U << 4U)) per_layer += site_parameters(q, d);
    if (target_mask & (1U << 5U)) per_layer += site_parameters(f, d);
    if (target_mask & (1U << 6U)) per_layer += site_parameters(f, d);
    if (target_mask & (1U << 7U)) per_layer += site_parameters(d, f);
    if (per_layer <= 0 || per_layer > std::numeric_limits<std::int64_t>::max() /
            target.num_layers() ||
        target_nbytes != per_layer * target.num_layers() * 2) {
        throw std::runtime_error("Muse Glimmer LoRA artifact byte extent is inconsistent");
    }
    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("Muse Glimmer LoRA artifact_path must be relative");
    }
    std::error_code filesystem_error;
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(artifact_root, filesystem_error);
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(root / requested, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error) ||
        !path_is_within(root, candidate) ||
        !std::filesystem::is_regular_file(candidate, filesystem_error) ||
        std::filesystem::file_size(candidate, filesystem_error) !=
            static_cast<std::uintmax_t>(target_nbytes)) {
        throw std::runtime_error("Muse Glimmer LoRA path/extent is invalid or escapes its artifact root");
    }
    require_checkpoint_sha256(candidate, target_sha256, "Muse Glimmer LoRA companion");
    *output_rank = rank;
    *output_alpha = alpha;
    *output_target_mask = target_mask;
    return candidate;
}

struct ExpectedMoeTensor {
    std::string name;
    std::vector<std::int64_t> shape;
};

void validate_standard_moe_tensor_table(
    PyObject* manifest,
    const MoeInferenceConfig& config,
    std::int64_t target_nbytes) {
    const std::int64_t kv_width = checked_contract_mul(
        config.num_kv_heads, config.head_dim, "standard-MoE KV width");
    std::vector<ExpectedMoeTensor> expected = {
        {"token_embedding.weight", {config.padded_vocab_size, config.model_dim}},
        {"final_norm.weight", {config.model_dim}},
        {"lm_head.weight", {config.padded_vocab_size, config.model_dim}},
    };
    expected.reserve(static_cast<std::size_t>(3 + 9 * config.num_layers));
    for (std::int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm.weight", {config.model_dim}});
        expected.push_back({prefix + "q_proj.weight", {config.model_dim, config.model_dim}});
        expected.push_back({prefix + "k_proj.weight", {kv_width, config.model_dim}});
        expected.push_back({prefix + "v_proj.weight", {kv_width, config.model_dim}});
        expected.push_back({prefix + "attention_out.weight", {config.model_dim, config.model_dim}});
        expected.push_back({prefix + "ffn_norm.weight", {config.model_dim}});
        expected.push_back({prefix + "router.weight", {config.experts, config.model_dim}});
        expected.push_back(
            {prefix + "experts.gate_up.weight",
             {2, config.experts, config.model_dim, config.hidden_dim}});
        expected.push_back(
            {prefix + "experts.down.weight",
             {config.experts, config.hidden_dim, config.model_dim}});
    }
    OwnedPyObject raw_tensors(optional_item(manifest, "tensors"));
    OwnedPyObject tensors(
        raw_tensors ? PySequence_Fast(
            raw_tensors.get(), "resident standard-MoE tensors must be an array") : nullptr);
    if (!tensors || PySequence_Fast_GET_SIZE(tensors.get()) !=
            static_cast<Py_ssize_t>(expected.size())) {
        throw std::runtime_error(
            "resident standard-MoE tensor table does not match the canonical layout");
    }
    std::int64_t expected_offset = 0;
    PyObject** rows = PySequence_Fast_ITEMS(tensors.get());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        PyObject* row = rows[index];
        if (!PyMapping_Check(row)) {
            throw std::runtime_error("resident standard-MoE tensor entries must be objects");
        }
        std::string name;
        std::string source_name;
        std::string dtype;
        std::string byte_order;
        std::string layout;
        std::string role;
        std::string tensor_sha256;
        std::int64_t offset = -1;
        std::int64_t nbytes = -1;
        if (!mapping_string(row, "name", true, &name) ||
            !mapping_string(row, "source_name", true, &source_name) ||
            !mapping_string(row, "dtype", true, &dtype) ||
            !mapping_string(row, "byte_order", true, &byte_order) ||
            !mapping_string(row, "layout", true, &layout) ||
            !mapping_string(row, "role", true, &role) ||
            !mapping_string(row, "sha256", true, &tensor_sha256) ||
            !mapping_int64(row, "offset", true, -1, &offset) ||
            !mapping_int64(row, "nbytes", true, -1, &nbytes)) {
            throw std::runtime_error("resident standard-MoE tensor entry is invalid");
        }
        const auto& expected_tensor = expected[index];
        if (name != expected_tensor.name || source_name != name || dtype != "float32" ||
            byte_order != "little" || layout != "row_major" || role != "parameter" ||
            offset != expected_offset ||
            tensor_sha256.size() != 64 ||
            !std::all_of(
                tensor_sha256.begin(), tensor_sha256.end(), [](unsigned char character) {
                    return (character >= '0' && character <= '9') ||
                        (character >= 'a' && character <= 'f');
                })) {
            throw std::runtime_error(
                "resident standard-MoE tensor identity, encoding, offset, or checksum is not canonical");
        }
        OwnedPyObject raw_shape(optional_item(row, "shape"));
        OwnedPyObject shape(
            raw_shape ? PySequence_Fast(
                raw_shape.get(), "resident standard-MoE tensor shape must be an array") : nullptr);
        if (!shape || PySequence_Fast_GET_SIZE(shape.get()) !=
                static_cast<Py_ssize_t>(expected_tensor.shape.size())) {
            throw std::runtime_error("resident standard-MoE tensor shape is not canonical");
        }
        std::int64_t elements = 1;
        PyObject** dimensions = PySequence_Fast_ITEMS(shape.get());
        for (std::size_t dimension = 0; dimension < expected_tensor.shape.size(); ++dimension) {
            std::int64_t actual = 0;
            if (!py_int64(dimensions[dimension], "tensor.shape", &actual) ||
                actual != expected_tensor.shape[dimension]) {
                throw std::runtime_error("resident standard-MoE tensor shape is not canonical");
            }
            elements = checked_contract_mul(elements, actual, "standard-MoE tensor shape");
        }
        const std::int64_t expected_nbytes = checked_contract_mul(
            elements, static_cast<std::int64_t>(sizeof(float)), "standard-MoE tensor bytes");
        if (nbytes != expected_nbytes ||
            expected_offset > std::numeric_limits<std::int64_t>::max() - nbytes) {
            throw std::runtime_error("resident standard-MoE tensor byte extent is not canonical");
        }
        expected_offset += nbytes;
    }
    if (expected_offset != target_nbytes) {
        throw std::runtime_error(
            "resident standard-MoE tensor table length does not match the checkpoint fingerprint");
    }
}

void require_standard_moe_source_graph(PyObject* checkpoint) {
    OwnedPyObject source_graph(optional_item(checkpoint, "source_graph"));
    if (!source_graph || !PyMapping_Check(source_graph.get())) {
        throw std::runtime_error(
            "resident standard-MoE checkpoint requires source-graph provenance");
    }
    std::string filename;
    std::string sha256;
    bool byte_identity = false;
    if (PyMapping_Size(source_graph.get()) != 3 ||
        !mapping_string(source_graph.get(), "filename", true, &filename) ||
        !mapping_string(source_graph.get(), "sha256", true, &sha256) ||
        !mapping_bool(
            source_graph.get(), "byte_identity_verified", true, false, &byte_identity) ||
        filename.empty() || filename.find('/') != std::string::npos ||
        filename.find('\\') != std::string::npos ||
        std::filesystem::path(filename).filename().string() != filename ||
        sha256.size() != 64 ||
        !std::all_of(sha256.begin(), sha256.end(), [](unsigned char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        }) || !byte_identity) {
        throw std::runtime_error(
            "resident standard-MoE source-graph provenance is not canonical");
    }
}

std::filesystem::path standard_moe_checkpoint_from_manifest(
    const std::string& artifact_root,
    PyObject* manifest,
    MoeInferenceConfig* output_config) {
    if (output_config == nullptr) {
        throw std::runtime_error("resident standard-MoE inference config output is null");
    }
    std::string schema;
    std::int64_t version = 0;
    if (!mapping_string(manifest, "schema", true, &schema) ||
        !mapping_int64(manifest, "version", true, 0, &version) ||
        schema != "neuralfn.native_execution_manifest" || version != 1) {
        throw std::runtime_error(
            "resident standard-MoE binding requires Native Execution Manifest version 1");
    }
    OwnedPyObject capabilities(optional_item(manifest, "capabilities"));
    bool native_inference = false;
    bool resident_inference = false;
    bool lossless_cache = false;
    bool turboquant_cache = true;
    if (!capabilities || !PyMapping_Check(capabilities.get()) ||
        !mapping_bool(capabilities.get(), "native_inference", true, false, &native_inference) ||
        !mapping_bool(capabilities.get(), "resident_inference", true, false, &resident_inference) ||
        !mapping_bool(capabilities.get(), "lossless_kv_cache", true, false, &lossless_cache) ||
        !mapping_bool(capabilities.get(), "turboquant_kv_cache", true, false, &turboquant_cache) ||
        !native_inference || !resident_inference || !lossless_cache || turboquant_cache) {
        throw std::runtime_error(
            "standard-MoE manifest must prove resident lossless inference and keep TurboQuant disabled");
    }
    OwnedPyObject kernel_abi(optional_item(manifest, "kernel_abi"));
    OwnedPyObject resident_abi(
        kernel_abi && PyMapping_Check(kernel_abi.get())
            ? optional_item(kernel_abi.get(), "resident_inference")
            : nullptr);
    std::int64_t resident_version = 0;
    std::string resident_status;
    if (!resident_abi || !PyMapping_Check(resident_abi.get()) ||
        !mapping_int64(resident_abi.get(), "version", true, 0, &resident_version) ||
        !mapping_string(resident_abi.get(), "status", true, &resident_status) ||
        resident_version != neuralfn::resident_dense::kResidentInferenceAbiVersion ||
        resident_status != "ready") {
        throw std::runtime_error("resident standard-MoE manifest ABI is not ready at version 1");
    }

    OwnedPyObject model(optional_item(manifest, "model"));
    std::string family;
    std::string family_class;
    if (!model || !PyMapping_Check(model.get()) ||
        !mapping_string(model.get(), "family", true, &family) ||
        !mapping_string(model.get(), "family_class", true, &family_class) ||
        family != "mixllama" || family_class != "autoregressive_transformer") {
        throw std::runtime_error(
            "resident standard-MoE binding requires the canonical mixllama autoregressive family");
    }
    OwnedPyObject spec(optional_item(model.get(), "template_spec"));
    OwnedPyObject block(
        spec && PyMapping_Check(spec.get()) ? optional_item(spec.get(), "block_spec") : nullptr);
    OwnedPyObject templ(
        spec && PyMapping_Check(spec.get()) ? optional_item(spec.get(), "template") : nullptr);
    if (!spec || !block || !templ || !PyMapping_Check(spec.get()) ||
        !PyMapping_Check(block.get()) || !PyMapping_Check(templ.get())) {
        throw std::runtime_error(
            "resident standard-MoE manifest requires template_spec, block_spec, and template objects");
    }
    for (const auto& [field, expected] : std::vector<std::pair<const char*, const char*>>{
             {"norm_type", "rmsnorm"}, {"mlp_type", "moe"},
             {"pos_encoding", "rope"}, {"attention_variant", "dense"},
             {"attention_backend", "sdpa"}, {"residual_type", "add"},
             {"compression", "none"}, {"adapter_type", "none"},
             {"activation_mode", "single"}, {"moe_balance_mode", "aux_loss"},
             {"router_score_fn", "softmax"},
         }) {
        require_mapping_string_value(block.get(), field, expected, "resident standard-MoE");
    }
    for (const auto& [field, expected] : std::vector<std::pair<const char*, const char*>>{
             {"backbone", "mixllama"}, {"objective", "ar"}, {"adapter", "none"},
             {"compression", "none"}, {"sparsity", "moe"}, {"router_mode", "none"},
         }) {
        require_mapping_string_value(templ.get(), field, expected, "resident standard-MoE");
    }
    std::string runtime;
    if (!mapping_string(templ.get(), "runtime", true, &runtime) ||
        (runtime != "eager" && runtime != "compile")) {
        throw std::runtime_error(
            "resident standard-MoE requires eager or reviewed compile runtime");
    }
    bool linear_bias = true;
    bool use_qk_norm = true;
    bool causal = false;
    bool tie_embeddings = true;
    double dropout = -1.0;
    double logit_softcap = -1.0;
    std::int64_t shared_experts = -1;
    if (!mapping_bool(block.get(), "linear_bias", true, true, &linear_bias) || linear_bias ||
        !mapping_bool(block.get(), "use_qk_norm", true, true, &use_qk_norm) || use_qk_norm ||
        !mapping_bool(block.get(), "is_causal", true, false, &causal) || !causal ||
        !mapping_double(block.get(), "dropout_p", true, -1.0, &dropout) || dropout != 0.0 ||
        !mapping_bool(spec.get(), "tie_embeddings", true, true, &tie_embeddings) || tie_embeddings ||
        !mapping_double(spec.get(), "logit_softcap", true, -1.0, &logit_softcap) ||
        logit_softcap != 0.0 ||
        !mapping_int64(block.get(), "shared_experts", true, -1, &shared_experts) ||
        shared_experts != 0) {
        throw std::runtime_error(
            "resident standard-MoE requires causal biasless dropout-free untied semantics without shared experts");
    }
    OwnedPyObject rope_scaling(optional_item(block.get(), "rope_scaling"));
    if (!rope_scaling || rope_scaling.get() != Py_None) {
        throw std::runtime_error("resident standard-MoE requires unscaled RoPE");
    }

    OwnedPyObject checkpoint(optional_item(manifest, "checkpoint"));
    std::string checkpoint_format;
    std::string relative_path;
    std::string target_sha256;
    std::string checkpoint_family;
    std::string preset;
    std::int64_t target_nbytes = -1;
    if (!checkpoint || !PyMapping_Check(checkpoint.get()) ||
        !mapping_string(checkpoint.get(), "format", true, &checkpoint_format) ||
        checkpoint_format != "neuralfn.native_family_standard_moe.f32.v1" ||
        !mapping_string(checkpoint.get(), "artifact_path", true, &relative_path) ||
        !mapping_string(checkpoint.get(), "target_sha256", true, &target_sha256) ||
        !mapping_int64(checkpoint.get(), "target_nbytes", true, -1, &target_nbytes) ||
        !mapping_string(checkpoint.get(), "family", true, &checkpoint_family) ||
        checkpoint_family != "mixllama" ||
        !mapping_string(checkpoint.get(), "preset", true, &preset) ||
        (preset != "mixllama" && preset != "moe" && preset != "mixllama-fast")) {
        throw std::runtime_error(
            "resident standard-MoE binding requires a canonical fingerprinted float32 checkpoint");
    }
    const std::string expected_runtime = preset == "mixllama-fast" ? "compile" : "eager";
    if (runtime != expected_runtime) {
        throw std::runtime_error(
            "resident standard-MoE preset/runtime identity is not canonical");
    }
    const bool sha_is_hex = target_sha256.size() == 64 && std::all_of(
        target_sha256.begin(), target_sha256.end(), [](unsigned char character) {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        });
    if (relative_path.empty() || target_nbytes <= 0 || !sha_is_hex) {
        throw std::runtime_error("resident standard-MoE checkpoint fingerprint is invalid");
    }
    require_standard_moe_source_graph(checkpoint.get());

    OwnedPyObject geometry(optional_item(checkpoint.get(), "geometry"));
    if (!geometry || !PyMapping_Check(geometry.get()) || PyMapping_Size(geometry.get()) != 17) {
        throw std::runtime_error("resident standard-MoE checkpoint geometry must be an object");
    }
    MoeInferenceConfig config;
    config.standard_moe = true;
    if (!mapping_int64(geometry.get(), "max_seq_len", true, 0, &config.max_seq_len) ||
        !mapping_int64(geometry.get(), "vocab_size", true, 0, &config.vocab_size) ||
        !mapping_int64(geometry.get(), "padded_vocab_size", true, 0, &config.padded_vocab_size) ||
        !mapping_int64(geometry.get(), "num_layers", true, 0, &config.num_layers) ||
        !mapping_int64(geometry.get(), "model_dim", true, 0, &config.model_dim) ||
        !mapping_int64(geometry.get(), "hidden_dim", true, 0, &config.hidden_dim) ||
        !mapping_int64(geometry.get(), "num_heads", true, 0, &config.num_heads) ||
        !mapping_int64(geometry.get(), "num_kv_heads", true, 0, &config.num_kv_heads) ||
        !mapping_int64(geometry.get(), "head_dim", true, 0, &config.head_dim) ||
        !mapping_int64(geometry.get(), "experts", true, 0, &config.experts) ||
        !mapping_int64(geometry.get(), "top_k", true, 0, &config.top_k) ||
        !mapping_int64(geometry.get(), "multiple_of", true, -1, &config.multiple_of) ||
        !mapping_double(geometry.get(), "rope_theta", true, 0.0, &config.rope_theta) ||
        !mapping_double(geometry.get(), "rope_scaling_factor", true, 0.0, &config.rope_scaling_factor) ||
        !mapping_double(geometry.get(), "rms_norm_eps", true, 0.0, &config.rms_norm_eps) ||
        !mapping_double(geometry.get(), "mlp_multiplier", true, 0.0, &config.mlp_multiplier) ||
        !mapping_double(
            geometry.get(), "router_aux_loss_coef", true, -1.0, &config.router_aux_loss_coef)) {
        throw std::runtime_error("resident standard-MoE checkpoint geometry is invalid");
    }
    OwnedPyObject semantics(optional_item(checkpoint.get(), "semantics"));
    if (!semantics || !PyMapping_Check(semantics.get()) || PyMapping_Size(semantics.get()) != 13) {
        throw std::runtime_error("resident standard-MoE checkpoint semantics must be an object");
    }
    for (const auto& [field, expected] : std::vector<std::pair<const char*, const char*>>{
             {"norm_type", "rmsnorm"}, {"mlp_type", "moe"}, {"pos_encoding", "rope"},
             {"attention_variant", "dense"}, {"residual_type", "add"},
             {"router_score_fn", "softmax"}, {"router_selection", "topk_renormalized"},
             {"moe_balance_mode", "aux_loss"},
         }) {
        require_mapping_string_value(
            semantics.get(), field, expected, "resident standard-MoE checkpoint semantics");
    }
    bool semantic_bias = true;
    bool semantic_tied = true;
    bool semantic_qk_norm = true;
    double semantic_dropout = -1.0;
    std::int64_t semantic_shared = -1;
    if (!mapping_bool(semantics.get(), "linear_bias", true, true, &semantic_bias) || semantic_bias ||
        !mapping_bool(semantics.get(), "tie_embeddings", true, true, &semantic_tied) || semantic_tied ||
        !mapping_bool(semantics.get(), "use_qk_norm", true, true, &semantic_qk_norm) || semantic_qk_norm ||
        !mapping_double(semantics.get(), "dropout_p", true, -1.0, &semantic_dropout) ||
        semantic_dropout != 0.0 ||
        !mapping_int64(semantics.get(), "shared_experts", true, -1, &semantic_shared) ||
        semantic_shared != 0) {
        throw std::runtime_error("resident standard-MoE checkpoint semantics are not canonical");
    }

    std::int64_t spec_model_dim = 0;
    std::int64_t spec_layers = 0;
    std::int64_t spec_vocab = 0;
    std::int64_t spec_heads = 0;
    std::int64_t spec_kv_heads = 0;
    std::int64_t spec_experts = 0;
    std::int64_t spec_top_k = 0;
    double spec_rope_theta = 0.0;
    double spec_multiplier = 0.0;
    double spec_aux = 0.0;
    if (!mapping_int64(spec.get(), "model_dim", true, 0, &spec_model_dim) ||
        !mapping_int64(spec.get(), "num_layers", true, 0, &spec_layers) ||
        !mapping_int64(spec.get(), "vocab_size", true, 0, &spec_vocab) ||
        !mapping_int64(block.get(), "num_heads", true, 0, &spec_heads) ||
        !mapping_int64(block.get(), "num_kv_heads", true, 0, &spec_kv_heads) ||
        !mapping_int64(block.get(), "experts", true, 0, &spec_experts) ||
        !mapping_int64(block.get(), "top_k", true, 0, &spec_top_k) ||
        !mapping_double(block.get(), "rope_theta", true, 0.0, &spec_rope_theta) ||
        !mapping_double(block.get(), "mlp_multiplier", true, 0.0, &spec_multiplier) ||
        !mapping_double(block.get(), "router_aux_loss_coef", true, 0.0, &spec_aux) ||
        spec_model_dim != config.model_dim || spec_layers != config.num_layers ||
        spec_vocab != config.vocab_size || spec_heads != config.num_heads ||
        spec_kv_heads != config.num_kv_heads || spec_experts != config.experts ||
        spec_top_k != config.top_k || spec_rope_theta != config.rope_theta ||
        spec_multiplier != config.mlp_multiplier || spec_aux != config.router_aux_loss_coef) {
        throw std::runtime_error(
            "resident standard-MoE manifest graph geometry does not match its checkpoint contract");
    }
    OwnedPyObject graph_multiple(optional_item(block.get(), "multiple_of"));
    std::int64_t graph_multiple_value = 0;
    if (!graph_multiple) {
        throw std::runtime_error("resident standard-MoE graph is missing multiple_of metadata");
    }
    if (graph_multiple.get() != Py_None &&
        !py_int64(graph_multiple.get(), "multiple_of", &graph_multiple_value)) {
        throw std::runtime_error("resident standard-MoE graph multiple_of is invalid");
    }
    if (graph_multiple_value != config.multiple_of) {
        throw std::runtime_error(
            "resident standard-MoE graph alignment does not match checkpoint geometry");
    }
    OwnedPyObject context(optional_item(manifest, "context_limits"));
    std::int64_t max_context_tokens = 0;
    if (!context || !PyMapping_Check(context.get()) ||
        !mapping_int64(context.get(), "max_context_tokens", true, 0, &max_context_tokens) ||
        max_context_tokens != config.max_seq_len) {
        throw std::runtime_error(
            "resident standard-MoE context limit does not match its checkpoint contract");
    }

    const std::filesystem::path requested(relative_path);
    if (requested.is_absolute()) {
        throw std::runtime_error("resident standard-MoE checkpoint artifact_path must be relative");
    }
    std::error_code filesystem_error;
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(artifact_root, filesystem_error);
    const std::filesystem::path candidate =
        std::filesystem::weakly_canonical(root / requested, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error) ||
        !path_is_within(root, candidate) ||
        !std::filesystem::is_regular_file(candidate, filesystem_error)) {
        throw std::runtime_error(
            "resident standard-MoE checkpoint path is invalid or escapes the artifact root");
    }
    const std::uintmax_t candidate_size = std::filesystem::file_size(candidate, filesystem_error);
    if (filesystem_error ||
        candidate_size > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()) ||
        static_cast<std::int64_t>(candidate_size) != target_nbytes) {
        throw std::runtime_error(
            "resident standard-MoE checkpoint length does not match its manifest fingerprint");
    }
    validate_standard_moe_tensor_table(manifest, config, target_nbytes);
    require_checkpoint_sha256(candidate, target_sha256, "resident standard-MoE");
    config.checkpoint_sha256 = target_sha256;
    *output_config = config;
    return candidate;
}

void validate_loaded_model_geometry(PyObject* manifest, const DenseModel& model_value) {
    PyObject* model = optional_item(manifest, "model");
    if (model == nullptr || !PyMapping_Check(model)) {
        Py_XDECREF(model);
        throw std::runtime_error("resident manifest model must be an object");
    }
    PyObject* spec = optional_item(model, "template_spec");
    Py_DECREF(model);
    if (spec == nullptr || !PyMapping_Check(spec)) {
        Py_XDECREF(spec);
        throw std::runtime_error("resident dense manifest model.template_spec must be an object");
    }
    PyObject* block = optional_item(spec, "block_spec");
    if (block == nullptr || !PyMapping_Check(block)) {
        Py_XDECREF(block);
        Py_DECREF(spec);
        throw std::runtime_error("resident dense manifest template_spec.block_spec must be an object");
    }
    std::int64_t model_dim = 0;
    std::int64_t num_layers = 0;
    std::int64_t vocab_size = 0;
    std::int64_t num_heads = 0;
    const bool valid =
        mapping_int64(spec, "model_dim", true, 0, &model_dim) &&
        mapping_int64(spec, "num_layers", true, 0, &num_layers) &&
        mapping_int64(spec, "vocab_size", true, 0, &vocab_size) &&
        mapping_int64(block, "num_heads", true, 0, &num_heads);
    Py_DECREF(block);
    Py_DECREF(spec);
    if (!valid) {
        throw std::runtime_error("resident dense manifest has invalid model geometry");
    }
    const auto stats = model_value.stats();
    if (model_dim != stats.channels || num_layers != stats.num_layers ||
        vocab_size != stats.vocab_size || num_heads != stats.num_heads) {
        throw std::runtime_error(
            "resident dense manifest model geometry does not match the native checkpoint header");
    }

    PyObject* context = optional_item(manifest, "context_limits");
    if (context == nullptr || !PyMapping_Check(context)) {
        Py_XDECREF(context);
        throw std::runtime_error("resident dense manifest context_limits must be an object");
    }
    std::int64_t max_context_tokens = 0;
    const bool context_valid = mapping_int64(
        context, "max_context_tokens", true, 0, &max_context_tokens);
    Py_DECREF(context);
    if (!context_valid) {
        throw std::runtime_error("resident dense manifest has an invalid max_context_tokens");
    }
    if (max_context_tokens != stats.max_seq_len) {
        throw std::runtime_error(
            "resident dense manifest max_context_tokens does not match the native checkpoint header");
    }
}

PyObject* resident_inference_abi_version(PyObject*, PyObject*) {
    return PyLong_FromLong(neuralfn::resident_dense::kResidentInferenceAbiVersion);
}

PyObject* resident_inference_capabilities(PyObject*, PyObject*) {
    OwnedPyObject prefix_cow_abi(Py_BuildValue(
        "{s:i,s:s,s:[s,s,s,s]}",
        "version", 1,
        "operation", "fork_session",
        "profiles",
        "dense-full-cache-kv-final-hidden-v1",
        "dense-cpu-turboquant-mse-qjl-packed-kv-final-hidden-v1",
        "llama-full-cache-gqa-kv-final-hidden-v1",
        "standard-moe-full-cache-gqa-kv-final-hidden-v1"));
    if (!prefix_cow_abi) {
        return nullptr;
    }
    OwnedPyObject result(Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:[s,s],s:O,s:s,s:s}",
        "native_inference", Py_True,
        "resident_inference", Py_True,
        "lossless_kv_cache", Py_True,
        "turboquant_kv_cache", Py_True,
        "turboquant_tile_attention", Py_True,
        "session_prefix_cow", Py_True,
        "session_prefix_cow_cpu_turboquant", Py_True,
        "function_tools", Py_False,
        "structured_output", Py_False,
        "current_logits_exact_prefill", Py_True,
        "whole_model_cuda", Py_True,
        "weight_kernel_profiles",
        "muse-glimmer-bf16-mapped-v1",
        "muse-glimmer-gguf-kquant-mapped-v1",
        "session_prefix_cow_abi", prefix_cow_abi.get(),
        "backend", "cpu-and-native-cuda-resident",
        "cache_policy", "off-full-hybrid-or-native-turboquant"));
    if (!result) return nullptr;
    OwnedPyObject speculative_abi(Py_BuildValue(
        "{s:i,s:s,s:s,s:i,s:i}",
        "version", 1,
        "load_operation", "load_companion",
        "decode_operation", "decode_speculative_block",
        "block_size", 16,
        "proposal_tokens", 15));
    if (!speculative_abi ||
        PyDict_SetItemString(result.get(), "speculative_decoding", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "dflash_cpu", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "dflash_cuda", Py_True) < 0 ||
        PyDict_SetItemString(
            result.get(), "speculative_decoding_abi", speculative_abi.get()) < 0) {
        return nullptr;
    }
    OwnedPyObject media_abi(Py_BuildValue(
        "{s:i,s:s,s:s,s:s,s:i}",
        "version", 1,
        "load_operation", "load_companion",
        "encode_operation", "encode_media",
        "prefill_operation", "prefill_with_embeddings",
        "projection_width", 6656));
    if (!media_abi ||
        PyDict_SetItemString(result.get(), "vision", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "video", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "vision_cpu", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "vision_cuda", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "media_encoder_abi", media_abi.get()) < 0) {
        return nullptr;
    }
    OwnedPyObject adapter_abi(Py_BuildValue(
        "{s:i,s:s,s:s,s:[s,s,s,s,s,s,s,s]}",
        "version", 1,
        "load_operation", "load_companion",
        "format", "neuralfn.native_muse_glimmer_lora.bf16.v1",
        "targets",
        "q_proj", "k_proj", "v_proj", "o_proj", "attn_gate_proj",
        "gate_proj", "up_proj", "down_proj"));
    if (!adapter_abi ||
        PyDict_SetItemString(result.get(), "native_lora", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "native_lora_cpu", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "native_lora_cuda", Py_True) < 0 ||
        PyDict_SetItemString(result.get(), "native_lora_abi", adapter_abi.get()) < 0) {
        return nullptr;
    }
    return result.release();
}

PyObject* fork_session(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* source_capsule = nullptr;
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOO!:fork_session",
            &model_capsule,
            &source_capsule,
            &PyDict_Type,
            &config)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* source = nullptr;
    if (!owned_session(model_capsule, source_capsule, &model, &source)) {
        return nullptr;
    }
    std::int64_t token_count = 0;
    std::int64_t seed = 0;
    if (!mapping_int64(config, "token_count", true, 0, &token_count) ||
        !mapping_int64(config, "seed", true, 0, &seed)) {
        return nullptr;
    }
    try {
        auto handle = std::make_unique<SessionHandle>();
        handle->kind = model->kind;
        handle->model_identity = model->identity;
        if (model->kind == ModelHandle::Kind::Dense) {
            if (!model->dense || !source->dense) {
                throw std::runtime_error(
                    "resident dense prefix COW session storage is unavailable");
            }
            const auto model_stats = model->dense->stats();
            if (model_stats.turboquant_tile_attention_configured) {
                throw std::runtime_error(
                    "resident prefix COW rejects models configured for Tile-CUDA TurboQuant attention");
            }
            handle->dense = source->dense->fork_prefix(token_count, seed);
        } else if (model->kind == ModelHandle::Kind::Llama) {
            if (!model->llama || !source->llama) {
                throw std::runtime_error(
                    "resident LLaMA prefix COW session storage is unavailable");
            }
            handle->llama = source->llama->fork_prefix(token_count, seed);
        } else if (model->kind == ModelHandle::Kind::Glimmer) {
            if (!model->glimmer || !source->glimmer) {
                throw std::runtime_error(
                    "resident Glimmer prefix session storage is unavailable");
            }
            handle->glimmer = source->glimmer->fork_prefix(token_count, seed);
        } else {
            if (!model->moe || !source->moe) {
                throw std::runtime_error(
                    "resident standard-MoE prefix COW session storage is unavailable");
            }
            handle->moe = source->moe->fork_prefix(token_count, seed);
        }
        PyObject* capsule = PyCapsule_New(
            handle.get(), kSessionCapsuleName, destroy_session_capsule);
        if (capsule == nullptr) {
            handle->close();
            return nullptr;
        }
        handle.release();
        return capsule;
    } catch (const std::exception& error) {
        return return_cpp_error(error);
    }
}

PyObject* load_model(PyObject*, PyObject* args) {
    const char* artifact_root = nullptr;
    PyObject* manifest = nullptr;
    if (!PyArg_ParseTuple(args, "sO!:load_model", &artifact_root, &PyDict_Type, &manifest)) {
        return nullptr;
    }
    try {
        auto handle = std::make_unique<ModelHandle>();
        const std::string format = checkpoint_format_from_manifest(manifest);
        if (format == "neuralfn.native_dense_gpt.v5") {
            DenseInferenceConfig inference_config;
            const std::filesystem::path checkpoint = checkpoint_from_manifest(
                artifact_root, manifest, &inference_config);
            std::shared_ptr<DenseModel> resident_model = DenseModel::load(
                checkpoint.string(), inference_config);
            validate_loaded_model_geometry(manifest, *resident_model);
            handle->kind = ModelHandle::Kind::Dense;
            handle->dense = std::move(resident_model);
        } else if (format == "neuralfn.native_family_llama.f32.v1") {
            LlamaInferenceConfig inference_config;
            const std::filesystem::path checkpoint = llama_checkpoint_from_manifest(
                artifact_root, manifest, &inference_config);
            handle->kind = ModelHandle::Kind::Llama;
            handle->llama = LlamaModel::load(checkpoint.string(), inference_config);
        } else if (format == "neuralfn.native_family_muse_glimmer.bf16.v1" ||
                   format == "neuralfn.native_family_muse_glimmer.gguf.kquant.v1") {
            GlimmerInferenceConfig inference_config;
            const std::filesystem::path checkpoint = glimmer_checkpoint_from_manifest(
                artifact_root, manifest, &inference_config);
            handle->kind = ModelHandle::Kind::Glimmer;
            handle->glimmer = GlimmerModel::load(checkpoint.string(), inference_config);
        } else if (format == "neuralfn.native_family_standard_moe.f32.v1") {
            MoeInferenceConfig inference_config;
            const std::filesystem::path checkpoint = standard_moe_checkpoint_from_manifest(
                artifact_root, manifest, &inference_config);
            handle->kind = ModelHandle::Kind::Moe;
            handle->moe = MoeModel::load(checkpoint.string(), inference_config);
        } else {
            throw std::runtime_error(
                "resident inference binding does not implement checkpoint format " + format);
        }
        handle->identity = next_model_identity.fetch_add(1);
        if (handle->identity == 0) {
            throw std::runtime_error("resident inference model identity space is exhausted");
        }
        PyObject* capsule = PyCapsule_New(
            handle.get(), kModelCapsuleName, destroy_model_capsule);
        if (capsule == nullptr) {
            handle->close();
            return nullptr;
        }
        handle.release();
        return capsule;
    } catch (const std::exception& error) {
        if (PyErr_Occurred()) {
            return nullptr;
        }
        return return_cpp_error(error);
    }
}

PyObject* load_model_with_options(PyObject*, PyObject* args) {
    const char* artifact_root = nullptr;
    PyObject* manifest = nullptr;
    PyObject* options = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "sO!O!:load_model_with_options",
            &artifact_root,
            &PyDict_Type,
            &manifest,
            &PyDict_Type,
            &options)) {
        return nullptr;
    }
    try {
        const std::string format = checkpoint_format_from_manifest(manifest);
        if (format != "neuralfn.native_family_muse_glimmer.bf16.v1" &&
            format != "neuralfn.native_family_muse_glimmer.gguf.kquant.v1") {
            throw std::runtime_error(
                "load_model_with_options whole-model CUDA currently accepts only Muse Glimmer");
        }
        std::int64_t cuda_device = -1;
        std::string tile_ops_lib;
        std::string cuda_runtime_lib;
        std::string requested_precision;
        std::string selected_precision;
        if (!mapping_int64(options, "cuda_device", true, -1, &cuda_device) ||
            cuda_device < 0 || cuda_device > std::numeric_limits<int>::max() ||
            !mapping_string(options, "tile_ops_lib", true, &tile_ops_lib) ||
            tile_ops_lib.empty() ||
            !mapping_string(options, "cuda_runtime_lib", false, &cuda_runtime_lib) ||
            !mapping_string(options, "weight_precision", true, &requested_precision) ||
            !mapping_string(manifest, "selected_weight_precision", true, &selected_precision) ||
            requested_precision != selected_precision) {
            throw std::runtime_error(
                "Muse Glimmer CUDA load options/selected precision proof are invalid");
        }
        OwnedPyObject selection_proof(optional_item(options, "selection_proof"));
        if (!selection_proof || !PyMapping_Check(selection_proof.get())) {
            throw std::runtime_error("Muse Glimmer CUDA requires a VRAM selection proof");
        }
        std::string proof_runtime;
        std::string proof_precision;
        if (!mapping_string(selection_proof.get(), "runtime", true, &proof_runtime) ||
            !mapping_string(
                selection_proof.get(), "effective_weight_precision", true, &proof_precision) ||
            proof_runtime != "native-cuda" || proof_precision != selected_precision) {
            throw std::runtime_error("Muse Glimmer CUDA selection proof is inconsistent");
        }

        GlimmerInferenceConfig inference_config;
        const std::filesystem::path checkpoint = glimmer_checkpoint_from_manifest(
            artifact_root, manifest, &inference_config);
        inference_config.whole_model_cuda = true;
        inference_config.cuda_device = static_cast<int>(cuda_device);
        inference_config.tile_ops_lib = tile_ops_lib;
        inference_config.cuda_runtime_lib = cuda_runtime_lib;
        auto handle = std::make_unique<ModelHandle>();
        handle->kind = ModelHandle::Kind::Glimmer;
        handle->glimmer = GlimmerModel::load(checkpoint.string(), inference_config);
        handle->identity = next_model_identity.fetch_add(1);
        if (handle->identity == 0) {
            throw std::runtime_error("resident inference model identity space is exhausted");
        }
        PyObject* capsule = PyCapsule_New(
            handle.get(), kModelCapsuleName, destroy_model_capsule);
        if (capsule == nullptr) {
            handle->close();
            return nullptr;
        }
        handle.release();
        return capsule;
    } catch (const std::exception& error) {
        if (PyErr_Occurred()) return nullptr;
        return return_cpp_error(error);
    }
}

PyObject* load_companion(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    const char* artifact_root = nullptr;
    PyObject* descriptor = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OsO!:load_companion",
            &model_capsule,
            &artifact_root,
            &PyDict_Type,
            &descriptor)) {
        return nullptr;
    }
    ModelHandle* handle = model_handle(model_capsule);
    if (handle == nullptr) return nullptr;
    try {
        if (handle->kind != ModelHandle::Kind::Glimmer || !handle->glimmer) {
            throw std::runtime_error("companion loading requires a Muse Glimmer target");
        }
        if (handle->glimmer->stats().open_sessions != 0) {
            throw std::runtime_error("companions must be loaded before creating target sessions");
        }
        std::string format;
        if (!mapping_string(descriptor, "format", true, &format)) {
            throw std::runtime_error("Muse Glimmer companion format is missing");
        }
        if (format == "neuralfn.native_muse_glimmer_lora.bf16.v1") {
            if (handle->glimmer_assistant) {
                throw std::runtime_error(
                    "Muse Glimmer LoRA cannot be combined with the stock DFlash assistant");
            }
            std::int64_t rank = 0;
            std::int64_t adapter_nbytes = 0;
            double alpha = 0.0;
            std::uint32_t target_mask = 0;
            const std::filesystem::path checkpoint = glimmer_lora_checkpoint_from_descriptor(
                artifact_root, descriptor, *handle->glimmer, &rank, &alpha, &target_mask);
            std::string adapter_digest;
            if (!mapping_string(descriptor, "target_sha256", true, &adapter_digest) ||
                !mapping_int64(descriptor, "target_nbytes", true, 0, &adapter_nbytes)) {
                throw std::runtime_error("Muse Glimmer LoRA digest/extent is missing");
            }
            handle->glimmer->load_lora_adapter(
                checkpoint.string(), adapter_digest, rank, alpha, target_mask);
            return Py_BuildValue(
                "{s:O,s:s,s:i,s:L,s:L,s:O}",
                "loaded", Py_True,
                "component", "lora",
                "feature_abi_version", 1,
                "rank", static_cast<long long>(rank),
                "resident_weight_bytes", static_cast<long long>(adapter_nbytes),
                "whole_model_cuda", handle->glimmer->whole_model_cuda() ? Py_True : Py_False);
        }
        if (format == "neuralfn.native_family_muse_glimmer_mmproj.gguf.kquant.v1") {
            if (handle->glimmer->has_vision()) {
                throw std::runtime_error("Muse Glimmer vision weights are already loaded");
            }
            const std::filesystem::path checkpoint = glimmer_vision_checkpoint_from_descriptor(
                artifact_root, descriptor, *handle->glimmer);
            handle->glimmer->load_vision_companion(checkpoint.string());
            return Py_BuildValue(
                "{s:O,s:s,s:i,s:i,s:i,s:L,s:O}",
                "loaded", Py_True,
                "component", "mmproj",
                "feature_abi_version", 1,
                "packed_patch_width", 588,
                "output_width", 6656,
                "resident_weight_bytes",
                static_cast<long long>(handle->glimmer->vision_weight_bytes()),
                "whole_model_cuda",
                handle->glimmer->vision_whole_model_cuda() ? Py_True : Py_False);
        }
        if (handle->glimmer_assistant) {
            throw std::runtime_error("DFlash companion is already loaded for this target");
        }
        if (handle->glimmer->has_lora_adapter()) {
            throw std::runtime_error(
                "stock DFlash is disabled for a Muse Glimmer target with a native LoRA adapter");
        }
        GlimmerAssistantConfig config;
        const std::filesystem::path checkpoint = glimmer_assistant_checkpoint_from_descriptor(
            artifact_root, descriptor, *handle->glimmer, &config);
        handle->glimmer_assistant = GlimmerAssistantModel::load(
            checkpoint.string(), std::move(config), handle->glimmer);
        return Py_BuildValue(
            "{s:O,s:s,s:i,s:i,s:L,s:L,s:O}",
            "loaded", Py_True,
            "component", "dflash",
            "feature_abi_version", 1,
            "block_size", 16,
            "proposal_tokens", static_cast<long long>(15),
            "resident_weight_bytes",
            static_cast<long long>(handle->glimmer_assistant->weight_bytes()),
            "whole_model_cuda",
            handle->glimmer_assistant->whole_model_cuda() ? Py_True : Py_False);
    } catch (const std::exception& error) {
        if (PyErr_Occurred()) return nullptr;
        return return_cpp_error(error);
    }
}

PyObject* close_model(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O:close_model", &model_capsule)) {
        return nullptr;
    }
    ModelHandle* handle = model_handle(model_capsule, false);
    if (handle == nullptr) {
        return nullptr;
    }
    handle->close();
    Py_RETURN_NONE;
}

PyObject* configure_model_turboquant_attention(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OO!:configure_model_turboquant_attention",
            &model_capsule,
            &PyDict_Type,
            &config)) {
        return nullptr;
    }
    ModelHandle* model = model_handle(model_capsule);
    if (model == nullptr) {
        return nullptr;
    }
    if (model->kind != ModelHandle::Kind::Dense) {
        PyErr_SetString(
            PyExc_ValueError,
            "Tile-CUDA TurboQuant attention is implemented only for resident dense models");
        return nullptr;
    }
    TileTurboQuantConfig parsed;
    if (!mapping_string(config, "backend", true, &parsed.backend) ||
        !mapping_string(config, "tile_ops_lib", true, &parsed.tile_ops_lib) ||
        !mapping_optional_string_or_none(
            config, "cuda_runtime_lib", &parsed.cuda_runtime_lib) ||
        !mapping_int64(config, "device", false, 0, &parsed.device)) {
        return nullptr;
    }
    if (parsed.backend != "tile-cuda") {
        PyErr_SetString(
            PyExc_ValueError,
            "resident TurboQuant attention configuration backend must be 'tile-cuda'");
        return nullptr;
    }
    try {
        const TileTurboQuantModelStats stats =
            model->dense->configure_turboquant_attention(std::move(parsed));
        return Py_BuildValue(
            "{s:O,s:s,s:s,s:s,s:L}",
            "configured", stats.configured ? Py_True : Py_False,
            "backend", stats.backend.c_str(),
            "tile_ops_lib", stats.tile_ops_lib.c_str(),
            "cuda_runtime_lib", stats.cuda_runtime_lib.c_str(),
            "device", static_cast<long long>(stats.device));
    } catch (const std::exception& error) {
        return return_cpp_error(error);
    }
}

PyObject* create_session(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* config = nullptr;
    if (!PyArg_ParseTuple(args, "OO!:create_session", &model_capsule, &PyDict_Type, &config)) {
        return nullptr;
    }
    ModelHandle* model = model_handle(model_capsule);
    if (model == nullptr) {
        return nullptr;
    }
    std::int64_t seed = 0;
    if (!mapping_int64(config, "seed", true, 0, &seed)) {
        return nullptr;
    }
    PyObject* cache = optional_item(config, "kv_cache");
    if (cache == nullptr || !PyMapping_Check(cache)) {
        Py_XDECREF(cache);
        PyErr_SetString(PyExc_ValueError, "resident session config must contain a kv_cache object");
        return nullptr;
    }
    std::string effective_mode;
    std::string turboquant_attention_backend = "cpu";
    const bool parsed =
        mapping_string(cache, "effective_mode", true, &effective_mode) &&
        mapping_string(
            cache,
            "turboquant_attention_backend",
            false,
            &turboquant_attention_backend);
    if (!parsed) {
        Py_DECREF(cache);
        return nullptr;
    }
    KVCacheMode cache_mode = KVCacheMode::Off;
    std::optional<TurboQuantTables> parsed_turboquant;
    bool tile_turboquant_attention = false;
    if (effective_mode == "full") {
        cache_mode = KVCacheMode::Full;
    } else if (effective_mode == "turboquant") {
        if (model->kind != ModelHandle::Kind::Dense) {
            Py_DECREF(cache);
            PyErr_SetString(
                PyExc_ValueError,
                model->kind == ModelHandle::Kind::Llama
                    ? "canonical LLaMA resident inference has not proved TurboQuant GQA storage"
                    : (model->kind == ModelHandle::Kind::Glimmer
                        ? "Muse Glimmer requires its exact hybrid lossless KV cache"
                        : "canonical standard-MoE resident inference has not proved TurboQuant GQA storage"));
            return nullptr;
        }
        cache_mode = KVCacheMode::TurboQuant;
        TurboQuantTables tables;
        if (!turboquant_tables(cache, &tables)) {
            Py_DECREF(cache);
            return nullptr;
        }
        parsed_turboquant = std::move(tables);
        if (turboquant_attention_backend == "tile-cuda") {
            tile_turboquant_attention = true;
        } else if (turboquant_attention_backend != "cpu") {
            Py_DECREF(cache);
            PyErr_SetString(
                PyExc_ValueError,
                "turboquant_attention_backend must be 'cpu' or 'tile-cuda'");
            return nullptr;
        }
    } else if (effective_mode != "off") {
        Py_DECREF(cache);
        PyErr_SetString(
            PyExc_ValueError,
            "resident inference supports kv_cache effective_mode 'off', 'full', or 'turboquant'");
        return nullptr;
    }
    if (effective_mode != "turboquant" && turboquant_attention_backend != "cpu") {
        Py_DECREF(cache);
        PyErr_SetString(
            PyExc_ValueError,
            "Tile-CUDA TurboQuant attention requires kv_cache effective_mode 'turboquant'");
        return nullptr;
    }
    Py_DECREF(cache);
    try {
        auto handle = std::make_unique<SessionHandle>();
        handle->kind = model->kind;
        handle->model_identity = model->identity;
        if (model->kind == ModelHandle::Kind::Dense) {
            handle->dense = model->dense->create_session(
                seed,
                cache_mode,
                std::move(parsed_turboquant),
                tile_turboquant_attention);
        } else if (model->kind == ModelHandle::Kind::Llama) {
            if (parsed_turboquant.has_value()) {
                throw std::runtime_error(
                    "canonical LLaMA resident inference does not accept TurboQuant tables");
            }
            handle->llama = model->llama->create_session(seed, cache_mode);
        } else if (model->kind == ModelHandle::Kind::Glimmer) {
            if (parsed_turboquant.has_value()) {
                throw std::runtime_error(
                    "Muse Glimmer resident inference does not accept TurboQuant tables");
            }
            handle->glimmer = model->glimmer_assistant
                ? model->glimmer->create_speculative_session(
                      seed, cache_mode, model->glimmer_assistant)
                : model->glimmer->create_session(seed, cache_mode);
        } else {
            if (parsed_turboquant.has_value()) {
                throw std::runtime_error(
                    "canonical standard-MoE resident inference does not accept TurboQuant tables");
            }
            handle->moe = model->moe->create_session(seed, cache_mode);
        }
        PyObject* capsule = PyCapsule_New(
            handle.get(), kSessionCapsuleName, destroy_session_capsule);
        if (capsule == nullptr) {
            handle->close();
            return nullptr;
        }
        handle.release();
        return capsule;
    } catch (const std::exception& error) {
        return return_cpp_error(error);
    }
}

PyObject* close_session(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO:close_session", &model_capsule, &session_capsule)) {
        return nullptr;
    }
    ModelHandle* model = model_handle(model_capsule, false);
    SessionHandle* session = session_handle(session_capsule, false);
    if (model == nullptr || session == nullptr) {
        return nullptr;
    }
    if (session->has_value()) {
        const bool owned = model->identity != 0 &&
            model->identity == session->model_identity &&
            model->kind == session->kind;
        if (!owned) {
            PyErr_SetString(PyExc_ValueError, "resident inference session does not belong to this model");
            return nullptr;
        }
        session->close();
    }
    Py_RETURN_NONE;
}

PyObject* prefill(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    PyObject* token_ids = nullptr;
    long long start_position = 0;
    if (!PyArg_ParseTuple(args, "OOOL:prefill", &model_capsule, &session_capsule, &token_ids, &start_position)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    std::vector<std::int64_t> tokens;
    if (!token_vector(token_ids, "token_ids", &tokens)) {
        return nullptr;
    }
    std::string error_message;
    bool cancelled = false;
    const std::shared_ptr<DenseSession> dense_session = session->dense;
    const std::shared_ptr<LlamaSession> llama_session = session->llama;
    const std::shared_ptr<GlimmerSession> glimmer_session = session->glimmer;
    const std::shared_ptr<MoeSession> moe_session = session->moe;
    const ModelHandle::Kind kind = session->kind;
    Py_BEGIN_ALLOW_THREADS
    try {
        if (kind == ModelHandle::Kind::Dense) {
            dense_session->prefill(tokens, static_cast<std::int64_t>(start_position));
        } else if (kind == ModelHandle::Kind::Llama) {
            llama_session->prefill(tokens, static_cast<std::int64_t>(start_position));
        } else if (kind == ModelHandle::Kind::Glimmer) {
            glimmer_session->prefill(tokens, static_cast<std::int64_t>(start_position));
        } else {
            moe_session->prefill(tokens, static_cast<std::int64_t>(start_position));
        }
    } catch (const ResidentCancellationError&) {
        cancelled = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }
    Py_END_ALLOW_THREADS
    if (cancelled) {
        PyErr_SetString(PyExc_InterruptedError, "resident inference prefill was cancelled");
        return nullptr;
    }
    if (!error_message.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* prefill_with_embeddings(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    PyObject* token_ids = nullptr;
    PyObject* replacement_positions = nullptr;
    PyObject* replacement_embeddings = nullptr;
    long long start_position = 0;
    if (!PyArg_ParseTuple(
            args, "OOOLOO:prefill_with_embeddings", &model_capsule,
            &session_capsule, &token_ids, &start_position,
            &replacement_positions, &replacement_embeddings)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) return nullptr;
    if (model->kind != ModelHandle::Kind::Glimmer || !session->glimmer) {
        PyErr_SetString(
            PyExc_ValueError,
            "prefill_with_embeddings is supported only by resident Muse Glimmer");
        return nullptr;
    }
    std::vector<std::int64_t> tokens;
    std::vector<std::int64_t> positions;
    std::vector<float> embeddings;
    if (!token_vector(token_ids, "token_ids", &tokens) ||
        !token_vector(replacement_positions, "replacement_positions", &positions) ||
        !float_vector(replacement_embeddings, "replacement_embeddings", &embeddings)) {
        return nullptr;
    }
    std::string error_message;
    bool cancelled = false;
    const std::shared_ptr<GlimmerSession> glimmer_session = session->glimmer;
    Py_BEGIN_ALLOW_THREADS
    try {
        glimmer_session->prefill_with_embeddings(
            tokens, static_cast<std::int64_t>(start_position), positions, embeddings);
    } catch (const ResidentCancellationError&) {
        cancelled = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }
    Py_END_ALLOW_THREADS
    if (cancelled) {
        PyErr_SetString(PyExc_InterruptedError, "resident multimodal prefill was cancelled");
        return nullptr;
    }
    if (!error_message.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* encode_media(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* packed_patches_object = nullptr;
    PyObject* grid_thw_object = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOO:encode_media", &model_capsule,
            &packed_patches_object, &grid_thw_object)) {
        return nullptr;
    }
    ModelHandle* model = model_handle(model_capsule);
    if (model == nullptr) {
        return nullptr;
    }
    if (model->kind != ModelHandle::Kind::Glimmer || !model->glimmer) {
        PyErr_SetString(
            PyExc_ValueError,
            "encode_media is supported only by resident Muse Glimmer");
        return nullptr;
    }
    std::vector<float> packed_patches;
    std::vector<std::int64_t> grid_thw;
    if (!float_vector(packed_patches_object, "packed_patches", &packed_patches) ||
        !token_vector(grid_thw_object, "grid_thw", &grid_thw)) {
        return nullptr;
    }
    std::vector<float> encoded;
    std::string error_message;
    bool cancelled = false;
    const std::shared_ptr<GlimmerModel> glimmer_model = model->glimmer;
    std::atomic<bool> cancellation{false};
    Py_BEGIN_ALLOW_THREADS
    try {
        encoded = glimmer_model->encode_media(packed_patches, grid_thw, cancellation);
    } catch (const ResidentCancellationError&) {
        cancelled = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }
    Py_END_ALLOW_THREADS
    if (cancelled) {
        PyErr_SetString(PyExc_InterruptedError, "resident media encoding was cancelled");
        return nullptr;
    }
    if (!error_message.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    const std::int64_t width = glimmer_model->vision_output_size();
    if (width <= 0 || encoded.size() % static_cast<std::size_t>(width) != 0 ||
        encoded.size() > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
        PyErr_SetString(PyExc_RuntimeError, "resident media encoder returned malformed rows");
        return nullptr;
    }
    OwnedPyObject values(PyList_New(static_cast<Py_ssize_t>(encoded.size())));
    if (!values) {
        return nullptr;
    }
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        PyObject* value = PyFloat_FromDouble(encoded[index]);
        if (value == nullptr) {
            return nullptr;
        }
        PyList_SET_ITEM(values.get(), static_cast<Py_ssize_t>(index), value);
    }
    return Py_BuildValue(
        "{s:L,s:L,s:O}",
        "rows", static_cast<long long>(encoded.size() / static_cast<std::size_t>(width)),
        "width", static_cast<long long>(width),
        "values", values.get());
}

PyObject* current_logits(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO:current_logits", &model_capsule, &session_capsule)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    std::vector<float> logits;
    std::string error_message;
    bool cancelled = false;
    const std::shared_ptr<DenseSession> dense_session = session->dense;
    const std::shared_ptr<LlamaSession> llama_session = session->llama;
    const std::shared_ptr<GlimmerSession> glimmer_session = session->glimmer;
    const std::shared_ptr<MoeSession> moe_session = session->moe;
    const ModelHandle::Kind kind = session->kind;
    Py_BEGIN_ALLOW_THREADS
    try {
        if (kind == ModelHandle::Kind::Dense) {
            logits = dense_session->current_logits();
        } else if (kind == ModelHandle::Kind::Llama) {
            logits = llama_session->current_logits();
        } else if (kind == ModelHandle::Kind::Glimmer) {
            logits = glimmer_session->current_logits();
        } else {
            logits = moe_session->current_logits();
        }
    } catch (const ResidentCancellationError&) {
        cancelled = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }
    Py_END_ALLOW_THREADS
    if (cancelled) {
        PyErr_SetString(PyExc_InterruptedError, "resident inference logits request was cancelled");
        return nullptr;
    }
    if (!error_message.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(static_cast<Py_ssize_t>(logits.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (std::size_t index = 0; index < logits.size(); ++index) {
        PyObject* value = PyFloat_FromDouble(logits[index]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(index), value);
    }
    return result;
}

bool generation_config(PyObject* mapping, GenerationConfig* output) {
    if (!mapping_double(mapping, "temperature", true, 0.8, &output->temperature) ||
        !mapping_double(mapping, "top_p", false, 1.0, &output->top_p)) {
        return false;
    }
    PyObject* top_k = optional_item(mapping, "top_k");
    if (top_k == nullptr) {
        if (PyErr_Occurred()) {
            return false;
        }
        output->top_k = 0;
    } else if (top_k == Py_None) {
        Py_DECREF(top_k);
        output->top_k = 0;
    } else {
        const bool parsed = py_int64(top_k, "top_k", &output->top_k);
        Py_DECREF(top_k);
        if (!parsed) {
            return false;
        }
    }
    PyObject* seed = optional_item(mapping, "seed");
    if (seed != nullptr && seed != Py_None) {
        std::int64_t parsed_seed = 0;
        const bool parsed = py_int64(seed, "seed", &parsed_seed);
        Py_DECREF(seed);
        if (!parsed) {
            return false;
        }
        output->seed = parsed_seed;
    } else {
        Py_XDECREF(seed);
    }
    PyObject* stops = optional_item(mapping, "stop_token_ids");
    if (stops != nullptr) {
        const bool parsed = token_vector(stops, "stop_token_ids", &output->stop_token_ids);
        Py_DECREF(stops);
        if (!parsed) {
            return false;
        }
    } else if (PyErr_Occurred()) {
        return false;
    }
    PyObject* strict = optional_item(mapping, "strict_model_compute");
    if (strict != nullptr) {
        if (!PyBool_Check(strict)) {
            Py_DECREF(strict);
            PyErr_SetString(PyExc_TypeError, "strict_model_compute must be a boolean");
            return false;
        }
        const bool declared = strict == Py_True;
        Py_DECREF(strict);
        if (declared != (output->temperature == 0.0)) {
            PyErr_SetString(PyExc_ValueError, "strict_model_compute does not match exact-zero temperature");
            return false;
        }
    } else if (PyErr_Occurred()) {
        return false;
    }
    return true;
}

PyObject* decode_one(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    PyObject* config_mapping = nullptr;
    if (!PyArg_ParseTuple(args, "OOO!:decode_one", &model_capsule, &session_capsule, &PyDict_Type, &config_mapping)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    GenerationConfig config;
    if (!generation_config(config_mapping, &config)) {
        return nullptr;
    }
    DecodeResult result;
    std::string error_message;
    bool cancelled = false;
    const std::shared_ptr<DenseSession> dense_session = session->dense;
    const std::shared_ptr<LlamaSession> llama_session = session->llama;
    const std::shared_ptr<GlimmerSession> glimmer_session = session->glimmer;
    const std::shared_ptr<MoeSession> moe_session = session->moe;
    const ModelHandle::Kind kind = session->kind;
    Py_BEGIN_ALLOW_THREADS
    try {
        if (kind == ModelHandle::Kind::Dense) {
            result = dense_session->decode_one(config);
        } else if (kind == ModelHandle::Kind::Llama) {
            result = llama_session->decode_one(config);
        } else if (kind == ModelHandle::Kind::Glimmer) {
            result = glimmer_session->decode_one(config);
        } else {
            result = moe_session->decode_one(config);
        }
    } catch (const ResidentCancellationError&) {
        cancelled = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }
    Py_END_ALLOW_THREADS
    if (cancelled) {
        PyErr_SetString(PyExc_InterruptedError, "resident inference decode was cancelled");
        return nullptr;
    }
    if (!error_message.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    PyObject* finish_reason = result.finish_reason.empty()
        ? Py_NewRef(Py_None)
        : PyUnicode_FromString(result.finish_reason.c_str());
    if (finish_reason == nullptr) {
        return nullptr;
    }
    PyObject* payload = Py_BuildValue(
        "{s:L,s:s,s:O,s:f}",
        "token_id", static_cast<long long>(result.token_id),
        "text", "",
        "finish_reason", finish_reason,
        "selected_logit", result.selected_logit);
    Py_DECREF(finish_reason);
    return payload;
}

PyObject* decode_speculative_block(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    PyObject* config_mapping = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "OOO!:decode_speculative_block",
            &model_capsule,
            &session_capsule,
            &PyDict_Type,
            &config_mapping)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) return nullptr;
    if (model->kind != ModelHandle::Kind::Glimmer || !model->glimmer_assistant ||
        !session->glimmer) {
        PyErr_SetString(
            PyExc_ValueError,
            "decode_speculative_block requires a Glimmer model/session with DFlash loaded");
        return nullptr;
    }
    GenerationConfig config;
    std::int64_t max_tokens_remaining = 0;
    if (!generation_config(config_mapping, &config) ||
        !mapping_int64(
            config_mapping,
            "max_tokens_remaining",
            true,
            0,
            &max_tokens_remaining)) {
        return nullptr;
    }
    neuralfn::resident_glimmer::SpeculativeStepResult result;
    std::string error_message;
    bool cancelled = false;
    const std::shared_ptr<GlimmerSession> glimmer_session = session->glimmer;
    Py_BEGIN_ALLOW_THREADS
    try {
        result = glimmer_session->decode_speculative_block(config, max_tokens_remaining);
    } catch (const ResidentCancellationError&) {
        cancelled = true;
    } catch (const std::exception& error) {
        error_message = error.what();
    }
    Py_END_ALLOW_THREADS
    if (cancelled) {
        PyErr_SetString(PyExc_InterruptedError, "resident DFlash block was cancelled");
        return nullptr;
    }
    if (!error_message.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    OwnedPyObject tokens(PyList_New(static_cast<Py_ssize_t>(result.tokens.size())));
    if (!tokens) return nullptr;
    for (std::size_t index = 0; index < result.tokens.size(); ++index) {
        const DecodeResult& decoded = result.tokens[index];
        OwnedPyObject finish(decoded.finish_reason.empty()
            ? Py_NewRef(Py_None)
            : PyUnicode_FromString(decoded.finish_reason.c_str()));
        OwnedPyObject row(finish ? Py_BuildValue(
            "{s:L,s:s,s:O,s:f}",
            "token_id", static_cast<long long>(decoded.token_id),
            "text", "",
            "finish_reason", finish.get(),
            "selected_logit", decoded.selected_logit) : nullptr);
        if (!row) return nullptr;
        PyList_SET_ITEM(tokens.get(), static_cast<Py_ssize_t>(index), row.release());
    }
    return Py_BuildValue(
        "{s:O,s:L,s:L,s:L,s:L,s:L,s:O}",
        "tokens", tokens.get(),
        "proposed_tokens", static_cast<long long>(result.proposed_tokens),
        "accepted_tokens", static_cast<long long>(result.accepted_tokens),
        "rejected_tokens", static_cast<long long>(result.rejected_tokens),
        "target_rows", static_cast<long long>(result.target_rows),
        "assistant_blocks", static_cast<long long>(result.assistant_blocks),
        "target_only_warmup", result.target_only_warmup ? Py_True : Py_False);
}

PyObject* truncate_session(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    long long token_count = 0;
    if (!PyArg_ParseTuple(args, "OOL:truncate_session", &model_capsule, &session_capsule, &token_count)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    try {
        if (session->kind == ModelHandle::Kind::Dense) {
            session->dense->truncate(static_cast<std::int64_t>(token_count));
        } else if (session->kind == ModelHandle::Kind::Llama) {
            session->llama->truncate(static_cast<std::int64_t>(token_count));
        } else if (session->kind == ModelHandle::Kind::Glimmer) {
            session->glimmer->truncate(static_cast<std::int64_t>(token_count));
        } else {
            session->moe->truncate(static_cast<std::int64_t>(token_count));
        }
        Py_RETURN_NONE;
    } catch (const std::exception& error) {
        return return_cpp_error(error);
    }
}

PyObject* reset_session(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO:reset_session", &model_capsule, &session_capsule)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    try {
        if (session->kind == ModelHandle::Kind::Dense) {
            session->dense->reset();
        } else if (session->kind == ModelHandle::Kind::Llama) {
            session->llama->reset();
        } else if (session->kind == ModelHandle::Kind::Glimmer) {
            session->glimmer->reset();
        } else {
            session->moe->reset();
        }
        Py_RETURN_NONE;
    } catch (const std::exception& error) {
        return return_cpp_error(error);
    }
}

PyObject* cancel_session(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO:cancel_session", &model_capsule, &session_capsule)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    if (session->kind == ModelHandle::Kind::Dense) {
        session->dense->cancel();
    } else if (session->kind == ModelHandle::Kind::Llama) {
        session->llama->cancel();
    } else if (session->kind == ModelHandle::Kind::Glimmer) {
        session->glimmer->cancel();
    } else {
        session->moe->cancel();
    }
    Py_RETURN_NONE;
}

PyObject* model_stats(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O:model_stats", &model_capsule)) {
        return nullptr;
    }
    ModelHandle* handle = model_handle(model_capsule);
    if (handle == nullptr) {
        return nullptr;
    }
    const auto stats = handle->kind == ModelHandle::Kind::Dense
        ? handle->dense->stats()
        : (handle->kind == ModelHandle::Kind::Llama
            ? handle->llama->stats()
            : (handle->kind == ModelHandle::Kind::Glimmer
                ? handle->glimmer->stats()
                : handle->moe->stats()));
    const bool turboquant_supported =
        handle->kind == ModelHandle::Kind::Dense &&
        stats.num_heads > 0 && stats.channels > 0 &&
        stats.channels % stats.num_heads == 0 &&
        stats.channels / stats.num_heads >= 2 &&
        stats.channels / stats.num_heads % 2 == 0;
    const bool whole_model_cuda =
        handle->kind == ModelHandle::Kind::Glimmer && handle->glimmer->whole_model_cuda();
    PyObject* result = Py_BuildValue(
        "{s:s,s:s,s:O,s:O,s:O,s:O,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L}",
        "backend", whole_model_cuda ? "native-cuda-resident" : "cpu-reference-resident",
        "checkpoint_path", stats.checkpoint_path.c_str(),
        "resident_inference", Py_True,
        "lossless_kv_cache", Py_True,
        "turboquant_kv_cache", turboquant_supported ? Py_True : Py_False,
        "recompute_full_prefix", Py_False,
        "max_seq_len", static_cast<long long>(stats.max_seq_len),
        "vocab_size", static_cast<long long>(stats.vocab_size),
        "padded_vocab_size", static_cast<long long>(stats.padded_vocab_size),
        "num_layers", static_cast<long long>(stats.num_layers),
        "num_heads", static_cast<long long>(stats.num_heads),
        "channels", static_cast<long long>(stats.channels),
        "parameter_count", static_cast<long long>(stats.parameter_count),
        "resident_weight_bytes", static_cast<long long>(stats.weight_bytes),
        "weights_load_count", static_cast<long long>(stats.weights_load_count),
        "open_sessions", static_cast<long long>(stats.open_sessions),
        "forward_calls", static_cast<long long>(stats.forward_calls),
        "turboquant_table_load_count", static_cast<long long>(stats.turboquant_table_load_count),
        "subprocess_spawns", static_cast<long long>(0));
    if (result == nullptr) {
        return nullptr;
    }
    const char* family_name = handle->kind == ModelHandle::Kind::Dense
        ? "dense-gpt"
        : (handle->kind == ModelHandle::Kind::Llama
            ? "llama"
            : (handle->kind == ModelHandle::Kind::Glimmer ? "muse_glimmer" : "mixllama"));
    OwnedPyObject model_family(PyUnicode_FromString(family_name));
    const std::int64_t head_dim = handle->kind == ModelHandle::Kind::Glimmer
        ? handle->glimmer->head_dim()
        : (stats.num_heads > 0 ? stats.channels / stats.num_heads : 0);
    OwnedPyObject num_kv_heads(PyLong_FromLongLong(static_cast<long long>(
        handle->kind == ModelHandle::Kind::Dense
            ? stats.num_heads
            : (handle->kind == ModelHandle::Kind::Llama
                ? handle->llama->num_kv_heads()
                : (handle->kind == ModelHandle::Kind::Glimmer
                    ? handle->glimmer->num_kv_heads()
                    : handle->moe->num_kv_heads())))));
    OwnedPyObject head_dimension(PyLong_FromLongLong(static_cast<long long>(head_dim)));
    OwnedPyObject qk_norm_eps(PyFloat_FromDouble(stats.qk_norm_eps));
    OwnedPyObject logit_softcap(PyFloat_FromDouble(stats.logit_softcap));
    const bool tile_configured =
        handle->kind == ModelHandle::Kind::Dense &&
        stats.turboquant_tile_attention_configured;
    OwnedPyObject tile_backend(PyUnicode_FromString(
        tile_configured ? stats.turboquant_attention_backend.c_str() : "cpu"));
    OwnedPyObject tile_ops_lib(
        tile_configured
            ? PyUnicode_FromString(stats.turboquant_tile_ops_lib.c_str())
            : nullptr);
    OwnedPyObject cuda_runtime_lib(
        tile_configured
            ? PyUnicode_FromString(stats.turboquant_cuda_runtime_lib.c_str())
            : nullptr);
    OwnedPyObject cuda_device(
        tile_configured
            ? PyLong_FromLongLong(static_cast<long long>(stats.turboquant_cuda_device))
            : nullptr);
    if (!model_family || !num_kv_heads || !head_dimension ||
        !qk_norm_eps || !logit_softcap || !tile_backend ||
        (tile_configured && (!tile_ops_lib || !cuda_runtime_lib || !cuda_device)) ||
        PyDict_SetItemString(result, "model_family", model_family.get()) < 0 ||
        PyDict_SetItemString(result, "num_kv_heads", num_kv_heads.get()) < 0 ||
        PyDict_SetItemString(result, "head_dim", head_dimension.get()) < 0 ||
        PyDict_SetItemString(
            result, "use_qk_norm", stats.use_qk_norm ? Py_True : Py_False) < 0 ||
        PyDict_SetItemString(result, "qk_norm_eps", qk_norm_eps.get()) < 0 ||
        PyDict_SetItemString(result, "logit_softcap", logit_softcap.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_tile_attention_configured",
            tile_configured ? Py_True : Py_False) < 0 ||
        PyDict_SetItemString(
            result, "turboquant_attention_backend", tile_backend.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_tile_ops_lib",
            tile_configured ? tile_ops_lib.get() : Py_None) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_cuda_runtime_lib",
            tile_configured ? cuda_runtime_lib.get() : Py_None) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_cuda_device",
            tile_configured ? cuda_device.get() : Py_None) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    if (handle->kind != ModelHandle::Kind::Dense) {
        const std::int64_t hidden = handle->kind == ModelHandle::Kind::Llama
            ? handle->llama->hidden_dim()
            : (handle->kind == ModelHandle::Kind::Glimmer
                ? handle->glimmer->intermediate_dim()
                : handle->moe->hidden_dim());
        const double theta = handle->kind == ModelHandle::Kind::Llama
            ? handle->llama->rope_theta()
            : (handle->kind == ModelHandle::Kind::Glimmer
                ? handle->glimmer->rope_theta()
                : handle->moe->rope_theta());
        const double epsilon = handle->kind == ModelHandle::Kind::Llama
            ? handle->llama->rms_norm_eps()
            : (handle->kind == ModelHandle::Kind::Glimmer
                ? handle->glimmer->norm_eps()
                : handle->moe->rms_norm_eps());
        OwnedPyObject hidden_dim(PyLong_FromLongLong(static_cast<long long>(
            hidden)));
        OwnedPyObject rope_theta(PyFloat_FromDouble(theta));
        OwnedPyObject rms_norm_eps(PyFloat_FromDouble(epsilon));
        if (!hidden_dim || !rope_theta || !rms_norm_eps ||
            PyDict_SetItemString(result, "hidden_dim", hidden_dim.get()) < 0 ||
            PyDict_SetItemString(result, "rope_theta", rope_theta.get()) < 0 ||
            PyDict_SetItemString(result, "rms_norm_eps", rms_norm_eps.get()) < 0) {
            Py_DECREF(result);
            return nullptr;
        }
    } else {
        OwnedPyObject activation_mode(PyUnicode_FromString(
            stats.moa_mode ? "moa" : "single"));
        OwnedPyObject mlp_activation(PyUnicode_FromString(stats.mlp_activation.c_str()));
        OwnedPyObject moa_interval(PyLong_FromLongLong(static_cast<long long>(
            stats.moa_interval)));
        if (!activation_mode || !mlp_activation || !moa_interval ||
            PyDict_SetItemString(result, "activation_mode", activation_mode.get()) < 0 ||
            PyDict_SetItemString(result, "mlp_activation", mlp_activation.get()) < 0 ||
            PyDict_SetItemString(result, "moa_interval", moa_interval.get()) < 0) {
            Py_DECREF(result);
            return nullptr;
        }
    }
    if (handle->kind == ModelHandle::Kind::Glimmer) {
        const bool dflash_cuda = handle->glimmer_assistant &&
            handle->glimmer_assistant->whole_model_cuda();
        OwnedPyObject cuda_weight_bytes(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_resident_weight_bytes())));
        OwnedPyObject cuda_workspace_bytes(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_workspace_bytes())));
        OwnedPyObject cuda_launches(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_kernel_launches())));
        OwnedPyObject cuda_k_quant_mmq_linears(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_k_quant_mmq_linears())));
        OwnedPyObject cuda_q8_quantizations(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_q8_activation_quantizations())));
        OwnedPyObject cuda_q8_linears(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_q8_packed_linears())));
        OwnedPyObject cuda_argmax_calls(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_device_argmax_calls())));
        OwnedPyObject cuda_argmax_rows(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->cuda_device_argmax_rows())));
        OwnedPyObject cuda_device_value(
            whole_model_cuda
                ? PyLong_FromLong(handle->glimmer->cuda_device())
                : Py_NewRef(Py_None));
        OwnedPyObject cuda_tile_library(
            whole_model_cuda
                ? PyUnicode_FromString(handle->glimmer->cuda_tile_ops_library().c_str())
                : Py_NewRef(Py_None));
        OwnedPyObject cuda_runtime_library(
            whole_model_cuda
                ? PyUnicode_FromString(handle->glimmer->cuda_runtime_library().c_str())
                : Py_NewRef(Py_None));
        OwnedPyObject cpu_compute_rows(PyLong_FromLongLong(static_cast<long long>(
            whole_model_cuda ? 0 : stats.forward_calls)));
        OwnedPyObject assistant_weight_bytes(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer_assistant ? handle->glimmer_assistant->weight_bytes() : 0)));
        OwnedPyObject assistant_cuda_weight_bytes(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer_assistant
                ? handle->glimmer_assistant->cuda_resident_weight_bytes() : 0)));
        OwnedPyObject assistant_cuda_workspace_bytes(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer_assistant
                ? handle->glimmer_assistant->cuda_workspace_bytes() : 0)));
        OwnedPyObject assistant_cuda_launches(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer_assistant
                ? handle->glimmer_assistant->cuda_kernel_launches() : 0)));
        OwnedPyObject assistant_cuda_k_quant_mmq_linears(PyLong_FromLongLong(
            static_cast<long long>(
                handle->glimmer_assistant
                    ? handle->glimmer_assistant->cuda_k_quant_mmq_linears() : 0)));
        OwnedPyObject vision_weight_bytes(PyLong_FromLongLong(static_cast<long long>(
            handle->glimmer->vision_weight_bytes())));
        if (!cuda_weight_bytes || !cuda_workspace_bytes || !cuda_launches ||
            !cuda_k_quant_mmq_linears ||
            !cuda_q8_quantizations || !cuda_q8_linears ||
            !cuda_argmax_calls || !cuda_argmax_rows ||
            !cuda_device_value || !cuda_tile_library || !cuda_runtime_library ||
            !cpu_compute_rows || !assistant_weight_bytes || !assistant_cuda_weight_bytes ||
            !assistant_cuda_workspace_bytes || !assistant_cuda_launches ||
            !assistant_cuda_k_quant_mmq_linears ||
            !vision_weight_bytes ||
            PyDict_SetItemString(
                result, "whole_model_cuda", whole_model_cuda ? Py_True : Py_False) < 0 ||
            PyDict_SetItemString(
                result, "cuda_model_compute_only", whole_model_cuda ? Py_True : Py_False) < 0 ||
            PyDict_SetItemString(result, "cuda_resident_weight_bytes", cuda_weight_bytes.get()) < 0 ||
            PyDict_SetItemString(result, "cuda_workspace_bytes", cuda_workspace_bytes.get()) < 0 ||
            PyDict_SetItemString(result, "cuda_kernel_launches", cuda_launches.get()) < 0 ||
            PyDict_SetItemString(
                result, "cuda_k_quant_mmq_linears",
                cuda_k_quant_mmq_linears.get()) < 0 ||
            PyDict_SetItemString(
                result, "cuda_q8_activation_quantizations",
                cuda_q8_quantizations.get()) < 0 ||
            PyDict_SetItemString(
                result, "cuda_q8_packed_linears", cuda_q8_linears.get()) < 0 ||
            PyDict_SetItemString(
                result, "cuda_device_argmax_calls", cuda_argmax_calls.get()) < 0 ||
            PyDict_SetItemString(
                result, "cuda_device_argmax_rows", cuda_argmax_rows.get()) < 0 ||
            PyDict_SetItemString(result, "cuda_device", cuda_device_value.get()) < 0 ||
            PyDict_SetItemString(result, "cuda_tile_ops_lib", cuda_tile_library.get()) < 0 ||
            PyDict_SetItemString(result, "cuda_runtime_lib", cuda_runtime_library.get()) < 0 ||
            PyDict_SetItemString(result, "cpu_model_compute_rows", cpu_compute_rows.get()) < 0 ||
            PyDict_SetItemString(
                result, "dflash_loaded", handle->glimmer_assistant ? Py_True : Py_False) < 0 ||
            PyDict_SetItemString(
                result, "dflash_resident_weight_bytes", assistant_weight_bytes.get()) < 0 ||
            PyDict_SetItemString(
                result, "dflash_cuda_resident_weight_bytes",
                assistant_cuda_weight_bytes.get()) < 0 ||
            PyDict_SetItemString(
                result, "dflash_cuda_workspace_bytes",
                assistant_cuda_workspace_bytes.get()) < 0 ||
            PyDict_SetItemString(
                result, "dflash_cuda_kernel_launches", assistant_cuda_launches.get()) < 0 ||
            PyDict_SetItemString(
                result, "dflash_cuda_k_quant_mmq_linears",
                assistant_cuda_k_quant_mmq_linears.get()) < 0 ||
            PyDict_SetItemString(result, "dflash_cuda", dflash_cuda ? Py_True : Py_False) < 0 ||
            PyDict_SetItemString(
                result, "vision_loaded", handle->glimmer->has_vision() ? Py_True : Py_False) < 0 ||
            PyDict_SetItemString(
                result, "vision_resident_weight_bytes", vision_weight_bytes.get()) < 0 ||
            PyDict_SetItemString(
                result, "vision_cuda",
                handle->glimmer->vision_whole_model_cuda()
                    ? Py_True : Py_False) < 0) {
            Py_DECREF(result);
            return nullptr;
        }
    }
    if (handle->kind == ModelHandle::Kind::Moe) {
        OwnedPyObject experts(PyLong_FromLongLong(static_cast<long long>(
            handle->moe->experts())));
        OwnedPyObject top_k(PyLong_FromLongLong(static_cast<long long>(
            handle->moe->top_k())));
        OwnedPyObject mlp_multiplier(PyFloat_FromDouble(
            handle->moe->mlp_multiplier()));
        OwnedPyObject multiple_of(PyLong_FromLongLong(static_cast<long long>(
            handle->moe->multiple_of())));
        OwnedPyObject router_aux_loss_coef(PyFloat_FromDouble(
            handle->moe->router_aux_loss_coef()));
        if (!experts || !top_k || !mlp_multiplier || !multiple_of ||
            !router_aux_loss_coef ||
            PyDict_SetItemString(result, "experts", experts.get()) < 0 ||
            PyDict_SetItemString(result, "top_k", top_k.get()) < 0 ||
            PyDict_SetItemString(result, "mlp_multiplier", mlp_multiplier.get()) < 0 ||
            PyDict_SetItemString(result, "multiple_of", multiple_of.get()) < 0 ||
            PyDict_SetItemString(
                result, "router_aux_loss_coef", router_aux_loss_coef.get()) < 0) {
            Py_DECREF(result);
            return nullptr;
        }
    }
    return result;
}

PyObject* session_stats(PyObject*, PyObject* args) {
    PyObject* model_capsule = nullptr;
    PyObject* session_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO:session_stats", &model_capsule, &session_capsule)) {
        return nullptr;
    }
    ModelHandle* model = nullptr;
    SessionHandle* session = nullptr;
    if (!owned_session(model_capsule, session_capsule, &model, &session)) {
        return nullptr;
    }
    const auto stats = session->kind == ModelHandle::Kind::Dense
        ? session->dense->stats()
        : (session->kind == ModelHandle::Kind::Llama
            ? session->llama->stats()
            : (session->kind == ModelHandle::Kind::Glimmer
                ? session->glimmer->stats()
                : session->moe->stats()));
    const bool full_cache = stats.cache_mode == KVCacheMode::Full;
    const bool turboquant_cache = stats.cache_mode == KVCacheMode::TurboQuant;
    const double compression_ratio = stats.cache_bytes == 0
        ? 1.0
        : static_cast<double>(stats.uncompressed_cache_bytes) /
            static_cast<double>(stats.cache_bytes);
    PyObject* result = Py_BuildValue(
        "{s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:d,s:O,s:O,s:O,s:O,s:s,s:O,s:O,s:L}",
        "token_count", static_cast<long long>(stats.token_count),
        "prefill_calls", static_cast<long long>(stats.prefill_calls),
        "prefill_tokens", static_cast<long long>(stats.prefill_tokens),
        "decode_calls", static_cast<long long>(stats.decode_calls),
        "truncate_calls", static_cast<long long>(stats.truncate_calls),
        "reset_calls", static_cast<long long>(stats.reset_calls),
        "cached_tokens", static_cast<long long>(stats.cached_tokens),
        "cache_bytes", static_cast<long long>(stats.cache_bytes),
        "cache_capacity_bytes", static_cast<long long>(stats.cache_capacity_bytes),
        "uncompressed_cache_bytes", static_cast<long long>(stats.uncompressed_cache_bytes),
        "compression_ratio", compression_ratio,
        "strict_model_compute", stats.strict_model_compute ? Py_True : Py_False,
        "lossy_cache", stats.lossy_cache ? Py_True : Py_False,
        "cancelled", stats.cancelled ? Py_True : Py_False,
        "closed", stats.closed ? Py_True : Py_False,
        "effective_cache", turboquant_cache ? "turboquant" : (full_cache ? "full" : "off"),
        "recompute_full_prefix", stats.cache_mode == KVCacheMode::Off ? Py_True : Py_False,
        "fallback_reason", Py_None,
        "decode_rows_processed", static_cast<long long>(stats.decode_rows_processed));
    if (result == nullptr) {
        return nullptr;
    }
    OwnedPyObject speculative_blocks(PyLong_FromLongLong(
        static_cast<long long>(stats.speculative_blocks)));
    OwnedPyObject speculative_proposed(PyLong_FromLongLong(
        static_cast<long long>(stats.speculative_proposed_tokens)));
    OwnedPyObject speculative_accepted(PyLong_FromLongLong(
        static_cast<long long>(stats.speculative_accepted_tokens)));
    OwnedPyObject speculative_rejected(PyLong_FromLongLong(
        static_cast<long long>(stats.speculative_rejected_tokens)));
    OwnedPyObject assistant_cache_bytes(PyLong_FromLongLong(
        static_cast<long long>(stats.assistant_cache_bytes)));
    if (!speculative_blocks || !speculative_proposed || !speculative_accepted ||
        !speculative_rejected || !assistant_cache_bytes ||
        PyDict_SetItemString(
            result, "speculative_decoding", stats.speculative_decoding ? Py_True : Py_False) < 0 ||
        PyDict_SetItemString(result, "speculative_blocks", speculative_blocks.get()) < 0 ||
        PyDict_SetItemString(result, "speculative_proposed_tokens", speculative_proposed.get()) < 0 ||
        PyDict_SetItemString(result, "speculative_accepted_tokens", speculative_accepted.get()) < 0 ||
        PyDict_SetItemString(result, "speculative_rejected_tokens", speculative_rejected.get()) < 0 ||
        PyDict_SetItemString(result, "assistant_cache_bytes", assistant_cache_bytes.get()) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    const bool tile_attention =
        turboquant_cache && stats.turboquant_attention_backend == "tile-cuda";
    OwnedPyObject attention_backend(PyUnicode_FromString(
        turboquant_cache ? stats.turboquant_attention_backend.c_str() : "cpu"));
    OwnedPyObject tile_ops_lib(
        tile_attention
            ? PyUnicode_FromString(stats.turboquant_tile_ops_lib.c_str())
            : nullptr);
    OwnedPyObject cuda_runtime_lib(
        tile_attention
            ? PyUnicode_FromString(stats.turboquant_cuda_runtime_lib.c_str())
            : nullptr);
    OwnedPyObject cuda_device(
        tile_attention
            ? PyLong_FromLongLong(static_cast<long long>(stats.turboquant_cuda_device))
            : nullptr);
    OwnedPyObject gpu_launches(PyLong_FromLongLong(static_cast<long long>(
        stats.turboquant_gpu_launches)));
    OwnedPyObject row_uploads(PyLong_FromLongLong(static_cast<long long>(
        stats.turboquant_row_uploads)));
    OwnedPyObject h2d_bytes(PyLong_FromLongLong(static_cast<long long>(
        stats.turboquant_h2d_bytes)));
    OwnedPyObject d2h_bytes(PyLong_FromLongLong(static_cast<long long>(
        stats.turboquant_d2h_bytes)));
    OwnedPyObject cpu_attention_calls(PyLong_FromLongLong(static_cast<long long>(
        stats.turboquant_cpu_compressed_attention_calls)));
    OwnedPyObject prefix_cow_forks_created(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_forks_created)));
    OwnedPyObject prefix_cow_forked_from_tokens(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_forked_from_tokens)));
    OwnedPyObject prefix_cow_storage_use_count(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_storage_use_count)));
    OwnedPyObject prefix_cow_shared_cached_tokens(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_shared_cached_tokens)));
    OwnedPyObject prefix_cow_shared_capacity_bytes(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_shared_capacity_bytes)));
    OwnedPyObject prefix_cow_detach_count(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_detach_count)));
    OwnedPyObject prefix_cow_detached_capacity_bytes(PyLong_FromLongLong(
        static_cast<long long>(stats.prefix_cow_detached_capacity_bytes)));
    OwnedPyObject prefix_cow_shared_cached_tokens_scope(PyUnicode_FromString(
        "this-session-valid-rows-in-shared-allocation"));
    if (!attention_backend || !gpu_launches || !row_uploads || !h2d_bytes ||
        !d2h_bytes || !cpu_attention_calls || !prefix_cow_forks_created ||
        !prefix_cow_forked_from_tokens || !prefix_cow_storage_use_count ||
        !prefix_cow_shared_cached_tokens || !prefix_cow_shared_capacity_bytes ||
        !prefix_cow_detach_count || !prefix_cow_detached_capacity_bytes ||
        !prefix_cow_shared_cached_tokens_scope ||
        (tile_attention && (!tile_ops_lib || !cuda_runtime_lib || !cuda_device)) ||
        PyDict_SetItemString(
            result, "turboquant_attention_backend", attention_backend.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_tile_ops_lib",
            tile_attention ? tile_ops_lib.get() : Py_None) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_cuda_runtime_lib",
            tile_attention ? cuda_runtime_lib.get() : Py_None) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_cuda_device",
            tile_attention ? cuda_device.get() : Py_None) < 0 ||
        PyDict_SetItemString(result, "turboquant_gpu_launches", gpu_launches.get()) < 0 ||
        PyDict_SetItemString(result, "turboquant_row_uploads", row_uploads.get()) < 0 ||
        PyDict_SetItemString(result, "turboquant_h2d_bytes", h2d_bytes.get()) < 0 ||
        PyDict_SetItemString(result, "turboquant_d2h_bytes", d2h_bytes.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "turboquant_cpu_compressed_attention_calls",
            cpu_attention_calls.get()) < 0 ||
        PyDict_SetItemString(
            result, "prefix_cow_forks_created", prefix_cow_forks_created.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_forked_from_tokens",
            prefix_cow_forked_from_tokens.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_storage_use_count",
            prefix_cow_storage_use_count.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_shared_storage",
            stats.prefix_cow_shared_capacity_bytes > 0 ? Py_True : Py_False) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_shared_cached_tokens",
            prefix_cow_shared_cached_tokens.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_shared_cached_tokens_scope",
            prefix_cow_shared_cached_tokens_scope.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_shared_capacity_bytes",
            prefix_cow_shared_capacity_bytes.get()) < 0 ||
        PyDict_SetItemString(
            result, "prefix_cow_detach_count", prefix_cow_detach_count.get()) < 0 ||
        PyDict_SetItemString(
            result,
            "prefix_cow_detached_capacity_bytes",
            prefix_cow_detached_capacity_bytes.get()) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}

PyObject* turboquant_codec_probe(PyObject*, PyObject* args) {
    PyObject* cache = nullptr;
    PyObject* key_object = nullptr;
    PyObject* value_object = nullptr;
    PyObject* query_object = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "O!OOO:turboquant_codec_probe",
            &PyDict_Type,
            &cache,
            &key_object,
            &value_object,
            &query_object)) {
        return nullptr;
    }
    TurboQuantTables tables;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> query;
    if (!turboquant_tables(cache, &tables) ||
        !float_vector(key_object, "key", &key) ||
        !float_vector(value_object, "value", &value) ||
        !float_vector(query_object, "query", &query)) {
        return nullptr;
    }
    try {
        TurboQuantCodec codec(std::move(tables));
        const std::size_t dimension = static_cast<std::size_t>(codec.dimension());
        if (key.size() != dimension || value.size() != dimension || query.size() != dimension) {
            PyErr_SetString(
                PyExc_ValueError,
                "TurboQuant probe key, value, and query must match the table dimension");
            return nullptr;
        }
        const auto encoded_key = codec.encode_key(key.data());
        const auto encoded_value = codec.encode_value(value.data());
        std::vector<float> decoded_value(dimension, 0.0f);
        codec.accumulate_value(decoded_value.data(), 1.0, encoded_value);
        const double inner = codec.key_inner_product(query.data(), encoded_key);

        PyObject* result = PyDict_New();
        PyObject* key_indices = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(encoded_key.packed_indices.data()),
            static_cast<Py_ssize_t>(encoded_key.packed_indices.size()));
        PyObject* value_indices = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(encoded_value.packed_indices.data()),
            static_cast<Py_ssize_t>(encoded_value.packed_indices.size()));
        PyObject* qjl_signs = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(encoded_key.qjl_signs.data()),
            static_cast<Py_ssize_t>(encoded_key.qjl_signs.size()));
        PyObject* decoded = PyList_New(static_cast<Py_ssize_t>(decoded_value.size()));
        if (result == nullptr || key_indices == nullptr || value_indices == nullptr ||
            qjl_signs == nullptr || decoded == nullptr) {
            Py_XDECREF(result);
            Py_XDECREF(key_indices);
            Py_XDECREF(value_indices);
            Py_XDECREF(qjl_signs);
            Py_XDECREF(decoded);
            return nullptr;
        }
        for (std::size_t index = 0; index < decoded_value.size(); ++index) {
            PyObject* coordinate = PyFloat_FromDouble(decoded_value[index]);
            if (coordinate == nullptr) {
                Py_DECREF(result);
                Py_DECREF(key_indices);
                Py_DECREF(value_indices);
                Py_DECREF(qjl_signs);
                Py_DECREF(decoded);
                return nullptr;
            }
            PyList_SET_ITEM(decoded, static_cast<Py_ssize_t>(index), coordinate);
        }
        PyObject* key_norm = PyFloat_FromDouble(encoded_key.norm);
        PyObject* value_norm = PyFloat_FromDouble(encoded_value.norm);
        PyObject* residual_norm = PyFloat_FromDouble(encoded_key.residual_norm);
        PyObject* inner_product = PyFloat_FromDouble(inner);
        const bool inserted = key_norm != nullptr && value_norm != nullptr &&
            residual_norm != nullptr && inner_product != nullptr &&
            PyDict_SetItemString(result, "key_norm", key_norm) == 0 &&
            PyDict_SetItemString(result, "value_norm", value_norm) == 0 &&
            PyDict_SetItemString(result, "residual_norm", residual_norm) == 0 &&
            PyDict_SetItemString(result, "key_indices", key_indices) == 0 &&
            PyDict_SetItemString(result, "value_indices", value_indices) == 0 &&
            PyDict_SetItemString(result, "qjl_signs", qjl_signs) == 0 &&
            PyDict_SetItemString(result, "key_inner_product", inner_product) == 0 &&
            PyDict_SetItemString(result, "decoded_value", decoded) == 0;
        Py_XDECREF(key_norm);
        Py_XDECREF(value_norm);
        Py_XDECREF(residual_norm);
        Py_XDECREF(inner_product);
        Py_DECREF(key_indices);
        Py_DECREF(value_indices);
        Py_DECREF(qjl_signs);
        Py_DECREF(decoded);
        if (!inserted) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (const std::exception& error) {
        if (PyErr_Occurred()) {
            return nullptr;
        }
        return return_cpp_error(error);
    }
}

PyMethodDef methods[] = {
    {"resident_inference_abi_version", resident_inference_abi_version, METH_NOARGS, "Return resident inference ABI version 1."},
    {"resident_inference_capabilities", resident_inference_capabilities, METH_NOARGS, "Return fail-closed resident inference capabilities."},
    {"load_model", load_model, METH_VARARGS, "Load immutable validated native weights once."},
    {"load_model_with_options", load_model_with_options, METH_VARARGS, "Load a proved whole-model CUDA model after VRAM selection."},
    {"load_companion", load_companion, METH_VARARGS, "Load and bind a strict Glimmer DFlash companion before sessions exist."},
    {"close_model", close_model, METH_VARARGS, "Close a resident inference model."},
    {"configure_model_turboquant_attention", configure_model_turboquant_attention, METH_VARARGS, "Configure explicit strict Tile-CUDA TurboQuant attention for a dense model."},
    {"create_session", create_session, METH_VARARGS, "Create isolated resident inference session state."},
    {"fork_session", fork_session, METH_VARARGS, "Fork one supported full-cache or dense CPU TurboQuant session prefix with copy-on-write storage."},
    {"close_session", close_session, METH_VARARGS, "Close a resident inference session."},
    {"prefill", prefill, METH_VARARGS, "Append a validated token suffix to resident session state."},
    {"prefill_with_embeddings", prefill_with_embeddings, METH_VARARGS, "Append Glimmer tokens with exact image/video embedding replacements."},
    {"encode_media", encode_media, METH_VARARGS, "Encode packed Glimmer image/video patches into decoder replacement rows."},
    {"current_logits", current_logits, METH_VARARGS, "Return current logits for native cache parity diagnostics."},
    {"decode_one", decode_one, METH_VARARGS, "Run one real decode and commit its sampled token."},
    {"decode_speculative_block", decode_speculative_block, METH_VARARGS, "Atomically propose, verify, and commit one DFlash block."},
    {"truncate_session", truncate_session, METH_VARARGS, "Truncate resident token history."},
    {"reset_session", reset_session, METH_VARARGS, "Reset resident token and cancellation state."},
    {"cancel_session", cancel_session, METH_VARARGS, "Cancel resident work without spawning a process."},
    {"model_stats", model_stats, METH_VARARGS, "Return resident model telemetry."},
    {"session_stats", session_stats, METH_VARARGS, "Return isolated resident session telemetry."},
    {"turboquant_codec_probe", turboquant_codec_probe, METH_VARARGS, "Compare native packed codec output with the portable oracle."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native_inference",
    "In-process NeuralFn resident inference binding.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__native_inference() {
    return PyModule_Create(&module);
}
