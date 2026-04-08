# Dataset Tools

Tools for managing the dataset catalog: listing, downloading from HuggingFace, loading into graphs, and controlling project access.

---

## list_datasets

Lists all datasets accessible to the project.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |

**Returns:** list of dataset metadata objects (name, source, row count, size).

---

## download_dataset

Downloads a dataset from HuggingFace and registers it in the NeuralFn catalog.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `hf_path` | string | yes | | HuggingFace dataset path (e.g. `"karpathy/tiny_shakespeare"`). |
| `hf_split` | string | no | `"train"` | Dataset split to download. |
| `max_rows` | int | no | | Maximum rows to download. |
| `variant` | string | no | | Dataset variant/configuration. |
| `train_shards` | int | no | | Number of training shards to create. |
| `skip_manifest` | bool | no | | Skip manifest generation. |
| `with_docs` | bool | no | | Include dataset documentation card. |
| `repo_id` | string | no | | Alternative repository identifier. |
| `remote_root_prefix` | string | no | `"datasets"` | Remote storage prefix. |
| `project_ids` | list[string] | no | | Additional projects to grant access to. |

**Returns:** download result with dataset metadata.

---

## load_dataset_source

Loads a dataset into the session graph as a data-source node. This both registers the dataset (if not already present) and wires it into the graph.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `project_id` | string | yes | | Project ID. |
| `session_id` | string | yes | | Session ID. |
| `dataset_names` | list[string] | no | | Names of already-downloaded datasets to load. |
| `hf_path` | string | no | | HuggingFace path to download and load in one step. |
| `hf_split` | string | no | | Dataset split. |
| `text_column` | string | no | | Column to use as text input. |
| `max_rows` | int | no | | Maximum rows. |
| `alias` | string | no | | Display alias for the data-source node. |
| `variant` | string | no | | Dataset variant. |
| `train_shards` | int | no | | Number of training shards. |
| `skip_manifest` | bool | no | | Skip manifest generation. |
| `with_docs` | bool | no | | Include documentation card. |
| `repo_id` | string | no | | Alternative repository identifier. |
| `remote_root_prefix` | string | no | | Remote storage prefix. |
| `seq_len` | int | no | `64` | Sequence length for tokenization. |
| `node_id` | string | no | `"dataset_source"` | ID for the data-source node in the graph. |
| `append` | bool | no | | Append to existing data-source node instead of replacing. |
| `project_ids` | list[string] | no | | Additional projects to grant access to. |

**Returns:** result with the updated graph and revision.

---

## set_dataset_access

Updates which projects can access a specific dataset.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID (owner context). |
| `ds_name` | string | yes | Dataset name. |
| `project_ids` | list[string] | yes | Full list of project IDs that should have access. |

**Returns:** confirmation with the updated access list.

---

## delete_dataset

Deletes a dataset from the catalog and removes its stored files.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `project_id` | string | yes | Project ID. |
| `ds_name` | string | yes | Dataset name to delete. |

**Returns:** deletion confirmation.
