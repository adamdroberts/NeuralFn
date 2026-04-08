# Datasets

Dataset catalog endpoints for downloading, uploading, and managing dataset access across projects.

All endpoints are prefixed with:

```
/api/projects/{project_id}/datasets
```

**Authentication:** required for all endpoints.

---

## GET /

Lists all datasets accessible to the project.

**Response:**

```json
[
  {
    "name": "tiny_shakespeare",
    "source": "huggingface",
    "hf_path": "karpathy/tiny_shakespeare",
    "rows": 40000,
    "size_bytes": 1115394
  },
  {
    "name": "custom_upload",
    "source": "upload",
    "rows": 500,
    "size_bytes": 24000
  }
]
```

---

## POST /download

Downloads a dataset from HuggingFace and registers it in the catalog.

**Request Body:**

```json
{
  "hf_path": "karpathy/tiny_shakespeare",
  "hf_split": "train",
  "max_rows": 50000,
  "variant": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hf_path` | string | yes | HuggingFace dataset path (e.g. `"karpathy/tiny_shakespeare"`). |
| `hf_split` | string | no | Dataset split. Defaults to `"train"`. |
| `max_rows` | int | no | Maximum number of rows to download. |
| `variant` | string | no | Dataset variant/configuration name. |

Additional optional fields may be accepted depending on the dataset type.

**Response:** download result with dataset metadata.

---

## POST /upload

Uploads a local file as a dataset.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | The dataset file (CSV, JSON, JSONL, Parquet, or text). |
| `name` | string | yes | Display name for the dataset. |
| `project_ids` | string | no | Comma-separated project IDs that should have access. Defaults to the current project. |

**Response:** upload result with dataset metadata.

---

## PUT /{ds_name}/access

Updates which projects can access a dataset.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `ds_name` | The dataset name (as returned by the list endpoint). |

**Request Body:**

```json
{
  "project_ids": ["p_xyz", "p_abc"]
}
```

**Response:** confirmation with the updated access list.

---

## DELETE /{ds_name}

Deletes a dataset from the catalog and removes its files.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `ds_name` | The dataset name to delete. |

**Response:** deletion confirmation.

Returns `404` if the dataset does not exist or is not accessible to the project.
