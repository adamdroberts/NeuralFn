from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from neuralfn.semantic import ConversationalVocabulary, NUM_SEMANTIC_DIMS


class SettingsDefaultsTest(unittest.TestCase):
    def test_artifacts_dir_defaults_to_home_neuralfn_artifacts(self) -> None:
        import server.settings as settings_module

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEURALFN_ARTIFACTS_DIR", None)
            settings_module.get_settings.cache_clear()
            try:
                settings = settings_module.get_settings()
                self.assertEqual(Path.home() / "NeuralFn" / "artifacts", settings.artifacts_dir)
            finally:
                settings_module.get_settings.cache_clear()


class PlatformApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="neuralfn-platform-api-")
        os.environ["NEURALFN_DATABASE_URL"] = f"sqlite:///{self.temp_dir}/platform.db"
        os.environ["NEURALFN_CREATE_SCHEMA_ON_STARTUP"] = "1"
        os.environ["NEURALFN_SNAPSHOTS_DIR"] = os.path.join(self.temp_dir, "session_snapshots")
        os.environ["NEURALFN_ARTIFACTS_DIR"] = os.path.join(self.temp_dir, "artifacts")
        os.environ.pop("NEURALFN_REDIS_URL", None)

        import server.settings as settings_module
        import server.db as db_module
        import server.services.live_state as live_state_module
        import server.services.auth_service as auth_service_module
        import server.services.dataset_service as dataset_service_module
        import server.services.session_service as session_service_module
        import server.services.run_service as run_service_module
        import server.dataset_manager as dataset_manager_module
        import server.app as app_module

        settings_module.get_settings.cache_clear()
        db_module.get_engine.cache_clear()
        db_module.get_session_factory.cache_clear()
        live_state_module.get_live_state_store.cache_clear()
        auth_service_module._auth_service = None
        dataset_service_module._dataset_service = None
        session_service_module._workspace_service = None
        run_service_module._run_service = None
        self._original_datasets_dir = dataset_manager_module.DATASETS_DIR
        dataset_manager_module.DATASETS_DIR = Path(self.temp_dir) / "datasets"
        dataset_manager_module.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

        app_module = importlib.reload(app_module)
        self.client_manager = TestClient(app_module.app)
        self.client = self.client_manager.__enter__()

    def tearDown(self) -> None:
        self.client_manager.__exit__(None, None, None)
        import server.db as db_module
        import server.dataset_manager as dataset_manager_module
        db_module.get_engine().dispose()
        db_module.get_engine.cache_clear()
        db_module.get_session_factory.cache_clear()
        dataset_manager_module.DATASETS_DIR = self._original_datasets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.pop("NEURALFN_DATABASE_URL", None)
        os.environ.pop("NEURALFN_CREATE_SCHEMA_ON_STARTUP", None)
        os.environ.pop("NEURALFN_SNAPSHOTS_DIR", None)
        os.environ.pop("NEURALFN_ARTIFACTS_DIR", None)
        os.environ.pop("NEURALFN_REDIS_URL", None)

    def _bootstrap_admin(self) -> dict:
        response = self.client.post(
            "/api/auth/bootstrap-admin",
            json={
                "email": "admin@example.com",
                "password": "secret123",
                "display_name": "Admin",
            },
        )
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def _create_user(
        self,
        *,
        email: str = "user@example.com",
        password: str = "secret123",
        display_name: str = "User",
    ) -> dict:
        response = self.client.post(
            "/api/admin/users",
            json={
                "email": email,
                "password": password,
                "display_name": display_name,
                "is_admin": False,
            },
        )
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def _login(self, *, email: str, password: str) -> dict:
        response = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def test_bootstrap_admin_and_refresh_restore_active_session(self) -> None:
        initial = self.client.get("/api/bootstrap")
        self.assertEqual(200, initial.status_code)
        self.assertTrue(initial.json()["requires_setup"])
        self.assertFalse(initial.json()["authenticated"])

        self._bootstrap_admin()

        bootstrapped = self.client.get("/api/bootstrap")
        self.assertEqual(200, bootstrapped.status_code, bootstrapped.text)
        payload = bootstrapped.json()
        self.assertFalse(payload["requires_setup"])
        self.assertTrue(payload["authenticated"])
        self.assertEqual("admin@example.com", payload["user"]["email"])

        project_id = payload["active_project_id"]
        session_id = payload["active_session_id"]
        revision = payload["active_session"]["revision"]
        graph = payload["active_session"]["graph"]
        graph["name"] = "restored after refresh"

        updated = self.client.put(
            f"/api/projects/{project_id}/sessions/{session_id}/graph",
            json={
                "graph": graph,
                "expected_revision": revision,
                "persist_snapshot": True,
                "snapshot_reason": "test_restore",
            },
        )
        self.assertEqual(200, updated.status_code, updated.text)
        updated_payload = updated.json()
        self.assertEqual("restored after refresh", updated_payload["graph"]["name"])
        self.assertGreater(updated_payload["revision"], revision)

        refreshed = self.client.get("/api/bootstrap")
        self.assertEqual(200, refreshed.status_code, refreshed.text)
        refreshed_payload = refreshed.json()
        self.assertEqual("restored after refresh", refreshed_payload["active_session"]["graph"]["name"])
        self.assertEqual(updated_payload["revision"], refreshed_payload["active_session"]["revision"])

    def test_active_session_switch_and_run_idle_endpoint(self) -> None:
        self._bootstrap_admin()
        payload = self.client.get("/api/bootstrap").json()
        project_id = payload["active_project_id"]
        first_session_id = payload["active_session_id"]

        created = self.client.post(
            f"/api/projects/{project_id}/sessions",
            json={"name": "Second Session"},
        )
        self.assertEqual(200, created.status_code, created.text)
        second_session_id = created.json()["id"]
        self.assertNotEqual(first_session_id, second_session_id)

        switched = self.client.put(
            "/api/auth/active-session",
            json={"project_id": project_id, "session_id": second_session_id},
        )
        self.assertEqual(200, switched.status_code, switched.text)

        refreshed = self.client.get("/api/bootstrap")
        self.assertEqual(200, refreshed.status_code, refreshed.text)
        refreshed_payload = refreshed.json()
        self.assertEqual(second_session_id, refreshed_payload["active_session_id"])
        self.assertEqual(second_session_id, refreshed_payload["active_session"]["session"]["id"])

        run_status = self.client.get(f"/api/projects/{project_id}/sessions/{second_session_id}/runs/active")
        self.assertEqual(200, run_status.status_code, run_status.text)
        self.assertEqual("idle", run_status.json()["status"])
        self.assertFalse(run_status.json()["running"])

    def test_graph_revision_conflict_returns_409(self) -> None:
        self._bootstrap_admin()
        payload = self.client.get("/api/bootstrap").json()
        project_id = payload["active_project_id"]
        session_id = payload["active_session_id"]
        revision = payload["active_session"]["revision"]
        graph = payload["active_session"]["graph"]

        graph_a = dict(graph)
        graph_a["name"] = "revision-a"
        first_update = self.client.put(
            f"/api/projects/{project_id}/sessions/{session_id}/graph",
            json={"graph": graph_a, "expected_revision": revision},
        )
        self.assertEqual(200, first_update.status_code, first_update.text)

        graph_b = dict(graph)
        graph_b["name"] = "revision-b"
        conflict = self.client.put(
            f"/api/projects/{project_id}/sessions/{session_id}/graph",
            json={"graph": graph_b, "expected_revision": revision},
        )
        self.assertEqual(409, conflict.status_code, conflict.text)
        detail = conflict.json()["detail"]
        self.assertIn("current_revision", detail)
        self.assertGreater(detail["current_revision"], revision)

    def test_non_admin_can_create_project_with_seeded_main_session(self) -> None:
        self._bootstrap_admin()
        self._create_user(email="builder@example.com", display_name="Builder")

        logout = self.client.post("/api/auth/logout")
        self.assertEqual(200, logout.status_code, logout.text)

        self._login(email="builder@example.com", password="secret123")
        initial = self.client.get("/api/bootstrap")
        self.assertEqual(200, initial.status_code, initial.text)
        self.assertEqual([], initial.json()["projects"])

        created = self.client.post(
            "/api/projects",
            json={
                "name": "Builder Personal",
                "description": "Personal workspace",
            },
        )
        self.assertEqual(200, created.status_code, created.text)
        payload = created.json()
        self.assertEqual("Builder Personal", payload["project"]["name"])
        self.assertEqual("Main session", payload["default_session"]["name"])

        refreshed = self.client.get("/api/bootstrap")
        self.assertEqual(200, refreshed.status_code, refreshed.text)
        refreshed_payload = refreshed.json()
        self.assertEqual(payload["project"]["id"], refreshed_payload["active_project_id"])
        self.assertEqual(payload["default_session"]["id"], refreshed_payload["active_session_id"])
        self.assertEqual(1, len(refreshed_payload["projects"]))

        sessions = self.client.get(f"/api/projects/{payload['project']['id']}/sessions")
        self.assertEqual(200, sessions.status_code, sessions.text)
        session_payload = sessions.json()
        self.assertEqual(1, len(session_payload))
        self.assertEqual(payload["default_session"]["id"], session_payload[0]["id"])

    def test_session_apply_endpoint_loads_full_jepa_semantic_root_graph(self) -> None:
        self._bootstrap_admin()
        payload = self.client.get("/api/bootstrap").json()
        project_id = payload["active_project_id"]
        session_id = payload["active_session_id"]

        applied = self.client.post(
            f"/api/projects/{project_id}/sessions/{session_id}/templates/gpt/apply",
            json={"name": "jepa_semantic", "config": {"preset": "jepa_semantic_hybrid"}},
        )
        self.assertEqual(200, applied.status_code, applied.text)
        graph = applied.json()["graph"]

        self.assertIn("dataset_source", graph["nodes"])
        self.assertIn("semantic_data_source", graph["nodes"])
        self.assertIn("model", graph["nodes"])
        self.assertIn("loss_out", graph["nodes"])
        self.assertNotIn("tokens_in", graph["nodes"])
        self.assertEqual(["dataset_source", "semantic_data_source"], graph["input_node_ids"])
        self.assertEqual(["loss_out"], graph["output_node_ids"])

        root_edges_to_model = [
            edge for edge in graph["edges"].values() if edge["dst_node"] == "model"
        ]
        self.assertEqual(2, len(root_edges_to_model))

    def test_semantic_dimensions_returns_dynamic_vocab_topic_counts(self) -> None:
        self._bootstrap_admin()
        payload = self.client.get("/api/bootstrap").json()
        project_id = payload["active_project_id"]
        session_id = payload["active_session_id"]

        response = self.client.get(
            f"/api/projects/{project_id}/sessions/{session_id}/semantic/dimensions"
        )
        self.assertEqual(200, response.status_code, response.text)
        dims = response.json()
        by_name = {item["name"]: item for item in dims}
        self.assertEqual(NUM_SEMANTIC_DIMS, len(dims))

        vocab = ConversationalVocabulary()
        for dim_name in vocab.dim_names:
            self.assertEqual(len(vocab.terms(dim_name)), by_name[dim_name]["num_topics"])
        self.assertGreater(by_name["entity_type"]["num_topics"], 40)
        self.assertIsNone(by_name["taxonomy_hash"]["expert_id"])
        self.assertEqual(0, by_name["taxonomy_hash"]["num_topics"])

    def test_dataset_access_filtering_and_graph_driven_run_resolution(self) -> None:
        self._bootstrap_admin()
        self._create_user(email="datasets@example.com", display_name="Datasets User")

        logout = self.client.post("/api/auth/logout")
        self.assertEqual(200, logout.status_code, logout.text)
        self._login(email="datasets@example.com", password="secret123")

        project_a = self.client.post("/api/projects", json={"name": "Project A"}).json()
        project_b = self.client.post("/api/projects", json={"name": "Project B"}).json()
        project_a_id = project_a["project"]["id"]
        session_a_id = project_a["default_session"]["id"]
        project_b_id = project_b["project"]["id"]

        upload = self.client.post(
            f"/api/projects/{project_a_id}/datasets/upload",
            data={
                "name": "tiny_tokens",
                "project_ids": json.dumps([project_a_id]),
            },
            files={"file": ("tiny.txt", b"hello neuralfn dataset\n" * 64, "text/plain")},
        )
        self.assertEqual(200, upload.status_code, upload.text)
        self.assertEqual([project_a_id], upload.json()["project_ids"])

        list_a = self.client.get(f"/api/projects/{project_a_id}/datasets")
        self.assertEqual(200, list_a.status_code, list_a.text)
        self.assertEqual(["tiny_tokens"], [item["name"] for item in list_a.json()])

        list_b = self.client.get(f"/api/projects/{project_b_id}/datasets")
        self.assertEqual(200, list_b.status_code, list_b.text)
        self.assertEqual([], list_b.json())

        shared = self.client.put(
            f"/api/projects/{project_a_id}/datasets/tiny_tokens/access",
            json={"project_ids": [project_a_id, project_b_id]},
        )
        self.assertEqual(200, shared.status_code, shared.text)
        self.assertEqual(sorted([project_a_id, project_b_id]), sorted(shared.json()["project_ids"]))

        shared_list_b = self.client.get(f"/api/projects/{project_b_id}/datasets")
        self.assertEqual(200, shared_list_b.status_code, shared_list_b.text)
        self.assertEqual(["tiny_tokens"], [item["name"] for item in shared_list_b.json()])

        template = self.client.post(
            f"/api/projects/{project_a_id}/sessions/{session_a_id}/templates/gpt/apply",
            json={"name": "tiny_gpt", "config": {"preset": "nanogpt", "n_layer": 1, "n_head": 1, "n_embd": 32}},
        )
        self.assertEqual(200, template.status_code, template.text)

        loaded = self.client.post(
            f"/api/projects/{project_a_id}/sessions/{session_a_id}/datasets/load",
            json={"dataset_names": ["tiny_tokens"], "seq_len": 8},
        )
        self.assertEqual(200, loaded.status_code, loaded.text)
        self.assertEqual(["tiny_tokens"], loaded.json()["result"]["dataset_names"])

        def fake_torch_train(_trainer, train_inputs, train_targets, *, on_epoch=None):
            self.assertTrue(train_inputs)
            self.assertTrue(train_targets)
            if on_epoch is not None:
                on_epoch(0, 0.25)
            return [0.25]

        with patch("server.services.run_service.TorchTrainer.train", new=fake_torch_train):
            started = self.client.post(
                f"/api/projects/{project_a_id}/sessions/{session_a_id}/runs",
                json={
                    "method": "torch",
                    "epochs": 1,
                    "batch_size": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                },
            )
        self.assertEqual(200, started.status_code, started.text)
        self.assertIn("\"done\": true", started.text)

        latest = self.client.get(f"/api/projects/{project_a_id}/sessions/{session_a_id}/runs/active")
        self.assertEqual(200, latest.status_code, latest.text)
        self.assertEqual(["tiny_tokens"], latest.json()["dataset_names"])
        self.assertEqual(8, latest.json()["seq_len"])


if __name__ == "__main__":
    unittest.main()
