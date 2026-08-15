"""Official OpenAI SDK checks over real loopback transports.

The broader SDK contract suite uses an in-process ASGI transport.  This file is
deliberately bounded: it covers direct Uvicorn HTTP/1.1, local-CA TLS, an HTTP
CONNECT proxy to an HTTPS origin, and a TLS/ALPN HTTP/2 reverse edge without
contacting an external service.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import ctypes
from datetime import datetime, timedelta, timezone
import http.client
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import select
import socket
import ssl
import threading
import time
from typing import Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict
import pytest
import uvicorn


_AUDITED_OPENAI_VERSION = "2.44.0"
openai = pytest.importorskip(
    "openai",
    minversion=_AUDITED_OPENAI_VERSION,
    reason="OpenAI SDK TCP compatibility requires the optional openai package",
)
if openai.__version__ != _AUDITED_OPENAI_VERSION:
    pytest.skip(
        "OpenAI SDK TCP compatibility was audited only against "
        f"openai=={_AUDITED_OPENAI_VERSION}; found {openai.__version__}",
        allow_module_level=True,
    )

from openai import (  # noqa: E402
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
)
from openai.types import Model  # noqa: E402
from openai.types.chat import ChatCompletion, ChatCompletionChunk  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ParsedResponse,
    ParsedResponseFunctionToolCall,
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
)

from neuralfn.native_inference import KVCacheConfig, NativeInferenceModel  # noqa: E402
from neuralfn.native_serve import (  # noqa: E402
    NativeServingRuntime,
    _PlainRolesRenderer,
    _TextCodec,
)

from test_native_serve import (  # noqa: E402
    BearerAuth,
    ConstrainedModel,
    FakeModel,
    FakeSession,
    _constrained_stateful_runtime,
    _runtime,
    _stateful_runtime,
    create_native_inference_app,
)


class StrictAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    city: str
    temperature_c: int


class LookupWeatherArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    city: str


class _TrackingBlockingModel(FakeModel):
    def __init__(self, *, release: threading.Event) -> None:
        super().__init__(release=release)
        self.sessions: list[FakeSession] = []

    def create_session(self) -> FakeSession:
        self.session_creates += 1
        session = FakeSession(self)
        self.sessions.append(session)
        return session


class _TinyResidentCodec(_TextCodec):
    name = "tiny-resident-cuda-test"

    def encode(self, _text: str) -> tuple[int, ...]:
        return (1, 2, 3)

    def decode(self, token_ids: Sequence[int]) -> str:
        return b"".join(self.token_bytes(token_id) for token_id in token_ids).decode(
            "ascii"
        )

    def token_bytes(self, token_id: int) -> bytes:
        return (b"A", b"B", b"C", b"D")[token_id]


class _LoopbackUvicorn:
    """Own one pre-bound loopback socket and one Uvicorn worker thread."""

    def __init__(
        self,
        app,
        *,
        tls_certfile: Path | None = None,
        tls_keyfile: Path | None = None,
    ) -> None:
        if (tls_certfile is None) != (tls_keyfile is None):
            raise ValueError("TLS certificate and key must be provided together")
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            lifespan="on",
            ssl_certfile=str(tls_certfile) if tls_certfile is not None else None,
            ssl_keyfile=str(tls_keyfile) if tls_keyfile is not None else None,
        )
        self._scheme = "https" if tls_certfile is not None else "http"
        self._socket = config.bind_socket()
        self._port = int(self._socket.getsockname()[1])
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._socket]},
            name="nfn-openai-sdk-loopback",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"{self._scheme}://127.0.0.1:{self._port}/v1"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("Uvicorn exited before accepting loopback requests")
            if time.monotonic() >= deadline:
                raise TimeoutError("Uvicorn did not accept loopback requests within 5s")
            time.sleep(0.01)

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=5.0)
        self._socket.close()
        if self._thread.is_alive():
            raise RuntimeError("Uvicorn worker did not stop cleanly")


def _write_loopback_tls_material(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a private CA and a loopback server certificate for one test."""

    pytest.importorskip(
        "cryptography",
        reason="local TLS transport tests require the cryptography package",
    )
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "NeuralFn loopback test CA")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]
    )
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_path = tmp_path / "loopback-ca.pem"
    cert_path = tmp_path / "loopback-server.pem"
    key_path = tmp_path / "loopback-server-key.pem"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return ca_path, cert_path, key_path


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _forward_to_loopback_upstream(
    upstream_base_url: str,
    *,
    method: str,
    path: str,
    headers: list[tuple[str, str]],
    body: bytes,
) -> tuple[int, list[tuple[str, str]], bytes]:
    upstream = urlsplit(upstream_base_url)
    if upstream.scheme != "http" or upstream.hostname != "127.0.0.1":
        raise ValueError("test transport may forward only to a loopback HTTP origin")
    if upstream.port is None:
        raise ValueError("loopback upstream must use an explicit port")

    forwarded_headers = {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS
        and name.lower() not in {"content-length", "host"}
    }
    forwarded_headers["Host"] = f"127.0.0.1:{upstream.port}"
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        upstream.port,
        timeout=5.0,
    )
    try:
        connection.request(
            method,
            path,
            body=body or None,
            headers=forwarded_headers,
        )
        response = connection.getresponse()
        response_body = response.read()
        response_headers = [
            (name.lower(), value)
            for name, value in response.getheaders()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower() != "content-length"
        ]
        response_headers.append(("content-length", str(len(response_body))))
        return response.status, response_headers, response_body
    finally:
        connection.close()


class _LoopbackHttpProxy:
    """A bounded HTTP CONNECT proxy for one local HTTPS origin."""

    def __init__(self, upstream_base_url: str) -> None:
        upstream = urlsplit(upstream_base_url)
        if upstream.scheme != "https" or upstream.hostname != "127.0.0.1":
            raise ValueError("proxy upstream must be a loopback HTTPS origin")
        if upstream.port is None:
            raise ValueError("proxy upstream must use an explicit port")
        self._upstream_authority = f"127.0.0.1:{upstream.port}"
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen()
        self._socket.settimeout(0.1)
        self._port = int(self._socket.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="nfn-openai-sdk-http-proxy",
            daemon=True,
        )
        self.request_methods: list[str] = []
        self.request_targets: list[str] = []
        self.errors: list[BaseException] = []

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("loopback HTTP proxy did not stop cleanly")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            with connection:
                try:
                    self._handle(connection)
                except Exception as exc:  # surfaced by the owning test
                    self.errors.append(exc)
                    try:
                        connection.sendall(
                            b"HTTP/1.1 502 Bad Gateway\r\n"
                            b"Connection: close\r\n"
                            b"Content-Length: 0\r\n\r\n"
                        )
                    except OSError:
                        pass

    def _handle(self, connection: socket.socket) -> None:
        connection.settimeout(5.0)
        request = bytearray()
        while b"\r\n\r\n" not in request:
            chunk = connection.recv(65536)
            if not chunk:
                raise ConnectionError("proxy client closed before sending headers")
            request.extend(chunk)
            if len(request) > 65536:
                raise ValueError("proxy request headers exceeded 64 KiB")

        raw_headers, body = bytes(request).split(b"\r\n\r\n", 1)
        header_lines = raw_headers.decode("iso-8859-1").split("\r\n")
        method, target, version = header_lines[0].split(" ", 2)
        if version != "HTTP/1.1":
            raise ValueError(f"unexpected proxy request version: {version}")
        self.request_methods.append(method)
        self.request_targets.append(target)
        if method != "CONNECT":
            raise ValueError(f"expected CONNECT proxy request, got {method}")
        if target != self._upstream_authority:
            raise ValueError(f"proxy refused non-loopback tunnel: {target}")
        self._tunnel(connection, bytes(body))

    def _tunnel(self, client: socket.socket, initial_data: bytes) -> None:
        upstream = socket.create_connection(
            ("127.0.0.1", int(self._upstream_authority.rsplit(":", 1)[1])),
            timeout=5.0,
        )
        with upstream:
            client.sendall(
                b"HTTP/1.1 200 Connection Established\r\n"
                b"Proxy-Agent: NeuralFn-loopback-test\r\n\r\n"
            )
            if initial_data:
                upstream.sendall(initial_data)
            client.settimeout(None)
            upstream.settimeout(None)
            peers = {client: upstream, upstream: client}
            while not self._stop.is_set():
                readable, _writable, _exceptional = select.select(
                    list(peers),
                    [],
                    [],
                    0.1,
                )
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    peers[source].sendall(data)


class _LoopbackH2ReverseProxy:
    """Terminate real TLS/ALPN HTTP/2 and forward requests to local Uvicorn."""

    def __init__(
        self,
        upstream_base_url: str,
        *,
        tls_certfile: Path,
        tls_keyfile: Path,
    ) -> None:
        self._upstream_base_url = upstream_base_url
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen()
        self._socket.settimeout(0.1)
        self._port = int(self._socket.getsockname()[1])
        self._tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        self._tls_context.load_cert_chain(tls_certfile, tls_keyfile)
        self._tls_context.set_alpn_protocols(["h2"])
        self._stop = threading.Event()
        self._active_connection: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="nfn-openai-sdk-h2-edge",
            daemon=True,
        )
        self.negotiated_protocols: list[str | None] = []
        self.request_headers: list[list[tuple[str, str]]] = []
        self.errors: list[BaseException] = []

    @property
    def base_url(self) -> str:
        return f"https://127.0.0.1:{self._port}/v1"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._socket.close()
        active_connection = self._active_connection
        if active_connection is not None:
            try:
                active_connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("loopback HTTP/2 edge did not stop cleanly")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw_connection, _address = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            try:
                with self._tls_context.wrap_socket(
                    raw_connection,
                    server_side=True,
                ) as tls_connection:
                    self._active_connection = tls_connection
                    self._handle(tls_connection)
            except Exception as exc:  # surfaced by the owning test
                if not self._stop.is_set():
                    self.errors.append(exc)
            finally:
                self._active_connection = None

    def _handle(self, connection: ssl.SSLSocket) -> None:
        from h2.config import H2Configuration
        from h2.connection import H2Connection
        from h2.events import (
            ConnectionTerminated,
            DataReceived,
            RequestReceived,
            StreamEnded,
        )

        negotiated = connection.selected_alpn_protocol()
        self.negotiated_protocols.append(negotiated)
        if negotiated != "h2":
            raise RuntimeError(f"HTTP/2 ALPN negotiation failed: {negotiated!r}")
        connection.settimeout(0.1)
        h2_connection = H2Connection(
            config=H2Configuration(client_side=False, header_encoding="utf-8")
        )
        h2_connection.initiate_connection()
        connection.sendall(h2_connection.data_to_send())
        streams: dict[int, dict[str, object]] = {}

        while not self._stop.is_set():
            try:
                incoming = connection.recv(65536)
            except TimeoutError:
                continue
            if not incoming:
                return
            for event in h2_connection.receive_data(incoming):
                if isinstance(event, RequestReceived):
                    headers = list(event.headers)
                    streams[event.stream_id] = {"headers": headers, "body": bytearray()}
                    self.request_headers.append(headers)
                elif isinstance(event, DataReceived):
                    stream = streams[event.stream_id]
                    stream_body = stream["body"]
                    assert isinstance(stream_body, bytearray)
                    stream_body.extend(event.data)
                    h2_connection.acknowledge_received_data(
                        event.flow_controlled_length,
                        event.stream_id,
                    )
                elif isinstance(event, StreamEnded):
                    self._respond(
                        h2_connection,
                        event.stream_id,
                        streams.pop(event.stream_id),
                    )
                elif isinstance(event, ConnectionTerminated):
                    return
            pending = h2_connection.data_to_send()
            if pending:
                connection.sendall(pending)

    def _respond(
        self,
        h2_connection,
        stream_id: int,
        stream: dict[str, object],
    ) -> None:
        raw_headers = stream["headers"]
        raw_body = stream["body"]
        assert isinstance(raw_headers, list)
        assert isinstance(raw_body, bytearray)
        pseudo_headers = {
            name: value for name, value in raw_headers if name.startswith(":")
        }
        ordinary_headers = [
            (name, value) for name, value in raw_headers if not name.startswith(":")
        ]
        method = pseudo_headers[":method"]
        path = pseudo_headers[":path"]
        status, response_headers, response_body = _forward_to_loopback_upstream(
            self._upstream_base_url,
            method=method,
            path=path,
            headers=ordinary_headers,
            body=bytes(raw_body),
        )
        h2_headers = [(":status", str(status)), *response_headers]
        h2_connection.send_headers(
            stream_id,
            h2_headers,
            end_stream=not response_body,
        )
        if response_body:
            frame_size = h2_connection.max_outbound_frame_size
            for offset in range(0, len(response_body), frame_size):
                chunk = response_body[offset : offset + frame_size]
                h2_connection.send_data(
                    stream_id,
                    chunk,
                    end_stream=offset + len(chunk) == len(response_body),
                )


def _sdk_client(base_url: str, *, api_key: str) -> tuple[OpenAI, httpx.Client]:
    http_client = httpx.Client(
        transport=httpx.HTTPTransport(retries=0),
        timeout=5.0,
        trust_env=False,
    )
    return (
        OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        ),
        http_client,
    )


def _async_sdk_client(
    base_url: str,
    *,
    api_key: str,
) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=0),
        timeout=5.0,
        trust_env=False,
    )
    return (
        AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        ),
        http_client,
    )


def _load_resident_binding(path: Path):
    spec = importlib.util.spec_from_file_location("_native_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create resident binding spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tcp_sdk_client(
    tmp_path,
) -> Iterator[tuple[OpenAI, ConstrainedModel, str]]:
    runtime = _constrained_stateful_runtime(tmp_path / "openai-sdk-tcp.sqlite")
    app = create_native_inference_app(
        runtime,
        auth=BearerAuth(("sdk-tcp-key",)),
        queue_capacity=1,
    )
    server = _LoopbackUvicorn(app)
    server.start()
    client, http_client = _sdk_client(server.base_url, api_key="sdk-tcp-key")
    try:
        yield client, runtime.model, server.base_url
    finally:
        client.close()
        http_client.close()
        server.close()
        assert runtime.model.model_closes == 1


def test_sdk_over_tls_rejects_untrusted_cert_and_accepts_explicit_ca(
    tmp_path,
) -> None:
    ca_path, cert_path, key_path = _write_loopback_tls_material(tmp_path)
    runtime = _constrained_stateful_runtime(tmp_path / "openai-sdk-tls.sqlite")
    app = create_native_inference_app(
        runtime,
        auth=BearerAuth(("sdk-tls-key",)),
        queue_capacity=1,
    )
    server = _LoopbackUvicorn(
        app,
        tls_certfile=cert_path,
        tls_keyfile=key_path,
    )
    server.start()
    untrusted_http_client = httpx.Client(timeout=5.0, trust_env=False)
    untrusted_client = OpenAI(
        api_key="sdk-tls-key",
        base_url=server.base_url,
        http_client=untrusted_http_client,
        max_retries=0,
    )
    trusted_versions: list[str] = []
    trusted_tls_context = ssl.create_default_context(cafile=str(ca_path))
    trusted_http_client = httpx.Client(
        verify=trusted_tls_context,
        timeout=5.0,
        trust_env=False,
        event_hooks={
            "response": [
                lambda response: trusted_versions.append(response.http_version)
            ]
        },
    )
    trusted_client = OpenAI(
        api_key="sdk-tls-key",
        base_url=server.base_url,
        http_client=trusted_http_client,
        max_retries=0,
    )
    try:
        with pytest.raises(APIConnectionError) as untrusted_error:
            untrusted_client.models.list()
        error_chain: list[str] = []
        current_error: BaseException | None = untrusted_error.value
        while current_error is not None:
            error_chain.append(repr(current_error))
            current_error = current_error.__cause__ or current_error.__context__
        assert "CERTIFICATE_VERIFY_FAILED" in " ".join(error_chain)

        models = trusted_client.models.list()
        assert isinstance(models.data[0], Model)
        assert models.data[0].id == "nfn-test"
        assert trusted_versions == ["HTTP/1.1"]
    finally:
        untrusted_client.close()
        untrusted_http_client.close()
        trusted_client.close()
        trusted_http_client.close()
        server.close()
    assert runtime.model.model_closes == 1


def test_sdk_routes_https_through_real_loopback_http_proxy(tmp_path) -> None:
    ca_path, cert_path, key_path = _write_loopback_tls_material(tmp_path)
    runtime = _constrained_stateful_runtime(tmp_path / "openai-sdk-proxy.sqlite")
    app = create_native_inference_app(
        runtime,
        auth=BearerAuth(("sdk-proxy-key",)),
        queue_capacity=1,
    )
    server = _LoopbackUvicorn(
        app,
        tls_certfile=cert_path,
        tls_keyfile=key_path,
    )
    server.start()
    proxy = _LoopbackHttpProxy(server.base_url)
    proxy.start()
    trusted_tls_context = ssl.create_default_context(cafile=str(ca_path))
    http_client = httpx.Client(
        proxy=proxy.proxy_url,
        verify=trusted_tls_context,
        timeout=5.0,
        trust_env=False,
    )
    client = OpenAI(
        api_key="sdk-proxy-key",
        base_url=server.base_url,
        http_client=http_client,
        max_retries=0,
    )
    try:
        models = client.models.list()
        assert isinstance(models.data[0], Model)
        assert models.data[0].id == "nfn-test"
    finally:
        client.close()
        http_client.close()
        proxy.close()
        server.close()

    assert proxy.errors == []
    server_authority = urlsplit(server.base_url).netloc
    assert proxy.request_methods == ["CONNECT"]
    assert proxy.request_targets == [server_authority]
    assert runtime.model.model_closes == 1


def test_sdk_negotiates_real_http2_at_loopback_tls_edge(tmp_path) -> None:
    pytest.importorskip(
        "h2",
        reason="the real HTTP/2 transport check requires HTTPX's h2 dependency",
    )
    ca_path, cert_path, key_path = _write_loopback_tls_material(tmp_path)
    runtime = _constrained_stateful_runtime(tmp_path / "openai-sdk-h2.sqlite")
    app = create_native_inference_app(
        runtime,
        auth=BearerAuth(("sdk-h2-key",)),
        queue_capacity=1,
    )
    upstream = _LoopbackUvicorn(app)
    upstream.start()
    edge = _LoopbackH2ReverseProxy(
        upstream.base_url,
        tls_certfile=cert_path,
        tls_keyfile=key_path,
    )
    edge.start()
    response_versions: list[str] = []
    trusted_tls_context = ssl.create_default_context(cafile=str(ca_path))
    http_client = httpx.Client(
        http1=False,
        http2=True,
        verify=trusted_tls_context,
        timeout=5.0,
        trust_env=False,
        event_hooks={
            "response": [
                lambda response: response_versions.append(response.http_version)
            ]
        },
    )
    client = OpenAI(
        api_key="sdk-h2-key",
        base_url=edge.base_url,
        http_client=http_client,
        max_retries=0,
    )
    try:
        models = client.models.list()
        assert isinstance(models.data[0], Model)
        assert models.data[0].id == "nfn-test"

        runtime.model.queue_text("HTTP/2 stream")
        events = list(
            client.responses.create(
                model="nfn-test",
                input="Stream over a real HTTP/2 transport edge.",
                max_output_tokens=13,
                temperature=0,
                stream=True,
                store=True,
            )
        )
        assert events[0].type == "response.created"
        assert "".join(
            event.delta
            for event in events
            if event.type == "response.output_text.delta"
        ) == "HTTP/2 stream"
        assert events[-1].type == "response.completed"
    finally:
        client.close()
        http_client.close()
        edge.close()
        upstream.close()

    assert edge.errors == []
    assert edge.negotiated_protocols == ["h2"]
    assert response_versions == ["HTTP/2", "HTTP/2"]
    assert len(edge.request_headers) == 2
    model_headers = dict(edge.request_headers[0])
    response_headers = dict(edge.request_headers[1])
    assert model_headers[":method"] == "GET"
    assert model_headers[":scheme"] == "https"
    assert model_headers[":path"] == "/v1/models"
    assert response_headers[":method"] == "POST"
    assert response_headers[":scheme"] == "https"
    assert response_headers[":path"] == "/v1/responses"
    assert runtime.model.model_closes == 1


def test_sdk_over_tcp_models_chat_stream_and_structured_response(
    tcp_sdk_client,
) -> None:
    client, model, _base_url = tcp_sdk_client

    models = client.models.list()
    assert isinstance(models.data[0], Model)
    assert models.data[0].id == "nfn-test"

    model.queue_text("Hello!")
    completion = client.chat.completions.create(
        model="nfn-test",
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=6,
        temperature=0,
    )
    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Hello!"

    model.queue_text("Stream!")
    chunks = list(
        client.chat.completions.create(
            model="nfn-test",
            messages=[{"role": "user", "content": "Stream"}],
            max_completion_tokens=7,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert chunks
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices
    ) == "Stream!"
    assert chunks[-1].usage is not None

    raw_output = '{"city":"London","temperature_c":12}'
    model.queue_constrained(raw_output)
    response = client.responses.parse(
        model="nfn-test",
        input="Return the weather as strict JSON.",
        text_format=StrictAnswer,
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )
    assert isinstance(response, ParsedResponse)
    assert response.output_text == raw_output
    assert isinstance(response.output[0], ParsedResponseOutputMessage)
    content = response.output[0].content[0]
    assert isinstance(content, ParsedResponseOutputText)
    assert isinstance(content.parsed, StrictAnswer)
    assert content.parsed == StrictAnswer(city="London", temperature_c=12)
    assert response.output_parsed is content.parsed


def test_sdk_over_tcp_maps_invalid_bearer_to_authentication_error(
    tcp_sdk_client,
) -> None:
    _client, _model, base_url = tcp_sdk_client
    bad_client, bad_http_client = _sdk_client(base_url, api_key="wrong-key")
    try:
        with pytest.raises(AuthenticationError) as error:
            bad_client.models.list()
        assert error.value.status_code == 401
        assert error.value.code == "invalid_api_key"
    finally:
        bad_client.close()
        bad_http_client.close()


def test_sdk_over_tcp_maps_non_auth_openai_error_classes(
    tcp_sdk_client,
) -> None:
    client, model, _base_url = tcp_sdk_client

    with pytest.raises(NotFoundError) as not_found:
        client.responses.retrieve("resp_missing")
    assert not_found.value.status_code == 404
    assert not_found.value.code == "response_not_found"

    with pytest.raises(BadRequestError) as bad_request:
        client.responses.create(
            model="nfn-test",
            input="reject unknown fields",
            extra_body={"unknown_option": True},
        )
    assert bad_request.value.status_code == 400
    assert bad_request.value.code == "unsupported_feature"
    assert model.session_creates == 0


def test_sdk_chat_stream_close_cancels_live_tcp_request() -> None:
    release = threading.Event()
    model = _TrackingBlockingModel(release=release)
    runtime = _runtime(model)
    app = create_native_inference_app(runtime, queue_capacity=0)
    server = _LoopbackUvicorn(app)
    server.start()
    client, http_client = _sdk_client(server.base_url, api_key="unused-loopback-key")
    stream = None
    try:
        stream = client.chat.completions.create(
            model="nfn-test",
            messages=[{"role": "user", "content": "Cancel me"}],
            max_completion_tokens=2,
            temperature=0,
            stream=True,
        )
        first = next(stream)
        assert isinstance(first, ChatCompletionChunk)
        assert model.started.wait(timeout=1.0)

        stream.close()
        stream = None
        # The server polls the real socket every 100 ms while native decode is
        # blocked. Give that poll time to set the cooperative cancel signal
        # before allowing the scripted session to resume.
        time.sleep(0.3)
        release.set()
        deadline = time.monotonic() + 2.0
        while model.session_closes != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert model.session_creates == model.session_closes == 1
        assert model.sessions[0].cancelled is True
    finally:
        release.set()
        if stream is not None:
            stream.close()
        client.close()
        http_client.close()
        server.close()
    assert model.model_closes == 1


def test_sdk_background_stream_close_continues_and_retrieves_over_tcp(
    tmp_path,
) -> None:
    release = threading.Event()
    model = _TrackingBlockingModel(release=release)
    runtime = _stateful_runtime(tmp_path / "tcp-background.sqlite", model=model)
    app = create_native_inference_app(runtime, queue_capacity=1)
    server = _LoopbackUvicorn(app)
    server.start()
    client, http_client = _sdk_client(server.base_url, api_key="unused-loopback-key")
    stream = None
    try:
        stream = client.responses.create(
            model="nfn-test",
            input="Continue after the TCP client disconnects.",
            max_output_tokens=2,
            temperature=0,
            background=True,
            stream=True,
            store=True,
        )
        created = next(stream)
        assert created.type == "response.created"
        response_id = created.response.id
        assert model.started.wait(timeout=1.0)

        stream.close()
        stream = None
        time.sleep(0.3)
        release.set()

        deadline = time.monotonic() + 2.0
        terminal = None
        while time.monotonic() < deadline:
            terminal = client.responses.retrieve(response_id)
            if terminal.status in {"completed", "failed", "incomplete"}:
                break
            time.sleep(0.01)
        assert terminal is not None
        assert terminal.status == "completed"
        assert terminal.output_text == "Hello!"
        assert model.session_creates == model.session_closes == 1
        assert model.sessions[0].cancelled is False
    finally:
        release.set()
        if stream is not None:
            stream.close()
        client.close()
        http_client.close()
        server.close()
    assert model.model_closes == 1


def test_sdk_over_tcp_parses_semantic_responses_stream_in_order(
    tcp_sdk_client,
) -> None:
    client, model, _base_url = tcp_sdk_client
    model.queue_text("TCP stream")

    events = list(
        client.responses.create(
            model="nfn-test",
            input="Stream over TCP.",
            max_output_tokens=10,
            temperature=0,
            stream=True,
            store=True,
        )
    )

    assert events[0].type == "response.created"
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert "".join(
        event.delta
        for event in events
        if event.type == "response.output_text.delta"
    ) == "TCP stream"
    terminal_events = [
        event
        for event in events
        if event.type
        in {"response.completed", "response.failed", "response.incomplete"}
    ]
    assert terminal_events == [events[-1]]
    assert events[-1].type == "response.completed"
    assert events[-1].response.status == "completed"
    assert events[-1].response.output_text == "TCP stream"


def test_async_sdk_over_tcp_parses_semantic_responses_stream(
    tcp_sdk_client,
) -> None:
    _client, model, base_url = tcp_sdk_client

    async def scenario() -> None:
        client, http_client = _async_sdk_client(
            base_url,
            api_key="sdk-tcp-key",
        )
        try:
            models = await client.models.list()
            assert models.data[0].id == "nfn-test"

            model.queue_text("Async TCP")
            stream = await client.responses.create(
                model="nfn-test",
                input="Stream asynchronously over TCP.",
                max_output_tokens=9,
                temperature=0,
                stream=True,
                store=True,
            )
            events = [event async for event in stream]
            assert events[0].type == "response.created"
            assert [event.sequence_number for event in events] == list(
                range(len(events))
            )
            assert "".join(
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ) == "Async TCP"
            assert events[-1].type == "response.completed"
        finally:
            await client.close()
            await http_client.aclose()

    asyncio.run(scenario())


def test_sdk_over_tcp_resumes_stored_background_stream_after_cursor(
    tcp_sdk_client,
) -> None:
    client, model, _base_url = tcp_sdk_client
    model.queue_text("Background TCP")

    events = list(
        client.responses.create(
            model="nfn-test",
            input="Run in the background over TCP.",
            max_output_tokens=14,
            temperature=0,
            background=True,
            stream=True,
            store=True,
        )
    )
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    response_id = events[0].response.id
    cursor = next(
        event.sequence_number
        for event in events
        if event.type == "response.output_text.delta"
    )

    resumed = list(
        client.responses.retrieve(
            response_id,
            stream=True,
            starting_after=cursor,
            include_obfuscation=False,
        )
    )
    assert resumed
    assert all(event.sequence_number > cursor for event in resumed)
    assert [event.sequence_number for event in resumed] == list(
        range(cursor + 1, len(events))
    )
    assert resumed[-1].type == "response.completed"


def test_sdk_over_tcp_forced_function_call_and_client_result_continuation(
    tcp_sdk_client,
) -> None:
    client, model, _base_url = tcp_sdk_client
    raw_arguments = '{"city":"London"}'
    model.queue_constrained(raw_arguments)

    response = client.responses.parse(
        model="nfn-test",
        input="Look up the weather in London.",
        tools=[
            openai.pydantic_function_tool(
                LookupWeatherArguments,
                name="lookup_weather",
            )
        ],
        tool_choice={"type": "function", "name": "lookup_weather"},
        parallel_tool_calls=False,
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )

    assert isinstance(response, ParsedResponse)
    assert response.status == "completed"
    assert len(response.output) == 1
    call = response.output[0]
    assert isinstance(call, ParsedResponseFunctionToolCall)
    assert call.type == "function_call"
    assert call.status == "completed"
    assert call.name == "lookup_weather"
    assert isinstance(call.call_id, str) and call.call_id
    assert call.arguments == raw_arguments
    assert isinstance(call.parsed_arguments, LookupWeatherArguments)
    assert call.parsed_arguments.city == "London"
    assert model.function_executions == 0

    def lookup_weather(arguments: LookupWeatherArguments) -> dict[str, int]:
        assert arguments.city == "London"
        model.function_executions += 1
        return {"temperature_c": 12}

    function_result = lookup_weather(call.parsed_arguments)
    assert model.function_executions == 1

    final_text = "Weather: 12 C"
    model.queue_text(final_text)
    final_response = client.responses.create(
        model="nfn-test",
        previous_response_id=response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(function_result, separators=(",", ":")),
            }
        ],
        max_output_tokens=64,
        temperature=0,
        top_p=1,
        store=True,
    )

    assert isinstance(final_response, Response)
    assert final_response.status == "completed"
    assert final_response.previous_response_id == response.id
    assert final_response.tool_choice == "none"
    assert final_response.tools == []
    assert len(final_response.output) == 1
    final_message = final_response.output[0]
    assert isinstance(final_message, ResponseOutputMessage)
    assert final_message.type == "message"
    assert final_message.status == "completed"
    assert len(final_message.content) == 1
    final_content = final_message.content[0]
    assert isinstance(final_content, ResponseOutputText)
    assert final_content.type == "output_text"
    assert final_content.text == final_text
    assert final_response.output_text == final_text
    assert model.function_executions == 1
    assert model.session_creates == 2
    assert model.current_logits_calls > 0
    assert model.decode_calls == 1


@pytest.mark.skipif(
    os.environ.get("NFN_NATIVE_OPENAI_SDK_CUDA_TEST") != "1",
    reason="Set NFN_NATIVE_OPENAI_SDK_CUDA_TEST=1 for the live resident Tile-CUDA test",
)
def test_sdk_over_tcp_uses_real_resident_tile_cuda_attention() -> None:
    required_paths = {
        name: os.environ.get(name)
        for name in (
            "NFN_NATIVE_OPENAI_SDK_ARTIFACT",
            "NFN_NATIVE_OPENAI_SDK_BINDING_LIB",
            "NFN_NATIVE_OPENAI_SDK_TILE_OPS_LIB",
        )
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        pytest.fail("Missing live CUDA test paths: " + ", ".join(missing))
    artifact = Path(required_paths["NFN_NATIVE_OPENAI_SDK_ARTIFACT"] or "").resolve()
    binding_path = Path(required_paths["NFN_NATIVE_OPENAI_SDK_BINDING_LIB"] or "").resolve()
    tile_ops_path = Path(required_paths["NFN_NATIVE_OPENAI_SDK_TILE_OPS_LIB"] or "").resolve()
    for label, path in (
        ("artifact", artifact),
        ("binding", binding_path),
        ("Tile sidecar", tile_ops_path),
    ):
        if not path.exists():
            pytest.fail(f"Live CUDA {label} does not exist: {path}")
    cuda_runtime = os.environ.get(
        "NFN_NATIVE_OPENAI_SDK_CUDA_RUNTIME_LIB",
        "libcudart.so.13",
    )
    raw_device = os.environ.get("NFN_NATIVE_OPENAI_SDK_CUDA_DEVICE", "0")
    try:
        cuda_device = int(raw_device)
    except ValueError as exc:
        raise AssertionError("NFN_NATIVE_OPENAI_SDK_CUDA_DEVICE must be an integer") from exc

    sidecar = ctypes.CDLL(str(tile_ops_path))
    reset_launches = sidecar.nfn_native_tile_turboquant_attention_stats_reset
    reset_launches.argtypes = []
    reset_launches.restype = None
    launch_count = sidecar.nfn_native_tile_turboquant_attention_launch_count
    launch_count.argtypes = []
    launch_count.restype = ctypes.c_int64
    binding = _load_resident_binding(binding_path)
    model = NativeInferenceModel.load(
        artifact,
        binding=binding,
        kv_cache=KVCacheConfig(
            mode="turboquant",
            turboquant_profile="mse-3.5",
            turboquant_attention_backend="tile-cuda",
            tile_ops_lib=str(tile_ops_path),
            cuda_runtime_lib=cuda_runtime,
            cuda_device=cuda_device,
        ),
    )
    manifest = json.loads((artifact / "native-execution-manifest.json").read_text())
    runtime = NativeServingRuntime(
        model=model,
        manifest=manifest,
        codec=_TinyResidentCodec(),
        renderer=_PlainRolesRenderer(),
        served_model_name="nfn-live-cuda",
        context_limit=int(manifest["context_limits"]["max_context_tokens"]),
        max_output_tokens=1,
        created=1_700_000_000,
        chat_template_selection="plain_roles",
    )
    app = create_native_inference_app(runtime, queue_capacity=0)
    server = _LoopbackUvicorn(app)
    server.start()
    client, http_client = _sdk_client(server.base_url, api_key="unused-loopback-key")
    try:
        reset_launches()
        completion = client.chat.completions.create(
            model="nfn-live-cuda",
            messages=[{"role": "user", "content": "Run the real resident model."}],
            max_completion_tokens=1,
            temperature=0,
        )
        assert isinstance(completion, ChatCompletion)
        assert completion.choices[0].message.content in {"A", "B", "C", "D"}
        assert launch_count() > 0
    finally:
        client.close()
        http_client.close()
        server.close()
    assert model.closed is True
