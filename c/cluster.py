#!/usr/bin/env python3
"""Registration and discovery control plane for local expert workers."""

import argparse
import hmac
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import openai_server


PROTOCOL_VERSION = 1


class ClusterRegistry:
    def __init__(self, stale_after=30.0):
        self.stale_after = float(stale_after)
        self._nodes = {}
        self._lock = threading.Lock()

    def _purge_stale(self):
        """Drop nodes not seen within stale_after (caller holds the lock)."""
        now = time.time()
        for key in [key for key, node in self._nodes.items()
                    if now - node["last_seen"] > self.stale_after]:
            del self._nodes[key]

    def register(self, node, src_ip=None):
        required = {"node_id", "host", "port", "role"}
        missing = sorted(required - set(node))
        if missing:
            raise ValueError("missing node fields: " + ", ".join(missing))
        port = int(node["port"])
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        role = str(node["role"])
        if role not in ("expert", "dense", "coordinator"):
            raise ValueError("role must be expert, dense, or coordinator")
        record = dict(node)
        record.update(protocol_version=PROTOCOL_VERSION, port=port,
                      last_seen=time.time(), src_ip=src_ip or "")
        with self._lock:
            self._purge_stale()
            existing = self._nodes.get(str(node["node_id"]))
            # Rebind guard: a live node_id may not be re-registered from a
            # different source address. Without this, anyone who can reach the
            # coordinator could shadow or hijack a live worker's registration.
            if existing and src_ip and existing.get("src_ip") not in (None, src_ip):
                raise ValueError("node_id already registered from a different address")
            self._nodes[str(node["node_id"])] = record
        return record

    def heartbeat(self, node_id):
        with self._lock:
            self._purge_stale()
            node = self._nodes.get(str(node_id))
            if node is None:
                raise KeyError(node_id)
            node["last_seen"] = time.time()
            return dict(node)

    def snapshot(self):
        now = time.time()
        with self._lock:
            nodes = [dict(node) for node in self._nodes.values()
                     if now - node["last_seen"] <= self.stale_after]
        nodes.sort(key=lambda node: (node["role"], node["node_id"]))
        return {"protocol_version": PROTOCOL_VERSION, "nodes": nodes}

    def expert_endpoints(self):
        return [f"{node['host']}:{node['port']}"
                for node in self.snapshot()["nodes"] if node["role"] == "expert"]


class _Handler(openai_server.APIHandler):
    # The original registry served HTTP/1.0: one request per connection, no
    # keep-alive. Keep that wire shape; the shared APIHandler machinery (Host
    # guard, body cap, deadline reader, connection bookkeeping) still applies.
    protocol_version = "HTTP/1.0"
    server_version = "colibri-cluster/1"

    def log_message(self, *_args):
        return

    def _fail(self, error):
        self.send_json(error.status, {"error": error.message})

    def _check_token(self):
        token = self.server.cluster_token
        if not token:
            return
        presented = self.headers.get("X-Cluster-Token", "")
        if not hmac.compare_digest(presented, token):
            raise openai_server.APIError(401, "invalid or missing X-Cluster-Token header")

    def do_GET(self):  # noqa: N802 - stdlib handler API
        try:
            self._check_host()
            # /health stays public (plain liveness probe); the topology
            # carries LAN node addresses, so it sits behind the token check.
            if self.path == "/v1/cluster/topology":
                self._check_token()
        except openai_server.APIError as error:
            self._fail(error)
            return
        if self.path in ("/health", "/v1/cluster/topology"):
            self.send_json(200, self.server.registry.snapshot())
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802 - stdlib handler API
        try:
            self._check_host()
            self._check_token()
            body = self.read_json()
            if self.path == "/v1/cluster/register":
                self.send_json(200, self.server.registry.register(
                    body, src_ip=self.client_address[0] if self.client_address else None))
            elif self.path == "/v1/cluster/heartbeat":
                self.send_json(200, self.server.registry.heartbeat(body["node_id"]))
            else:
                self.send_json(404, {"error": "not found"})
        except openai_server.APIError as error:
            self._fail(error)
        except (KeyError, ValueError, TypeError) as error:
            self.send_json(400, {"error": str(error)})


class ClusterServer(openai_server.APIServer):
    def __init__(self, address, registry, allowed_hosts=(), token=""):
        ThreadingHTTPServer.__init__(self, address, _Handler)
        self.registry = registry
        # Shared APIHandler plumbing reads these off the server. The registry
        # has no per-user API key or CORS surface, but it takes an optional
        # shared cluster token (--token / $COLI_CLUSTER_TOKEN) guarding the
        # topology/register/heartbeat routes. The Host guard stays loopback +
        # bind address (the same default openai_server enforces). A cross-host
        # worker registers from a LAN IP, so the operator opts those hosts in
        # via --allowed-host (the same #597 escape hatch coli serve exposes);
        # the default stays loopback + bind address.
        self.cluster_token = token or ""
        self.cors_origins = ()
        self.allowed_hosts = tuple(
            h.strip().lower() for h in allowed_hosts if h and h.strip())
        # Connection-cap bookkeeping for the inherited APIServer.process_request /
        # _release / close_request. There is no engine or model to carry, so
        # initialise the tracking state directly rather than APIServer.__init__.
        self._conn_lock = threading.Lock()
        self._conn_live = 0
        self._conn_by_ip = {}
        self._conn_owner = {}


def _endpoint(coordinator, path):
    return coordinator.rstrip("/") + "/v1/cluster/" + path


def _token_headers():
    """Auth headers for coordinator calls: X-Cluster-Token when set."""
    token = os.environ.get("COLI_CLUSTER_TOKEN")
    return {"X-Cluster-Token": token} if token else {}


def serve(host="127.0.0.1", port=8765, stale_after=30.0, allowed_hosts=(), token=None):
    if allowed_hosts and "*" in allowed_hosts:
        print("WARNING: --allowed-host '*' accepts ANY Host header "
              "(DNS-rebinding guard disabled)", file=sys.stderr)
    if token is None:
        token = os.environ.get("COLI_CLUSTER_TOKEN") or ""
    if not token:
        print("WARNING: no cluster token set; topology/register/heartbeat are "
              "unauthenticated (safe only on a trusted network). Set --token or "
              "COLI_CLUSTER_TOKEN to require the X-Cluster-Token header.",
              file=sys.stderr)
    server = ClusterServer((host, port), ClusterRegistry(stale_after), allowed_hosts, token)
    print(f"colibri cluster coordinator listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def register(coordinator, node):
    request = Request(_endpoint(coordinator, "register"),
                      data=json.dumps(node).encode(),
                      headers={"Content-Type": "application/json", **_token_headers()},
                      method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def heartbeat(coordinator, node_id):
    request = Request(_endpoint(coordinator, "heartbeat"),
                      data=json.dumps({"node_id": node_id}).encode(),
                      headers={"Content-Type": "application/json", **_token_headers()},
                      method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def discover_workers(coordinator):
    request = Request(_endpoint(coordinator, "topology"), headers=_token_headers())
    with urlopen(request, timeout=5) as response:
        topology = json.load(response)
    return [f"{node['host']}:{int(node['port'])}"
            for node in topology.get("nodes", []) if node.get("role") == "expert"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=30.0)
    parser.add_argument("--allowed-host", action="append",
                        default=[h.strip() for h in os.environ.get("COLI_ALLOWED_HOSTS", "").split(",") if h.strip()],
                        help="additional Host header accepted by the DNS-rebinding guard; repeat as needed")
    parser.add_argument("--token", default=os.environ.get("COLI_CLUSTER_TOKEN"),
                        help="shared secret required by the topology/register/heartbeat routes "
                             "(X-Cluster-Token header); defaults to $COLI_CLUSTER_TOKEN, "
                             "empty disables authentication")
    args = parser.parse_args()
    serve(args.host, args.port, args.stale_after, args.allowed_host, args.token)


if __name__ == "__main__":
    main()
