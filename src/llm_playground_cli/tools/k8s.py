import json
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from llm_playground_cli.tools.base import Tool, ToolResult

# Initialize Kubernetes client
try:
    # Works for local development with ~/.kube/config
    config.load_kube_config()
except Exception:
    try:
        # Works when running inside a Kubernetes cluster
        config.load_incluster_config()
    except Exception:
        pass # Handle gracefully in tools if uninitialized

def get_core_v1():
    return client.CoreV1Api()

def get_apps_v1():
    return client.AppsV1Api()

# ── Tool Functions ────────────────────────────────────────────

def _get_pod_status(namespace: str = "default") -> ToolResult:
    """Get status of all pods in a namespace."""
    try:
        core_v1 = get_core_v1()
        pods_list = core_v1.list_namespaced_pod(namespace=namespace)
        pods = []

        for pod in pods_list.items:
            container_statuses = pod.status.container_statuses or []
            restart_count = sum(cs.restart_count for cs in container_statuses)
            ready = all(cs.ready for cs in container_statuses) if container_statuses else False

            pods.append({
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "ready": ready,
                "restart_count": restart_count,
                "node": pod.spec.node_name or "Unscheduled",
            })

        summary = json.dumps(pods, indent=2)
        return ToolResult(success=True, data=f"Pods in namespace '{namespace}':\n{summary}")

    except ApiException as e:
        return ToolResult(success=False, data="", error=f"Kubernetes API error: {e.reason} ({e.status})")
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))


def _describe_deployment(deployment_name: str, namespace: str = "default") -> ToolResult:
    """Get detailed info about a Kubernetes deployment."""
    try:
        apps_v1 = get_apps_v1()
        deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)

        spec = deployment.spec
        status = deployment.status

        summary = {
            "name": deployment.metadata.name,
            "namespace": deployment.metadata.namespace,
            "replicas": {
                "desired": spec.replicas or 0,
                "ready": status.ready_replicas or 0,
                "available": status.available_replicas or 0,
            },
            "strategy": spec.strategy.type if spec.strategy else None,
            "containers": [
                {
                    "name": c.name,
                    "image": c.image,
                    "ports": [p.container_port for p in (c.ports or [])],
                    "resources": c.resources.to_dict() if c.resources else {},
                }
                for c in spec.template.spec.containers
            ],
        }

        return ToolResult(success=True, data=json.dumps(summary, indent=2))

    except ApiException as e:
        return ToolResult(success=False, data="", error=f"Kubernetes API error: {e.reason} ({e.status})")
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))


def _get_logs(pod_name: str, namespace: str = "default", lines: int = 50) -> ToolResult:
    """Get recent logs from a pod (by name prefix)."""
    try:
        core_v1 = get_core_v1()
        
        # First, find the exact pod name matching the prefix
        pods_list = core_v1.list_namespaced_pod(namespace=namespace)
        matching = [p for p in pods_list.items if p.metadata.name.startswith(pod_name)]

        if not matching:
            return ToolResult(
                success=False, data="",
                error=f"No pods found matching '{pod_name}' in namespace '{namespace}'",
            )

        actual_pod_name = matching[0].metadata.name

        # Get logs
        logs = core_v1.read_namespaced_pod_log(
            name=actual_pod_name,
            namespace=namespace,
            tail_lines=lines
        )

        logs = logs or "(no logs available)"

        # Truncate if too long
        if len(logs) > 5000:
            logs = logs[-5000:]
            logs = "[... truncated, showing last 5000 chars ...]\n" + logs

        return ToolResult(
            success=True,
            data=f"Logs from pod '{actual_pod_name}' (last {lines} lines):\n\n{logs}",
        )

    except ApiException as e:
        return ToolResult(success=False, data="", error=f"Kubernetes API error: {e.reason} ({e.status})")
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))

# ── Tool Definitions ─────────────────────────────────────────
get_pod_status_tool = Tool(
    name="get_pod_status",
    description=(
        "Get the status of all pods in a Kubernetes namespace including "
        "phase (Running/Pending/Failed), readiness, restart count, and node. "
        "Use this when the user asks about pod health, whether pods are running, "
        "or to check the state of a namespace. "
        "Do NOT use this for deployment configuration — use describe_deployment. "
        "Do NOT use this for pod logs — use get_logs."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "namespace": {
                "type": "STRING",
                "description": (
                    "The Kubernetes namespace to check. "
                    "Defaults to 'default'. Common values: 'default', 'production', 'kube-system'."
                ),
            },
        },
        "required": [],
    },
    function=_get_pod_status,
)
describe_deployment_tool = Tool(
    name="describe_deployment",
    description=(
        "Get detailed information about a Kubernetes deployment including "
        "replica counts (desired/ready/available), deployment strategy, "
        "container images, ports, and resource requests/limits. "
        "Use this when the user asks about a deployment's configuration, "
        "replica count, or container specs. "
        "Do NOT use this for pod-level status — use get_pod_status. "
        "Do NOT use this for logs — use get_logs."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "deployment_name": {
                "type": "STRING",
                "description": "The exact name of the Kubernetes deployment.",
            },
            "namespace": {
                "type": "STRING",
                "description": "The Kubernetes namespace. Defaults to 'default'.",
            },
        },
        "required": ["deployment_name"],
    },
    function=_describe_deployment,
)
get_logs_tool = Tool(
    name="get_logs",
    description=(
        "Retrieve recent stdout/stderr logs from a Kubernetes pod. "
        "Use this when the user asks to see logs, debug errors, or investigate "
        "what happened in a pod. Pass a pod name or name prefix. "
        "Do NOT use this for pod status — use get_pod_status. "
        "Do NOT use this for deployment info — use describe_deployment."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "pod_name": {
                "type": "STRING",
                "description": (
                    "Name or name prefix of the pod. "
                    "E.g. 'nginx' matches the first pod starting with 'nginx'."
                ),
            },
            "namespace": {
                "type": "STRING",
                "description": "The Kubernetes namespace. Defaults to 'default'.",
            },
            "lines": {
                "type": "INTEGER",
                "description": "Number of recent log lines to retrieve. Defaults to 50.",
            },
        },
        "required": ["pod_name"],
    },
    function=_get_logs,
)
# ── Convenience list for registration ─────────────────────────
ALL_K8S_TOOLS = [get_pod_status_tool, describe_deployment_tool, get_logs_tool]