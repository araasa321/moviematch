"""DNS/reachability check for candidate embedding oracles."""

import socket
import urllib.request

HOSTS = [
    "huggingface.co",
    "router.huggingface.co",
    "api-inference.huggingface.co",
    "hf-mirror.com",
    "integrate.api.nvidia.com",
    "generativelanguage.googleapis.com",
]


def main() -> None:
    for host in HOSTS:
        try:
            ip = socket.gethostbyname(host)
        except Exception as exc:  # noqa: BLE001
            print(f"  {host:40s} DNS FAIL ({exc})")
            continue
        status = ""
        try:
            with urllib.request.urlopen(f"https://{host}/", timeout=10) as resp:
                status = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            status = f"{type(exc).__name__}"
        print(f"  {host:40s} {ip:16s} {status}")


if __name__ == "__main__":
    main()
