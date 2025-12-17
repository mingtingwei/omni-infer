import os
import sys
import subprocess
from pathlib import Path
import port_manager

CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../../../omni/accelerators/sched/omni_proxy/omni_proxy.sh"


def generate_proxy_endpoints(port_list) -> str:
    return ",".join(f"127.0.0.1:{port}" for port in port_list)

def setup_proxy(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'

    prefill_list = generate_proxy_endpoints(prefill_port_list)
    decode_list = generate_proxy_endpoints(decode_port_list)
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx.conf",
            "--core-num", "1",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access.log",
            "--stream-ops", "add",
            "--omni-proxy-model-path", f"{CUR_DIR}/mock_model",
            "--no-reuseport"
        ]
        print(f"\n[SETUP] Starting proxy with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
        return 0
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        return 1

def teardown_proxy():
    try:
        cmd = [
            "bash", proxy_script_path,
            "--stop",
        ]
        print(f"\n[TEARDOWN] Stopping proxy with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[TEARDOWN] Script succeeded. Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Teardown script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    elif len(args) == 2:
        try:
            numeric_args = []
            for arg in args:
                numeric_args.append(int(arg))
            ports = port_manager.load_ports(*numeric_args)
            proxy_port = ports["proxy_port"]
            prefill_port_list = ports["prefill"]
            decode_port_list = ports["decode"]
            setup_proxy(proxy_port, prefill_port_list, decode_port_list)
        except ValueError as e:
            print(f"Error: All four arguments must be valid numbers. Got: {args}")
            print("Usage: python script.py <prefill_num> <decode_num>")
            sys.exit(1)
    elif len(args) == 1 and args[0] == "stop":
        teardown_proxy()
    else:
        print(f"Error: Invalid arguments: {args}")
        print("Usage:")
        print("  python run_proxy.py <prefill_num> <decode_num>   # Start with 2 numbers")
        print("  python run_proxy.py stop                         # Stop/cleanup")
        sys.exit(1)
