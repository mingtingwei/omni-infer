import json
import os
import socket

PORTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_ports.json")

def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def find_n_free_ports(num_ports):
    ports = []
    for _ in range(num_ports):
        port = find_free_port()
        ports.append(port)
    return ports

def find_free_port_excluding_existing():
    if not os.path.exists(PORTS_FILE):
        return find_free_port()
    
    try:
        with open(PORTS_FILE, "r") as f:
            ports_config = json.load(f)
        
        exclude_ports = []
        if "prefill" in ports_config:
            exclude_ports.extend(ports_config["prefill"])
        if "decode" in ports_config:
            exclude_ports.extend(ports_config["decode"])
        if "proxy_port" in ports_config:
            exclude_ports.extend(ports_config["proxy_port"])

        exclude_set = set(exclude_ports)
        max_attempts = 100
        
        for _ in range(max_attempts):
            port = find_free_port()
            if port not in exclude_set:
                return port
        
        raise RuntimeError(f"cannot find port in {max_attempts} times")
    
    except (json.JSONDecodeError, KeyError, TypeError):
        return find_free_port()

def ensure_ports_file(prefill_num=1, decode_num=1):
    if os.path.exists(PORTS_FILE):
        try:
            with open(PORTS_FILE, "r") as f:
                ports = json.load(f)
            
            if (isinstance(ports, dict) and 
                "proxy_port" in ports and
                "prefill" in ports and "decode" in ports and
                len(ports["prefill"]) == prefill_num and
                len(ports["decode"]) == decode_num):
                return ports
            print(f"port num does not match in {PORTS_FILE}, regenerating...")

        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"port conf file {PORTS_FILE} format invalid, regenerating...")
    
    proxy_port = find_free_port()
    prefill_ports = find_n_free_ports(prefill_num)
    decode_ports = find_n_free_ports(decode_num)
    
    ports = {
        "proxy_port": proxy_port,
        "prefill": prefill_ports,
        "decode": decode_ports
    }
    print(f"generate {ports=} and store in {PORTS_FILE}")

    temp_file = PORTS_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(ports, f)
    os.replace(temp_file, PORTS_FILE)
    
    return ports

def load_ports(prefill_num=1, decode_num=1):
    return ensure_ports_file(prefill_num, decode_num)

def get_ports_from_file():
    if not os.path.exists(PORTS_FILE):
        raise FileNotFoundError(f"Port configuration file does not exist: {PORTS_FILE}")
    
    try:
        with open(PORTS_FILE, "r") as f:
            ports = json.load(f)
        
        required_fields = ["proxy_port", "prefill", "decode"]
        for field in required_fields:
            if field not in ports:
                raise KeyError(f"Port configuration file is missing required field: {field}")
        
        return ports
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Port configuration file has invalid format (JSON parsing error): {e}")
    except KeyError as e:
        raise KeyError(f"Port configuration file is missing required field: {e}")
    except Exception as e:
        raise RuntimeError(f"Unknown error occurred while reading port configuration file: {e}")
