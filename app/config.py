import configparser
from pathlib import Path

config = configparser.ConfigParser()
config_path = Path(__file__).parent.parent / "config.ini"
files_read = config.read(config_path)

# Dataset
dataset_name = config.get("dataset", "dataset_name")
data_dir = config.get("dataset", "save_dir")

# Model
model_dir = config.get("model", "model_dir")

# LLM
use_llm = config.getboolean("llm", "use_llm")
llm_model_name = config.get("llm", "llm_model_name")
device = config.get("llm", "device")
low_cpu_mem_usage = config.getboolean("llm", "low_cpu_mem_usage")

# Streamlit
page_title = config.get("streamlit", "page_title")
server_port = config.getint("streamlit", "server_port")
server_address = config.get("streamlit", "server_address")
