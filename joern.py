import subprocess
import networkx as nx
import xml.etree.ElementTree as ET
import os
from enums.enums import settting
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch
from gensim.models import Word2Vec

def run_joern_analysis(c_file_path, output_dir):
    file_name = os.path.basename(c_file_path)
    export_dir = os.path.join(output_dir, file_name)
    print("file_name ",file_name)
    print("export_dir ",export_dir)
    if os.path.isdir(export_dir):
        print(f"Directory {export_dir} already exists.")
        return None

    subprocess.run(f'joern-parse "{os.path.abspath(c_file_path)}"', shell=True, check=True)
    subprocess.run(f'joern-export --repr=all --format=graphml --out "{output_dir}/{file_name}"', shell=True, check=True)


    file_path = os.path.join(export_dir, "export.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionary lưu thông tin node & mapping
    node_features = {}
    node_mapping = {}

    # Lấy danh sách node
    for idx, node in enumerate(root.findall("ns:graph/ns:node", settting.NAMESPACE)):
        node_id = node.get("id")
        node_data = {}

        # Lấy thuộc tính của node
        for data in node.findall("ns:data", settting.NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            node_data[key] = value

        # Lưu node vào dictionary
        node_features[node_id] = node_data
        node_mapping[node_id] = idx  # Đánh số lại cho node

    # Xử lý cạnh (edges) & thuộc tính cạnh
    edges = []
    edge_features = []

    for edge in root.findall("ns:graph/ns:edge", settting.NAMESPACE):
        src = edge.get("source")
        tgt = edge.get("target")
        edge_data = {}

        # Lấy thuộc tính của cạnh
        for data in edge.findall("ns:data", settting.NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            edge_data[key] = value

        if src in node_mapping and tgt in node_mapping:
            edges.append((node_mapping[src], node_mapping[tgt]))
            edge_features.append(edge_data)

    # Chuyển danh sách edges thành tensor
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        print(f"Lỗi: Không có cạnh trong {output_dir}")
        return None  # Bỏ qua nếu không có cạnh

    # Sử dụng Word2Vec để nhúng đặc trưng của node
    node_sentences = [list(data.values()) for data in node_features.values()]
    node_model = Word2Vec(sentences=node_sentences, vector_size=100, min_count=1, workers=4)

    x = []
    for data in node_features.values():
        words = list(data.values())
        vectors = [node_model.wv[word] for word in words if word in node_model.wv]
        x.append(np.mean(vectors, axis=0) if vectors else np.zeros(100))  # 100 chiều vector

    x = torch.tensor(x, dtype=torch.float)

    # Sử dụng Word2Vec để nhúng đặc trưng của cạnh
    edge_sentences = [list(data.values()) for data in edge_features]
    edge_model = Word2Vec(sentences=edge_sentences, vector_size=50, min_count=1, workers=4)  # Embedding cạnh 50 chiều

    edge_attr = []
    for data in edge_features:
        words = list(data.values())
        vectors = [edge_model.wv[word] for word in words if word in edge_model.wv]
        edge_attr.append(np.mean(vectors, axis=0) if vectors else np.zeros(50))  # 50 chiều vector

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Tạo đối tượng đồ thị PyG
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data
def load_graph_from_folder(c_file_path, output_dir):
    file_name = os.path.basename(c_file_path)
    export_dir = os.path.join(output_dir, file_name)
    print("file_name ",file_name)
    print("export_dir ",export_dir)
    if os.path.isdir(export_dir):
        print(f"Directory {export_dir} already exists.")
        return None

    subprocess.run(f'joern-parse "{os.path.abspath(c_file_path)}"', shell=True, check=True)
    subprocess.run(f'joern-export --repr=all --format=graphml --out "{output_dir}/{file_name}"', shell=True, check=True)
    file_path = os.path.join(export_dir, "export.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionary lưu thông tin node & mapping
    node_features = {}
    node_mapping = {}

    # Lấy danh sách node
    for idx, node in enumerate(root.findall("ns:graph/ns:node", settting.NAMESPACE)):
        node_id = node.get("id")
        node_data = {}

        # Lấy thuộc tính của node
        for data in node.findall("ns:data", settting.NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            node_data[key] = f"{key}:{value}"

        # Lưu node vào dictionary
        node_features[node_id] = node_data
        node_mapping[node_id] = idx  # Đánh số lại cho node

    # Xử lý cạnh (edges) & thuộc tính cạnh
    edges = []
    edge_features = []

    for edge in root.findall("ns:graph/ns:edge", settting.NAMESPACE):
        src = edge.get("source")
        tgt = edge.get("target")
        edge_data = {}

        # Lấy thuộc tính của cạnh
        for data in edge.findall("ns:data", settting.NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            edge_data[key] = f"{key}:{value}"

        if src in node_mapping and tgt in node_mapping:
            edges.append((node_mapping[src], node_mapping[tgt]))
            edge_features.append(edge_data)

    # Chuyển danh sách edges thành tensor
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        print(f"Lỗi: Không có cạnh trong {output_dir}")
        return None  # Bỏ qua nếu không có cạnh

    # Sử dụng Word2Vec để nhúng đặc trưng của node
    node_sentences = [list(data.values()) for data in node_features.values()]
    node_model = Word2Vec(sentences=node_sentences, vector_size=50, min_count=1, workers=4)

    x = np.array([
        np.mean([node_model.wv[word] for word in data.values() if word in node_model.wv], axis=0)
        if any(word in node_model.wv for word in data.values()) else np.zeros(100)
        for data in node_features.values()
    ], dtype=np.float32)

    x = torch.tensor(x, dtype=torch.float)



    # Sử dụng Word2Vec để nhúng đặc trưng của cạnh
    edge_sentences = [list(data.values()) for data in edge_features]
    edge_model = Word2Vec(sentences=edge_sentences, vector_size=50, min_count=1, workers=4)  # Embedding cạnh 50 chiều

    edge_attr = np.array([
        np.mean([edge_model.wv[word] for word in data.values() if word in edge_model.wv], axis=0)
        if any(word in edge_model.wv for word in data.values()) else np.zeros(50)
        for data in edge_features
    ], dtype=np.float32)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)


    # Tạo đối tượng đồ thị PyG
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data