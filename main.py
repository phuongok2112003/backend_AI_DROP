from fastapi import FastAPI, File, UploadFile, Form
import os
import torch
import joblib
import datetime
from torch_geometric.data import DataLoader
# from model.GraphModel import GraphModel
from  model.GCN import GraphModel as  GCN
from model.GraphModel import GraphModel as GCN_CNN
from model.SAGE_CNN import GraphModel as SAGE_CNN
import shutil
from enums.enums import settting
from joern import run_joern_analysis,load_graph_from_folder
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from enums.model_enum import ModelType 
from model.MLP import MLP
import uvicorn
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc thay bằng ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả method (GET, POST, ...)
    allow_headers=["*"],  # Cho phép tất cả headers
)
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_fixed_weights(model,WEIGHT_PATH):
    """Tải trọng số cố định nếu đã có file."""
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
        model.eval()  # Đặt chế độ inference
        print(f"✅ Đã tải trọng số cố định từ '{WEIGHT_PATH}'.")
set_seed(42)
scaler = joblib.load("scaler.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@app.post("/predict")
async def predict_vulnerability(files: list[UploadFile] = File(...),status: ModelType = Form(...)):
    results = []

    try:
        os.makedirs(settting.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settting.OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create necessary directories: {str(e)}")
    if status == ModelType.gcn_cnn:

        model = GCN_CNN(node_input_dim=50, node_hidden_dim=64, node_output_dim=16,
                   edge_input_dim=50, edge_output_dim=16, final_dim=2).to(device)
        WEIGHT_PATH = "fixed_weights.pth"
        load_fixed_weights(model=model,WEIGHT_PATH=WEIGHT_PATH) 
        try:
            model_loaded = joblib.load(settting.MODEL_PATH_GCN_CNN)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    elif status == ModelType.gcn:

        model = GCN(node_input_dim=50, node_hidden_dim=64, node_output_dim=32,
                edge_input_dim=50, edge_hidden_dim=32, edge_output_dim=16).to(device)
        WEIGHT_PATH = "GCN_drop_weights.pth"
        load_fixed_weights(model=model,WEIGHT_PATH=WEIGHT_PATH)
        try:
            model_loaded = joblib.load(settting.MODEL_PATH_GCN)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    else:
        model = SAGE_CNN(node_input_dim=50, node_hidden_dim=64, node_output_dim=32,
                edge_input_dim=50, edge_output_dim=16, final_dim=2).to(device)
        WEIGHT_PATH = "SAGE_CNN_drop_weights.pth"
        load_fixed_weights(model=model,WEIGHT_PATH=WEIGHT_PATH)
        try:

            input_size = 48
            loaded_model = MLP(input_size, dropout_rate=0.3)

            # Load trọng số đã lưu
            loaded_model.load_state_dict(torch.load("best_mlp_model.pth"))
            loaded_model.eval() 
            model_loaded = loaded_model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    model.eval()
    for file in files:
        # Tạo timestamp để tránh trùng lặp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_safe = f"{timestamp}_{file.filename}"

        file_path = os.path.join(settting.UPLOAD_DIR, filename_safe)

        # Lưu file lên server
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Phân tích file với Joern
        data = load_graph_from_folder(file_path, settting.OUTPUT_DIR)
        if data is None:
            os.remove(file_path)
            results.append({"filename": file.filename, "error": "Failed to process file"})
            continue  # Bỏ qua file lỗi và xử lý file tiếp theo
        
 
        # Xóa file sau khi xử lý
        os.remove(file_path)
        export_dir = os.path.join(settting.OUTPUT_DIR, filename_safe)
        if os.path.isdir(export_dir):
            print("Xóa ", export_dir)
            shutil.rmtree(export_dir, ignore_errors=True)

        data_loader = DataLoader([data], batch_size=1, shuffle=False)
        try: 
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(device)
                    features = model(batch)
                    X = features.cpu().numpy()
                    if status == ModelType.sage_cnn:
                        
                        X = scaler.transform(X)
                        X = torch.tensor(X, dtype=torch.float32)
                        with torch.no_grad():
                            y_pred_tensor = model_loaded(X)
                            y_pred = torch.argmax(y_pred_tensor, dim=1).numpy()
                            print(y_pred)

                        
                    else:
                        y_pred = model_loaded.predict(X)
        except Exception as e:
            results.append({"filename": file.filename, "error": f"Failed to process file: {str(e)}"})
            continue  # Tiếp tục với file tiếp theo

        # Lưu kết quả
        results.append({"filename": file.filename, "vulnerable": bool(y_pred[0])})

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000,reload=True)