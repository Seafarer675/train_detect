from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
import os
from PIL import Image
import shutil
import re
from pathlib import Path
import uuid
from bs4 import BeautifulSoup
import requests

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
WEIGHT_DIR = BASE_DIR / "weight"

model_spl = YOLO(WEIGHT_DIR / "best_detect.pt")
model_cls = YOLO(WEIGHT_DIR / "best_classify.pt")

BASE_TMP_DIR = "/tmp/yolo_predict"

train_name = [
    "DHL100", "E100", "E200", "E300", "E400", "E500", "E1000",
    "EMU100", "EMU300", "EMU400", "EMU500", "EMU600", "EMU700",
    "EMU800", "EMU900", "EMU1200", "EMU3000",
    "R20", "R100", "R150", "R180",
    "S200", "S300", "S400",
    "TEMU1000", "TEMU2000"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file)

    # 為每個 request 建立獨立資料夾
    request_id = str(uuid.uuid4())
    project_dir = os.path.join(BASE_TMP_DIR, request_id)
    model_spl.predict(
        img,
        conf=0.3,
        save=True,
        project=project_dir,
        name="predict"
    )

    split_dir = os.path.join(project_dir, "predict")
    split_img = os.listdir(split_dir)
    split_img_path = os.path.join(split_dir, split_img[0])
    result = model_cls.predict(
        split_img_path,
        conf=0.7
    )

    # 清理該 request 的資料
    shutil.rmtree("/tmp", ignore_errors=True)

    conf = result[0].probs.top1conf.item()
    label = model_cls.names[result[0].probs.top1]

    if conf < 0.6:
        txt = f"此預測結果信心值不高({round(conf, 2)}) 還請自行確認"
        return label, txt
    else:
        return label

@app.get("/detail")
def detail(r: str):
    schema = {
    "車型名稱": r,
    "製造廠商": None,
    "最大出力": None,
    "牽引力": None,
    "最高車速": None,
    "營運速限": None,
    "總重": None,
    "軸重": None,
    "最大尺寸_長": None,
    "最大尺寸_寬": None,
    "最大尺寸_高": None,
}
    url = ""
    new_list = []
    r = r.upper()
    if r.startswith(("EMU", "TEMU")):
        url = "traemu"
    elif r.startswith("E"):
        url = "trael"
    else:
        url = "tradl"
        
    if r in ("R180", "R190"):
        r = "R180-190"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(f"https://emu300ct.web.fc2.com/index/{url}/{r}.htm", headers = headers, timeout = 10)
    soup = BeautifulSoup(response.content, 'html.parser')

    cell = soup.find(
        lambda tag:
            tag.name in ("td", "th")
            and tag.get_text(strip=True) == "製造廠商"
    )

    if not cell:
        return "找不到目標 table"
    else:
        table = cell.find_parent("table")

        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if '最大尺寸長寬高' in cols or '車體尺寸長寬高' in cols:
                size_long = "".join(cols[2:7:4])
                size_width = "".join(cols[3:8:4])
                size_height = "".join(cols[4:9:4])
                rows.append(size_long)
                rows.append(size_width)
                rows.append(size_height)
                break
            else:
                rows.append(re.sub(r"[ \s]+", "","".join(cols)))

        for i in schema.keys():
            for j in rows:
                if j[0] in ("長", "寬", "高"):
                    j = f"最大尺寸_{j[0] + j[1:]}"
                if i in j:
                    schema[i] = j[len(i):]
                    break
        return schema

@app.get("/compare")
def compare(r1:str, r2:str):
    schema1 = detail(r1)
    schema2 = detail(r2)
    return schema1, schema2