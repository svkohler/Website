from tabnanny import check
from typing import Optional
from fastapi import FastAPI, Response, status, HTTPException, Request, File, UploadFile, Form
from fastapi.params import Body
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random
import time

from utils import get_sat_image, MLP_classifier, MLP
from SimCLR import ResNetSimCLR
import torchvision.models as models
import torch
from torch import nn
from torchvision.transforms import ToTensor

import psycopg2
from psycopg2.extras import RealDictCursor

from pydantic import BaseModel

from random import randrange

from PIL import Image
import io
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

device = torch.device('cpu')
model = ResNetSimCLR(models.__dict__['resnet50'])
model = nn.DataParallel(model)
checkpoint = torch.load(
    './static/preloaded_models/SimCLR_best_epoch_224.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
encoder = nn.Sequential(
    *list(model.module.encoder.children())[:-1])
encoder.eval()
encoder.to(device)
print('Successfully built encoder')

clf = MLP(4096, 100, 1)
clf.load_state_dict(torch.load('./static/preloaded_models/SimCLR224.pth'))
print('Successfully built classifier')


class Coordinates(BaseModel):
    bottom_left_long: float
    bottom_left_lat: float
    top_right_long: float
    top_right_lat: float


class Result(BaseModel):
    result: float


PATH = os.getcwd()+'/static/images/'


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/upload_drone_img")
async def handle_img(drone_img: UploadFile = File(...)):
    content_drone_img = await drone_img.read()
    img = Image.open(io.BytesIO(content_drone_img))
    img = img.resize((224, 224))
    img.save('./static/images/drone.png')


@app.get("/delete_images")
async def d():
    try:
        os.remove('./static/images/satellite.png')
        print('deleted satellite image')
    except:
        print('No satellite image to delete')
    try:
        os.remove('./static/images/drone.png')
        print('deleted drone image')
    except:
        print('No drone image to delete')


@app.post("/upload_coordinates")
async def handle_coordinates(bottom_left_long: float = Form(...), bottom_left_lat: float = Form(...), top_right_long: float = Form(...), top_right_lat: float = Form(...)):
    coordinates = {}
    coordinates['bottom_left_long'] = bottom_left_long
    coordinates['bottom_left_lat'] = bottom_left_lat
    coordinates['top_right_long'] = top_right_long
    coordinates['top_right_lat'] = top_right_lat
    get_sat_image(coordinates, PATH)


def val():
    img_drone = ToTensor()(Image.open('./static/images/drone.png'))
    img_sat = ToTensor()(Image.open('./static/images/satellite.png'))

    with torch.no_grad():
        drone_emb = encoder(torch.unsqueeze(img_drone, 0))
        sat_emb = encoder(torch.unsqueeze(img_sat, 0))
        result = clf(torch.squeeze(
            torch.cat((sat_emb, drone_emb), dim=1)).cpu().detach())
    print(result)
    result = {
        "result": int(result.round())
    }
    return result


@ app.get("/validate")
async def validate():
    result = val()
    return result
