import torch
from flask import Flask, render_template, request, jsonify
import sys
sys.path.append(".")
sys.path.append("..")
from lsafunctions import get_all_LSA_fns
from loss import *
from io import BytesIO
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from base64 import encodebytes

functions=get_all_LSA_fns()
app = Flask(__name__,template_folder='.')


def convert_to_fp8_e5m2(T):
    return T.to(torch.float8_e5m2).to(torch.float32)
def convert_to_fp8_e4m3(T):
    return T.to(torch.float8_e4m3fn).to(torch.float32)

    
def square(logits,img_buf = BytesIO()):
    plt.figure(figsize=(logits.shape[0],logits.shape[1]))
    plt.imshow(logits)
    plt.savefig(img_buf, format='png')
    encoded_img = encodebytes(img_buf.getvalue()).decode('ascii') # encode as base64

    return encoded_img

    
def draw(logits,buffer=BytesIO()):
    sides = len(logits.shape)
    if sides==2:
        return square(logits,buffer)

   
@app.route("/lsa") 
def index():
    return render_template("./index.html")

def attempt(func,x):
    try:
        return func(x)
    except:
        print("failed to run",func)

        return torch.zeros_like(x)
def process(x):
    try:
        return float(x)
    except:
        return 0

@torch.no_grad()
@app.route('/lsa/data', methods=['GET','POST'])
async def getplots():

    data=request.get_json()
    values=[[process(x) for x in row] for row in data["values"]]
    maximize=data["maximize"]
    fp8=data["precision"]
    x=torch.tensor(values,dtype=torch.float32)
    if fp8=="E5M2":
        x=convert_to_fp8_e5m2(x)

    elif fp8=="E4M3":
        x=convert_to_fp8_e4m3(x)

    out={}
    outputs={name:func(x,maximize=maximize) for name,func in functions.items()}
    out.update({str(name) + " LSA score": torch.sum(x*value).item() for name,value in outputs.items()})
    out.update({str(name): value.tolist() for name,value in outputs.items()})
    output=jsonify(out)

    logging.warning("out"+str(out))
    return output

if __name__ == "__main__":

    app.run(host="localhost", port=5000, debug=True )
  
