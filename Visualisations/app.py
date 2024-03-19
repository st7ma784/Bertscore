import torch
from flask import Flask, render_template, request, jsonify, send_file, make_response
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
from flask import jsonify
#make a dictionary of all the functions we want to use

functions=get_all_LSA_fns()
#normedfunctions={i:get_lsa_fn(i) for i in range(1,17)}
app = Flask(__name__,template_folder='.')


def convert_to_fp8_e5m2(T):
    return T.to(torch.float8_e5m2).to(torch.float32)
def convert_to_fp8_e4m3(T):
    return T.to(torch.float8_e4m3fn).to(torch.float32)

    
def square(logits,img_buf = BytesIO()):
    #a function that takes numpy logits and plots them on an x and y axis
    plt.figure(figsize=(logits.shape[0],logits.shape[1]))
    plt.imshow(logits)
    #do not display the graph, but save it to a buffer
    plt.savefig(img_buf, format='png')
    encoded_img = encodebytes(img_buf.getvalue()).decode('ascii') # encode as base64

    return encoded_img

    
def draw(logits,buffer=BytesIO()):
    # Defining the side of the cube
    sides = len(logits.shape) #(subtract to take slices)
    # Calling the cubes () function
    #cubes(sides)
    
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
    #take the string value x and convert it to a float
    try:
        return float(x)
    except:
        return 0

@torch.no_grad()
@app.route('/lsa/data', methods=['GET','POST'])
async def getplots():

    # print("request",request.get_data())
    data=request.get_json()
    #convert from list of list of strings to list of list of floats to a tensor 
    #any nan values are converted to 0 and remove non-numeric values
    values=[[process(x) for x in row] for row in data["values"]]
    maximize=data["maximize"]
    fp8=data["precision"]
    x=torch.tensor(values,dtype=torch.float32)
    #logging.warning("values"+str(values))
    #log size of x to console 
    if fp8=="E5M2":
        x=convert_to_fp8_e5m2(x)

    elif fp8=="E4M3":
        x=convert_to_fp8_e4m3(x)

    out={}
    # check if x is square i.e shape[0]==shape[1]
    outputs={name:func(x,maximize=maximize) for name,func in functions.items()}
    # for i in outputs.values():
    #         logging.warning(i)
    # for lossname,func in get_all_loss_fns().items():
    #     out.update({"{} with {} loss".format(name,lossname): str(loss(outputs[lossname],x,app))} for name,_ in functions.items())
    #out.update({str(name) + " loss": str(loss(outputs[name],x,app)) for name,_ in functions.items()})
    out.update({str(name) + " LSA score": torch.sum(x*value).item() for name,value in outputs.items()})
    out.update({str(name): value.tolist() for name,value in outputs.items()})
    output=jsonify(out)

    logging.warning("out"+str(out))
    return output

if __name__ == "__main__":

    app.run(host="localhost", port=5000, debug=True )
  
