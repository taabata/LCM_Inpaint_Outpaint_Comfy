from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from flask_cors import CORS
import os, io, base64, glob, json, time, random
from urllib import request as rq



app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("mix.html")

@app.route("/stickerize",methods=["GET","POST"])
def stickerize():
    imagedata = request.json["image"]
    pixels = imagedata
    if pixels != "":
        for i in range(0,len(pixels)):
            for j in range(0,len(pixels[i])):
                pixels[i][j] = tuple(pixels[i][j])
                
        
        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array)
        image.save('image1.png')
    list_of_files = glob.glob('../../../output/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    def queue_prompt(prompt_workflow):
        p = {"prompt": prompt_workflow}
        data = json.dumps(p).encode('utf-8')
        req =  rq.Request("http://127.0.0.1:8188/prompt", data=data)
        rq.urlopen(req)    
    prompt_workflow = json.load(open('./static/stickerize_api.json'))
    prompt_workflow['36']['inputs']['image'] = os.path.join(os.getcwd(),"image1.png")
    prompt_workflow['36']['inputs']['seed'] = random.randint(0,100000)
    queue_prompt(prompt_workflow)
    new_file = latest_file
    while new_file == latest_file:
        list_of_files = glob.glob('../../../output/*')
        new_file = max(list_of_files, key=os.path.getctime)
    time.sleep(1)
    image = Image.open(f'../../../output/{os.path.basename(new_file)}')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im}"  
    print(byte_im)
    return {"byte_im":byte_im}

@app.route("/mix",methods=["GET","POST"])
def mix():
    imagedata1 = request.json["image1"]
    imagedata2 = request.json["image2"]
    setting = request.json["setting"]
    print(setting)
    pixels = imagedata1
    if pixels != "":
        for i in range(0,len(pixels)):
            for j in range(0,len(pixels[i])):
                pixels[i][j] = tuple(pixels[i][j])
                
        
        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array)
        image.save('image1.png')
    pixels = imagedata2
    if pixels != "":
        for i in range(0,len(pixels)):
            for j in range(0,len(pixels[i])):
                pixels[i][j] = tuple(pixels[i][j])
                
        
        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array)
        image.save('image2.png')
    list_of_files = glob.glob('../../../output/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    def queue_prompt(prompt_workflow):
        p = {"prompt": prompt_workflow}
        data = json.dumps(p).encode('utf-8')
        req =  rq.Request("http://127.0.0.1:8188/prompt", data=data)
        rq.urlopen(req)    
    prompt_workflow = json.load(open('./static/mix_api.json'))
    prompt_workflow['171']['inputs']['image'] = os.path.join(os.getcwd(),"image1.png")
    prompt_workflow['171']['inputs']['seed'] = random.randint(0,100000)
    prompt_workflow['172']['inputs']['image'] = os.path.join(os.getcwd(),"image2.png")
    prompt_workflow['172']['inputs']['seed'] = random.randint(0,100000)
    if setting == 1:
        prompt_workflow['1']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['7']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['83']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['159']['inputs']['style_fidelity'] = 0.99

        prompt_workflow['1']['inputs']['strength'] = 0.75
        prompt_workflow['7']['inputs']['strength'] = 0.75
        prompt_workflow['83']['inputs']['strength'] = 0.75
        prompt_workflow['159']['inputs']['strength'] = 0.75

        prompt_workflow['1']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['7']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['83']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['159']['inputs']['ipadapter_scale'] = 0.80

        prompt_workflow['1']['inputs']['controlnet_conditioning_scale'] = 0.50
        prompt_workflow['7']['inputs']['controlnet_conditioning_scale'] = 0.50
        prompt_workflow['83']['inputs']['controlnet_conditioning_scale'] = 0.50
        prompt_workflow['159']['inputs']['controlnet_conditioning_scale'] = 0.50

        prompt_workflow['1']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['7']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['83']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['159']['inputs']['seed'] = random.randint(0,1000000000)
    elif setting == 2:
        prompt_workflow['1']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['7']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['83']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['159']['inputs']['style_fidelity'] = 0.99

        prompt_workflow['1']['inputs']['strength'] = 0.75
        prompt_workflow['7']['inputs']['strength'] = 0.75
        prompt_workflow['83']['inputs']['strength'] = 0.75
        prompt_workflow['159']['inputs']['strength'] = 0.75

        prompt_workflow['1']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['7']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['83']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['159']['inputs']['ipadapter_scale'] = 0.80

        prompt_workflow['1']['inputs']['controlnet_conditioning_scale'] = 0.24
        prompt_workflow['7']['inputs']['controlnet_conditioning_scale'] = 0.24
        prompt_workflow['83']['inputs']['controlnet_conditioning_scale'] = 0.24
        prompt_workflow['159']['inputs']['controlnet_conditioning_scale'] = 0.24

        prompt_workflow['1']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['7']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['83']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['159']['inputs']['seed'] = random.randint(0,1000000000)    
    elif setting ==3:
        prompt_workflow['1']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['7']['inputs']['style_fidelity'] = 0.50
        prompt_workflow['83']['inputs']['style_fidelity'] = 0.99
        prompt_workflow['159']['inputs']['style_fidelity'] = 0.99

        prompt_workflow['1']['inputs']['strength'] = 0.75
        prompt_workflow['7']['inputs']['strength'] = 0.75
        prompt_workflow['83']['inputs']['strength'] = 0.75
        prompt_workflow['159']['inputs']['strength'] = 0.75

        prompt_workflow['1']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['7']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['83']['inputs']['ipadapter_scale'] = 0.80
        prompt_workflow['159']['inputs']['ipadapter_scale'] = 0.80

        prompt_workflow['1']['inputs']['controlnet_conditioning_scale'] = 0.50
        prompt_workflow['7']['inputs']['controlnet_conditioning_scale'] = 0.24
        prompt_workflow['83']['inputs']['controlnet_conditioning_scale'] = 0.50
        prompt_workflow['159']['inputs']['controlnet_conditioning_scale'] = 0.50

        prompt_workflow['1']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['7']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['83']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['159']['inputs']['seed'] = random.randint(0,1000000000)
    queue_prompt(prompt_workflow)
    new_file = latest_file
    while new_file == latest_file:
        list_of_files = glob.glob('../../../output/*')
        new_file = max(list_of_files, key=os.path.getctime)
    time.sleep(1)
    image = Image.open(f'../../../output/{os.path.basename(new_file)}')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im}"  
    print(byte_im)
    return {"byte_im":byte_im}

if __name__ == "__main__":
    app.run(debug=True)