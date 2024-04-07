from flask import Flask, render_template,jsonify,request, send_from_directory
from urllib import parse
from urllib import request as rq
import webbrowser
import os,base64,io, time, random
from PIL import Image, ImageFilter, ImageOps
try:
	from signal import SIGKILL
except:
	from signal import SIGABRT
import json

def stopprocess():
    pid = os.getpid()
    print(pid)
    try:
        os.kill(int(pid), SIGKILL)
    except:
        os.kill(int(pid), SIGABRT)
    return "none"
from pathlib import Path
import numpy as np


tt = ""
genflag = False
sharedata = {
    "savedata":"",
    "imgs":{}
}

app = Flask(__name__)

image = ""

bs = "256"

@app.route("/")
def index():
    return render_template('index.html',byte_im="",w="",h="",bs=bs)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/grabwfsmodels",methods=['GET', 'POST','DELETE'])
def grabwfsmodels():
    path = Path('../../../models/diffusers')
    models = [i for i in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path,i))]
    path = Path('../../../models/checkpoints')
    ckpts = [i for i in sorted(os.listdir(path)) if i.endswith((".safetensors", ".ckpt"))]
    models += ckpts
    path = Path('./workflows')
    wfs = sorted(os.listdir(path))
    wfs = [i.replace(".json","") for i in wfs]
    return{"models":models,"wfs":wfs}

@app.route("/cancelgen",methods=['GET', 'POST','DELETE'])
def cancelgen():
    global genflag
    genflag = True
    return{}

@app.route("/saveimg",methods=['GET', 'POST','DELETE'])
def saveimg():
    pixels = request.json["pixels"]
    if pixels != "":
        for i in range(0,len(pixels)):
            for j in range(0,len(pixels[i])):
                pixels[i][j] = tuple(pixels[i][j])
                
        
        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array)
        sharedata["imgs"]["image"] = json.dumps(np.array(image).tolist())
        sharedata["imgs"]["mask"] = json.dumps(np.array(image).tolist())
        sharedata["imgs"]["reference"] = json.dumps(np.array(image).tolist())
        wf = "saveimg.json"
        def queue_prompt(prompt_workflow):
            p = {"prompt": prompt_workflow}
            data = json.dumps(p).encode('utf-8')
            req =  rq.Request("http://127.0.0.1:8188/prompt", data=data)
            rq.urlopen(req)
        path = Path('./workflows')
        prompt_workflow = json.load(open(os.path.join(path,wf)))
        for i in prompt_workflow:
            if "seed" in prompt_workflow[i]['inputs']:
                print(i)
                prompt_workflow[i]['inputs']['seed'] = random.randint(0,10000000000)
        queue_prompt(prompt_workflow)
    return{}

@app.route("/grabwfparams",methods=['GET', 'POST','DELETE'])
def grabwfparams():
    wf = request.json["wf"]
    wf +=".json"
    path = Path('./workflows')
    prompt_workflow = json.load(open(os.path.join(path,wf)))
    for i in prompt_workflow:
        if "steps" in prompt_workflow[i]["inputs"]:
            params = [j for j in prompt_workflow[i]["inputs"] if any(x in str(type(prompt_workflow[i]["inputs"][j])) for x in ["int","float"])]
    return{"params":params}


@app.route("/prepare",methods=['GET', 'POST','DELETE'])
def prep():
    return{"exist":"no"}

@app.route("/generate",methods=['GET', 'POST','DELETE'])
def generate():
    wf = request.json["wf"]
    wf+=".json"
    model = request.json["model"]
    params = request.json["params"]
    def queue_prompt(prompt_workflow):
        p = {"prompt": prompt_workflow}
        data = json.dumps(p).encode('utf-8')
        req =  rq.Request("http://127.0.0.1:8188/prompt", data=data)
        rq.urlopen(req)
    path = Path('./workflows')
    prompt_workflow = json.load(open(os.path.join(path,wf)))
    if "stickerize" not in wf:
        for i in prompt_workflow:
            if "seed" in prompt_workflow[i]['inputs']:
                print(i)
                prompt_workflow[i]['inputs']['seed'] = random.randint(0,10000000000)
            if "model_name" in prompt_workflow[i]['inputs'] and prompt_workflow[i]['class_type'] !='UpscaleModelLoader':
                print(i)
                prompt_workflow[i]['inputs']['model_name'] = model
            if "ckpt_name" in prompt_workflow[i]['inputs']:
                print(i)
                prompt_workflow[i]['inputs']['ckpt_name'] = model
            if "steps" in prompt_workflow[i]['inputs']:
                for j in params:
                    prompt_workflow[i]['inputs'][j] = params[j]
    else:
        for i in prompt_workflow:
            if "seed" in prompt_workflow[i]['inputs']:
                print(i)
                prompt_workflow[i]['inputs']['seed'] = random.randint(0,10000000000)
    queue_prompt(prompt_workflow)
    return{}


@app.route("/settaesd",methods=['GET', 'POST','DELETE'])
def settaesd():
    global tt
    tt = request.json["data"]
    return {}

@app.route("/getGenStatus",methods=['GET', 'POST','DELETE'])
def getGenStatus():
    global genflag
    genflag = request.json["data"]
    return {}

@app.route("/getSharedData",methods=['GET', 'POST','DELETE'])
def getSharedData():
    global sharedata
    return sharedata


@app.route("/savedata",methods=['GET', 'POST','DELETE'])
def savedata():
    global image, tt, genflag
    img2img = "inpaint"
    print("trigerred")
    taesd = request.json["taesd"]
    savedata = request.json["savedata"]
    if taesd == "false":
        ff = int(savedata["ff"])
        selectorsize = request.json["selectorsize"]
        pixels = savedata["pxlsarray"]
        if pixels != "":
            for i in range(0,len(pixels)):
                for j in range(0,len(pixels[i])):
                    pixels[i][j] = tuple(pixels[i][j])
                    
            
            array = np.array(pixels, dtype=np.uint8)
            image = Image.fromarray(array)
            #image.save('erased.png')
        #image.save("out.png")
        sharedata["imgs"]["out"] = json.dumps(np.array(image).tolist())
        left = int(float(savedata["crpdims"]["left"]))
        top = int(float(savedata["crpdims"]["top"]))
        right = int(float(savedata["crpdims"]["right"]))
        bottom = int(float(savedata["crpdims"]["bottom"]))
        croped = image.crop((left,top,right,bottom))
        try:
            left = int(float(savedata["crpdimsref"]["left"]))
            top = int(float(savedata["crpdimsref"]["top"]))
            right = int(float(savedata["crpdimsref"]["right"]))
            bottom = int(float(savedata["crpdimsref"]["bottom"]))
            if left==0 and right==0 and top ==0 and bottom==0:
                ref = image
            else:
                ref = image.crop((left,top,right,bottom))
        except:
            ref = image
        #ref.save("reference.png")
        sharedata["imgs"]["reference"] = json.dumps(np.array(ref).tolist())
        croped2 = croped.copy()
        px = croped2.load()
        for i in range(0,croped2.size[0]):
            for j in range(0,croped2.size[1]):
                try:
                    if px[i,j][3] <255:
                        px[i,j] = (255,255,255,255)
                    else:
                        px[i,j] = (0,0,0,255)
                except:
                    px[i,j] = (0,0,0)
        selectorsize = int(selectorsize)
        bg = Image.new("RGB",(selectorsize,selectorsize),(0,0,0))
        bg2 = Image.new("RGB",(selectorsize,selectorsize),(255,255,255))
        add = (int(float(savedata["additionaldims"]["left"])),int(float(savedata["additionaldims"]["top"])),int(float(selectorsize-savedata["additionaldims"]["right"])),int(float(selectorsize-savedata["additionaldims"]["bottom"])))
        print(croped,add,".......")
        bg.paste(croped,(add))
        bg2.paste(croped2,(add))



        ###########################
        toparr = []
        leftarr = []
        rightarr = []
        bottomarr = []

        topleftarr = []
        toprightarr = []
        bottomleftarr = []
        bottomrightarr = []
        whitepix = 0
        px = bg2.load()
        for i in range(0,bg2.size[0]):
            for j in range(0,bg2.size[1]):
                if px[i,j][0] == 255:
                    whitepix+=1
                try:
                    if px[i,j][0] == 255 and 0<i<bg.size[0]-1 and 0<j<bg.size[1]-1:
                        if px[i,j+1][0] <255:
                            rightarr.append([i,j])
                        if px[i,j-1][0] <255:
                            leftarr.append([i,j])

                        if px[i+1,j][0] <255:
                            bottomarr.append([i,j])

                        if px[i-1,j][0] <255:
                            toparr.append([i,j])

                        if px[i-1,j-1][0] <255:
                            topleftarr.append([i,j])
                        if px[i-1,j+1][0] <255:
                            toprightarr.append([i,j])

                        if px[i+1,j+1][0] <255:
                            bottomrightarr.append([i,j])

                        if px[i+1,j-1][0] <255:
                            bottomleftarr.append([i,j])

                except:
                    continue

        
        for i in range(0,len(toparr)):
            for k in range(0,ff):
                try:
                    px[toparr[i][0]-k,toparr[i][1]] = (255,255,255)
                except:
                    continue

        for i in range(0,len(leftarr)):
            for k in range(0,ff):
                try:
                    px[leftarr[i][0],leftarr[i][1]-k] = (255,255,255)
                except:
                    continue

        for i in range(0,len(rightarr)):
            for k in range(0,ff):
                try:
                    px[rightarr[i][0],rightarr[i][1]+k] = (255,255,255)
                except:
                    continue

        for i in range(0,len(bottomarr)):
            for k in range(0,ff):
                try:
                    px[bottomarr[i][0]+k,bottomarr[i][1]] = (255,255,255)
                except:
                    continue
        


        for i in range(0,len(topleftarr)):
            for k in range(0,ff):
                try:
                    px[topleftarr[i][0]-k,topleftarr[i][1]-k] = (255,255,255)
                except:
                    continue

        for i in range(0,len(toprightarr)):
            for k in range(0,ff):
                try:
                    px[toprightarr[i][0]-k,toprightarr[i][1]+k] = (255,255,255)
                except:
                    continue

        for i in range(0,len(bottomleftarr)):
            for k in range(0,ff):
                try:
                    px[bottomleftarr[i][0]+k,bottomleftarr[i][1]-k] = (255,255,255)
                except:
                    continue

        for i in range(0,len(bottomrightarr)):
            for k in range(0,ff):
                try:
                    px[bottomrightarr[i][0]+k,bottomrightarr[i][1]+k] = (255,255,255)
                except:
                    continue
            
                
                




        #########################3


        #croped.save("cropped.png")
        #bg.save("image.png")
        sharedata["imgs"]["image"] = json.dumps(np.array(bg).tolist())
        if whitepix == 0:
            border = Image.new("RGB",(bg2.size[0],bg2.size[1]),(0,0,0)) 
            bg2 = ImageOps.invert(bg2)
            bg2 = bg2.resize((bg2.size[0]-ff*2,bg2.size[1]-ff*2))
            border.paste(bg2,(ff,ff))
            bg2 = border
            img2img = "img2img"
            print(img2img)
        bg2 = bg2.filter(ImageFilter.BoxBlur(ff-int(ff/2)))
        #bg2.save("mask.png")
        sharedata["imgs"]["mask"] = json.dumps(np.array(bg2).tolist())
        width = int(float(savedata["additionaldims"]["left"])) + int(float(savedata["additionaldims"]["right"]))
        height = int(float(savedata["additionaldims"]["top"])) + int(float(savedata["additionaldims"]["bottom"]))
        
        data = {
            "savedata":savedata
        }
        json_object = json.dumps(data, indent=4)
        '''with open("data.json", "w") as outfile:
            outfile.write(json_object)'''
        sharedata["savedata"] = savedata
        #stopprocess()
        taesds = "no"
    else:
        img2img = request.json["img2img"]
        if img2img=="":
            img2img = "inpaint"
    flag = False
    realflag = flag
    '''path = Path("./lastimage.json")
    with open(path,"r") as filename:
        status = json.load(filename)["done"]
        if status == "y":
            flag = True
            print("done!!!")
        else:
            print("reading....")
    if flag:
        path = Path("./lastimage.json")
        with open(path,"r") as filename:
            image = Image.open(json.load(filename)["lastimage"])
        data = {
            "lastimage":"",
            "done":"n"
        }
        json_object = json.dumps(data, indent=4)
        path = Path("./lastimage.json")
        with open(path, "w") as outfile:
            outfile.write(json_object)
        try:
            path = Path("./taesd.png")
            os.remove(path)
        except:
            pass
    '''
    if genflag:
        try:
            image = Image.open(genflag["genimg"])
        except: 
            pass
        tt = ""
        genflag = False
        flag = True
    if flag:
        taesds = "no"
        realflag = flag
        flag = False
        try:
            width = image.size[0]
            height = image.size[1]
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            byte_im = buf.getvalue()
            byte_im = base64.b64encode(byte_im).decode('utf-8')
            byte_im = f"data:image/png;base64,{byte_im}"
        except:
            byte_im = ""
            width = ""
            height = ""

    else:
        try:
            #path = Path("./taesd.png")
            #image = Image.open(path)
            width = tt["width"]
            height = tt["height"]
            byte_im = tt["img"]
            '''with open("data.json","r") as json_file:
                savedata = json.load(json_file)["savedata"]
            bg = Image.open("out.png").convert("RGBA")
            msksmall = Image.open("mask.png").convert("L")
            cropped = Image.open("image.png").convert("RGBA")
            width = int(savedata["additionaldims"]["right"]) + int(savedata["additionaldims"]["left"]) + bg.size[0]
            height = int(savedata["additionaldims"]["top"]) + int(savedata["additionaldims"]["bottom"]) + bg.size[1]
            new = Image.new("RGBA",(width,height),(0,0,0,0))
            lft = 0
            tp = 0
            image = Image.composite(image, cropped, msksmall)
            if savedata["additionaldims"]["left"]>0:
                lft = int(savedata["additionaldims"]["left"])

            if savedata["additionaldims"]["top"]>0:
                tp = int(savedata["additionaldims"]["top"])
            new.paste(bg,(lft,tp))
            lft = int(savedata["crpdims"]["left"])
            tp = int(savedata["crpdims"]["top"])
            new.paste(image,(lft,tp))
            image = new'''
            taesds = "no"
        except:
            taesds = "yes"
    try:
        return {"img":byte_im,"width":width,"height":height,"data":savedata,"flag":str(realflag),"taesd":taesds,"img2img":img2img}
    except:
        byte_im = ""
        width = ""
        height = ""
        return {"img":byte_im,"width":width,"height":height,"data":savedata,"flag":str(realflag),"taesd":taesds,"img2img":img2img}



if __name__ == "__main__":
	webbrowser.open("http://localhost:5000")
	app.run(debug=False)
