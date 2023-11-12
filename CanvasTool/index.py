from flask import Flask, render_template,jsonify,request, send_from_directory
import webbrowser
import os,base64,io
from PIL import Image, ImageFilter, ImageOps
try:
	from signal import SIGKILL
except:
	from signal import SIGABRT
import json
from tkinter import filedialog
def stopprocess():
    pid = os.getpid()
    print(pid)
    try:
        os.kill(int(pid), SIGKILL)
    except:
        os.kill(int(pid), SIGABRT)
    return "none"

import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-s","--same")
parser.add_argument("-p","--path")
args = parser.parse_args()


app = Flask(__name__)
image = ""
if args.same != "yes":
    image = Image.open(filedialog.askopenfilename())
else:
    with open("./lastimage.json","r") as filename:
        image = Image.open(json.load(filename)["lastimage"])
w = str(int(float(image.size[0])))
h = str(int(float(image.size[1])))

try:
    with open("./data.json","r") as filename:
        bs = str(json.load(filename)["savedata"]["boxsz"])
        pixels = json.load(filename)["savedata"]["pxlsarray"]
        if pixels != "":
            for i in range(0,len(pixels)):
                for j in range(0,len(pixels[i])):
                    pixels[i][j] = tuple(pixels[i][j])
                    
            
            array = np.array(pixels, dtype=np.uint8)
            image = Image.fromarray(array)
            new_image.save('erased.png')
except:
    pass
bs = "256"

buf = io.BytesIO()
image.save(buf, format='PNG')
byte_im = buf.getvalue()
byte_im = base64.b64encode(byte_im).decode('utf-8')
byte_im = f"data:image/png;base64,{byte_im}"
@app.route("/")
def index():
    return render_template('index.html',byte_im=byte_im,w=w,h=h,bs=bs)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/prepare",methods=['GET', 'POST','DELETE'])
def prepare():
    with open("data.json","r") as json_file:
        d = json.load(json_file)
        data = {"width":str(image.size[0]),"height":str(image.size[1]),"boxsz":str(d["boxsz"]),"imgpos":str(d["imgpos"])}
    return data



@app.route("/savedata",methods=['GET', 'POST','DELETE'])
def savedata():
    global image
    print("trigerred")
    savedata = request.json["savedata"]
    #print(savedata)
    ff = int(savedata["ff"])
    selectorsize = request.json["selectorsize"]
    pixels = savedata["pxlsarray"]
    if pixels != "":
        for i in range(0,len(pixels)):
            for j in range(0,len(pixels[i])):
                pixels[i][j] = tuple(pixels[i][j])
                
        
        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array)
        image.save('erased.png')
    image.save("out.png")
    left = int(float(savedata["crpdims"]["left"]))
    top = int(float(savedata["crpdims"]["top"]))
    right = int(float(savedata["crpdims"]["right"]))
    bottom = int(float(savedata["crpdims"]["bottom"]))
    croped = image.crop((left,top,right,bottom))
    croped2 = croped.copy()
    px = croped2.load()
    for i in range(0,croped2.size[0]):
        for j in range(0,croped2.size[1]):
            try:
                if px[i,j][3] == 0:
                    px[i,j] = (255,255,255,255)
                else:
                    px[i,j] = (0,0,0,255)
            except:
                px[i,j] = (0,0,0)
    selectorsize = int(selectorsize)
    bg = Image.new("RGB",(selectorsize,selectorsize),(0,0,0))
    bg2 = Image.new("RGB",(selectorsize,selectorsize),(255,255,255))
    add = (int(float(savedata["additionaldims"]["left"])),int(float(savedata["additionaldims"]["top"])),int(float(selectorsize-savedata["additionaldims"]["right"])),int(float(selectorsize-savedata["additionaldims"]["bottom"])))
    bg.paste(croped,(add))
    bg2.paste(croped2,(add))



    ###########################
    toparr = []
    leftarr = []
    rightarr = []
    bottomarr = []
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
        
            
            




    #########################3


    croped.save("cropped.png")
    bg.save("image.png")
    if whitepix == 0:
        border = Image.new("RGB",(bg2.size[0],bg2.size[1]),(0,0,0)) 
        bg2 = ImageOps.invert(bg2)
        bg2 = bg2.resize((bg2.size[0]-ff*2,bg2.size[1]-ff*2))
        border.paste(bg2,(ff,ff))
        bg2 = border
    bg2 = bg2.filter(ImageFilter.BoxBlur(ff))
    bg2.save("mask.png")
    width = int(float(savedata["additionaldims"]["left"])) + int(float(savedata["additionaldims"]["right"]))
    height = int(float(savedata["additionaldims"]["top"])) + int(float(savedata["additionaldims"]["bottom"]))
    
    data = {
        "savedata":savedata
    }
    json_object = json.dumps(data, indent=4)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)
    stopprocess()
    return 0



if __name__ == "__main__":
	webbrowser.open("http://localhost:5000")
	app.run(debug=False)
