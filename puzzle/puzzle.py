import random, json, os, io, base64, time
from urllib import request as rq
from urllib import parse
from PIL import Image, ImageOps
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np

image = ""
newwidth = 0
newheight = 0
width = 0
height = 0
size = 5
sizerows = 5
sizecols = 5
sw = 0
pieces = []
imgs = []
dct = {
    "[0, 1, 0, 999]":"63.png",
    "[1, 1, 0, 1]":"62.png",
    "[1, 0, 999, 0]":"61.png",
    "[0, 999, 1, 1]":"60.png",
    "[999, 0, 0, 0]":"59.png",
    "[999, 0, 1, 1]":"58.png",
    "[1, 999, 999, 0]":"57.png",
    "[0, 1, 0, 1]":"56.png",
    "[0, 1, 1, 1]":"55.png",
    "[1, 999, 0, 1]":"54.png",
    "[1, 999, 1, 1]":"53.png",
    "[1, 1, 0, 999]":"52.png",
    "[999, 999, 1, 0]":"51.png",
    "[1, 1, 999, 0]":"50.png",
    "[0, 1, 999, 1]":"49.png",
    "[999, 1, 0, 999]":"48.png",
    "[999, 999, 1, 1]":"47.png",
    "[0, 0, 999, 999]":"46.png",
    "[0, 0, 999, 1]":"45.png",
    "[999, 1, 0, 0]":"44.png",
    "[0, 0, 0, 999]":"43.png",
    "[0, 999, 999, 1]":"42.png",
    "[0, 999, 0, 0]":"41.png",
    "[0, 0, 0, 0]":"40.png",
    "[1, 1, 0, 0]":"39.png",
    "[1, 1, 999, 999]":"38.png",
    "[0, 1, 0, 0]":"37.png",
    "[1, 0, 0, 0]":"36.png",
    "[0, 0, 1, 1]":"35.png",
    "[1, 1, 999, 1]":"34.png",
    "[0, 1, 999, 0]":"33.png",
    "[0, 999, 1, 0]":"32.png",
    "[0, 1, 1, 0]":"31.png",
    "[999, 0, 1, 0]":"30.png",
    "[1, 0, 1, 1]":"29.png",
    "[999, 0, 1, 999]":"28.png",
    "[1, 0, 1, 0]":"27.png",
    "[0, 0, 1, 999]":"26.png",
    "[1, 0, 1, 999]":"25.png",
    "[0, 1, 999, 999]":"24.png",
    "[1, 1, 1, 0]":"23.png",
    "[1, 0, 0, 999]":"22.png",
    "[999, 0, 0, 999]":"21.png",
    "[999, 999, 0, 1]":"20.png",
    "[0, 999, 0, 1]":"19.png",
    "[1, 999, 999, 1]":"18.png",
    "[0, 0, 0, 1]":"17.png",
    "[999, 1, 1, 999]":"16.png",
    "[1, 999, 1, 0]":"15.png",
    "[0, 0, 1, 0]":"14.png",
    "[1, 0, 0, 1]":"13.png",
    "[0, 0, 999, 0]":"12.png",
    "[999, 0, 0, 1]":"11.png",
    "[999, 1, 1, 0]":"10.png",
    "[999, 999, 0, 0]":"9.png",
    "[1, 1, 1, 999]":"8.png",
    "[1, 999, 0, 0]":"7.png",
    "[999, 1, 0, 1]":"6.png",
    "[1, 0, 999, 1]":"5.png",
    "[0, 999, 999, 0]":"4.png",
    "[0, 1, 1, 999]":"3.png",
    "[1, 1, 1, 1]":"2.png",
    "[1, 0, 999, 999]":"1.png",
    "[999, 1, 1, 1]":"0.png"
    

}

def preparepieces():
    global sw, newwidth, newheight, width, height, size,  sizerows, sizecols
    newwidth = 0
    newheight = 0
    '''if width > height:
        width = width
        height = int(round(height/int(round(width/height))))
    elif height > width:
        width = int(round(width/int(round(height/width))))
        height = height
    else:
        width = width
        height = height
    print(width,height)'''

    if width <= height:
        if width > 800:
            sw = int(800/size)/121
            newwidth = 800
            newheight = int(round((800/width)*height))
        else:
            sw = int(width/size)/121
    else:
        if height > 800:
            sw = int(800/size)/121
            newheight = 800
            newwidth = int(float(round((800/height)*width)))
        else:
            sw = int(width/size)/121
    if newwidth >0:
        if newwidth > newheight:
            sizerows = size
            sizecols = int(round(size/int(round(newwidth/newheight))))
        elif newheight > newwidth:
            sizerows = int(round(size/int(round(newheight/newwidth))))
            sizecols = size
        else:
            sizerows = size
            sizecols = size    
    else:
        if width > height:
            sizerows = size
            print("the size is     ",width,height,int(round(width/height)),size)
            sizecols = int(round(size/int(round(width/height))))
        elif height > width:
            sizerows = int(round(size/int(round(height/width))))
            sizecols = size
        else:
            sizerows = size
            sizecols = size





    
    pn = 0
    for i in range(0,sizecols):        
        for j in range(0,sizerows):
            print(i,j,pn, sizerows,sizecols)
            if i == 0 and j==0:
                p = [999,999,random.randint(0,1),random.randint(0,1)]
            elif i == (sizecols-1) and j==0:
                p = [999,1 if pieces[pn-sizerows][3]==0 else 0,random.randint(0,1),999]
            elif j == (sizerows-1) and i == 0:
                p = [1 if pieces[pn-1][2]==0 else 0,999,999,random.randint(0,1)]
            elif j == (sizerows-1) and i == (sizecols-1):
                p = [1 if pieces[pn-1][2]==0 else 0,1 if pieces[pn-sizerows][3]==0 else 0,999,999]
            else:
                if i == 0:
                    p = [1 if pieces[pn-1][2]==0 else 0,999,random.randint(0,1),random.randint(0,1)]
                elif i == (sizecols-1):
                    p = [1 if pieces[pn-1][2]==0 else 0,1 if pieces[pn-sizerows][3]==0 else 0,random.randint(0,1),999]
                elif j == 0:
                    p = [999,1 if pieces[pn-sizerows][3]==0 else 0,random.randint(0,1),random.randint(0,1)]
                elif j == (sizerows-1):
                    p = [1 if pieces[pn-1][2]==0 else 0,1 if pieces[pn-sizerows][3]==0 else 0,999,random.randint(0,1)]
                else:
                    p = [1 if pieces[pn-1][2]==0 else 0,1 if pieces[pn-sizerows][3]==0 else 0,random.randint(0,1),random.randint(0,1)]       
            pieces.append(p)
            pn+=1


def maskimg():
    global sw, image, newwidth, newheight, width, height, sizerows, sizecols
    print(newwidth,newheight,sw)
    if newwidth > 0:
        image = image.resize((newwidth,newheight))
    width = image.size[0]
    height = image.size[1]
    n = 0
    tpttl = 0
    lfttl = 0
    '''for i in os.listdir("./static/pieces"):
        os.remove("./static/pieces/"+i)'''

    '''if width > height:
        sizerows = size
        sizecols = int(round(size/int(round(width/height))))
    elif height > width:
        sizerows = int(round(size/int(round(height/width))))
        sizecols = size
    else:
        sizerows = size
        sizecols = size'''
    image = image.resize((int(sizerows*121*sw),int(sizecols*121*sw)))
    for i in range(0,sizecols):
        for j in range(0,sizerows):
            print(len(pieces),(sizecols-1)*(sizerows-1),sizecols,sizerows)
            mask = Image.open(f"./pieces/{dct[str(pieces[n])]}")
            if j ==0 and i == 0:
                sh = sw 
                
            bg = Image.new("L",(mask.size[0],mask.size[1]),255)
            
            bg.paste(mask,(0,0))
            mask = ImageOps.invert(bg)

            l = int(pieces[n][0]*30*sw) if pieces[n][0]!=999 else 0
            t = int(pieces[n][1]*30*sw) if pieces[n][1]!=999 else 0
            r = int(pieces[n][2]*30*sw) if pieces[n][2]!=999 else 0
            b = int(pieces[n][3]*30*sw) if pieces[n][3]!=999 else 0


            mask = mask.resize((int(121*sw)+l+r,int(121*sh)+t+b))
            bg = Image.new("RGBA",(mask.size[0],mask.size[1]),(0,0,0,0))
            lft = int(121*sw)
            tp = int(121*sw)
            print(l,t,r,b,n)
            imagecropped = image.crop((0+lft*j-l,0+tp*i-t,lft+lft*j+r,tp+tp*i+b))
            print(imagecropped.size)
            print(bg.size)
            print(mask.size)
            im = Image.composite(imagecropped, bg, mask)
            #im.save(f"./static/pieces/piece{n}.png")
            buf = io.BytesIO()
            im.save(buf, format='PNG')
            byte_im = buf.getvalue()
            byte_im = base64.b64encode(byte_im).decode('utf-8')
            byte_im = f"data:image/png;base64,{byte_im}"
            imgs.append(byte_im)
            n+=1










    


app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("puzzle.html")


@app.route("/getdata",methods=['POST'])
def getdata():
    global pieces, imgs, image, width, height, size, sizerows
    pixels = request.json["savedata"]["pxlsarray"]
    size = int(request.json["savedata"]["size"])
    if pixels != "":
        for i in range(0,len(pixels)):
            for j in range(0,len(pixels[i])):
                pixels[i][j] = tuple(pixels[i][j])
                
        
        array = np.array(pixels, dtype=np.uint8)
        image = Image.fromarray(array)
    width = image.size[0]
    height = image.size[1]
    preparepieces()
    maskimg()
    response = jsonify({"pieces":pieces,"ratio":sw,"size":sizerows,"imgs":imgs})
    pieces = []
    imgs = []
    return response



@app.route("/getdatagen",methods=['POST'])
def getdatagen():
    global pieces, imgs, image, width, height, size
    size = int(request.json["savedata"]["size"])
    model = request.json["model"]
    prompt = request.json["savedata"]["prompt"]
    def queue_prompt(prompt_workflow):
        p = {"prompt": prompt_workflow}
        data = json.dumps(p).encode('utf-8')
        req =  rq.Request("http://127.0.0.1:8188/prompt", data=data)
        rq.urlopen(req)    

    if model == "SDVEGA":
        prompt_workflow = json.load(open('./static/api.json'))
        prompt_workflow['32']['inputs']['image'] = os.path.join(os.getcwd(),"img222.png")
        prompt_workflow['25']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['25']['inputs']['prompt'] = prompt
    else:
        prompt_workflow = json.load(open('./static/apiv2.json'))
        prompt_workflow['32']['inputs']['image'] = os.path.join(os.getcwd(),"img222.png")
        prompt_workflow['34']['inputs']['seed'] = random.randint(0,1000000000)
        prompt_workflow['34']['inputs']['prompt'] = prompt
    queue_prompt(prompt_workflow)
    flag = False
    while not flag:
        with open("./lastimage.json","r") as filename:
            status = json.load(filename)["done"]
            if status == "y":
                flag = True
            else:
                time.sleep(1)
    with open("./lastimage.json","r") as filename:
        image = Image.open(json.load(filename)["lastimage"])
    data = {
        "lastimage":"",
        "done":"n"
    }
    json_object = json.dumps(data, indent=4)
    with open("./lastimage.json", "w") as outfile:
        outfile.write(json_object)
    width = image.size[0]
    height = image.size[1]
    preparepieces()
    maskimg()
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    byte_im = base64.b64encode(byte_im).decode('utf-8')
    byte_im = f"data:image/png;base64,{byte_im}"
    response = jsonify({"pieces":pieces,"ratio":sw,"size":size,"imgs":imgs,"image":byte_im,"width":width,"height":height})
    pieces = []
    imgs = []
    return response





if __name__ == "__main__":
    app.run()






    
        