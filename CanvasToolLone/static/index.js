var mousex = 0;
var boxleft = "";
var boxtop = "";
var imgflip = false;
var fsimgs = {};
var selmod = "";
var selparam = "";
var selwf = "";
var genflag = false;
var canvasflag = true;
var taesd = "false";
var img2img = "";
var backup = {
    "img":"",
    "left":"",
    "top":"",
    "width":"",
    "height":""
}
var params = {};
var mousey = 0;
var lockflag = false;
var prevmousex = 0;
var prevmousey = 0;
var clickflag = false;
var moveimgenableflag = false;
var selectorsize = 256;
var crplft = 0;
var crptp = 0;
var crprght = 0;
var crpbtm = 0;
var eraseflag = false;
var drawenableflag = false;
var eraseenableflag = false;
var drawflag = false;
var ll = false;
var lr = false;
var lt = false;
var lb = false;
var setref = false;
var comporder = [];
var imagetomove = "";
var cropflag = false;
var cropcount = 0;
var compareflag = false;
var savedata = {
    "crpdimsref": {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    },
    "crpdims": {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    },
    "additionaldims": {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    },
    "boxsz":512,
    "ff":10,
    "pxlsarray":""


};

var imagedims = {
    "width":128,
    "height":128
};

function undoimg(){
    document.getElementById("sourceimg").width = backup["width"];
    document.getElementById("sourceimg").height =backup["height"];
    document.getElementById("source").width = backup["width"];
    document.getElementById("source").height = backup["height"];
    document.getElementById("source").style.left = backup["left"];
    document.getElementById("source").style.top = backup["top"];
    document.getElementById("sourceimg").src = backup["img"];
    img = new Image();
    img.onload = function() {            
        var canvas = document.getElementById("source");
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);        
        imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
        data = imgData.data;
        pxls = [];
        for (let i = 0; i < data.length; i += 4) {
            var pixel = [];
            pixel.push(data[i]);
            pixel.push(data[i+1]);
            pixel.push(data[i+2]);
            pixel.push(data[i+3]);
            pxls.push(pixel);
        }
        var pxlsarray = [[]];
        for(i = 0;i<pxls.length;i++){
            pxlsarray[pxlsarray.length-1].push(pxls[i]);
            if(i<pxls.length-2){
                if((i+1)%canvas.width==0 && i>canvas.width-10){
                    pxlsarray.push([]);
                }
            }
            
        }
        savedata["pxlsarray"] = pxlsarray;
    }
    img.src = backup["img"];
    fsimgs["main"] = backup["img"];
    savedata["crpdims"] = {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    };
    savedata["additionaldims"] = {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    };
    document.getElementById("oldimg").src = document.getElementById("sourceimg").src;
    document.getElementById("sourceimg").onload = function(){
        document.getElementById("oldimg").width = document.getElementById("source").width;
        document.getElementById("oldimg").height = document.getElementById("source").height;
    }

}

function paramset(){
    if(document.getElementById("paramsbutton").innerHTML!="Parameter"){
        let paramname = document.getElementById("paramsbutton").innerHTML;
        params[paramname] = document.getElementById("paramvalue").value;
    }
}

window.onload = function(){
    document.getElementById("erasersize").value = 10;
    document.getElementById("ff").value = 10;
    document.getElementById("selectorsize").value = 256;
    prepare();

    //copied from https://jsfiddle.net/4N6D9/1/
    document.getElementById("openimg").onchange =function(e) {
        var file, img;   
        if ((file = this.files[0])) {
            img = new Image();
            img.onload = function() {
                document.getElementById("sourceimg").width = this.width;
                document.getElementById("sourceimg").height = this.height;
                document.getElementById("source").width = this.width;
                document.getElementById("source").height = this.height;
                var canvas = document.getElementById("source");
                var ctx = canvas.getContext("2d");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                var img = document.getElementById("sourceimg");
                ctx.drawImage(img, 0, 0);
                imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
                data = imgData.data;
                pxls = [];
                for (let i = 0; i < data.length; i += 4) {
                    var pixel = [];
                    pixel.push(data[i]);
                    pixel.push(data[i+1]);
                    pixel.push(data[i+2]);
                    pixel.push(data[i+3]);
                    pxls.push(pixel);
                }
                var pxlsarray = [[]];
                for(i = 0;i<pxls.length;i++){
                    pxlsarray[pxlsarray.length-1].push(pxls[i]);
                    if(i<pxls.length-2){
                        if((i+1)%canvas.width==0 && i>canvas.width-10){
                            pxlsarray.push([]);
                        }
                    }
                    
                }
                savedata["pxlsarray"] = pxlsarray;
                console.log(savedata["pxlsarray"]);
                backup["img"] = document.getElementById("sourceimg").src;
            };
            img.src = URL.createObjectURL(file);
            document.getElementById("sourceimg").src = URL.createObjectURL(file);
            document.getElementById("oldimg").src = URL.createObjectURL(file);
            document.getElementById("uploadcontainer").style.visibility = "hidden";
            
    
    
        }
        fsimgs["main"] = document.getElementById("sourceimg").src;
        
    
    }

    document.getElementById("openimgadd").onchange =function(e) {
        document.getElementById("settings").style.visibility = "hidden";
        document.getElementById("undo").style.visibility = "hidden";
        var file, img;
        if ((file = this.files[0])) {
            comporder = [];
            document.getElementById("selector").style.visibility = "hidden";
            var num = (document.getElementById("canvascontainer").children.length /2)+1;
            num = String(num);
            document.getElementById("settings2").style.visibility = "visible";
            document.getElementById("openimgadd").remove();
            var newel = document.createElement("input");
            newel.id = "openimgadd";
            newel.type = "file";
            newel.onchange =function(e) {
                var file, img;
                if ((file = this.files[0])) {
                    comporder = [];
                    document.getElementById("selector").style.visibility = "hidden";
                    var num = (document.getElementById("canvascontainer").children.length /2)+1;
                    num = String(num);
                    document.getElementById("settings2").style.visibility = "visible";
                    var newel = document.createElement("div");
                    newel.id = "moveimg"+num;
                    newel.addEventListener("click",moveimgenable);
                    newel.className = "moveimgclass txt";
                    newel.style.left = 5+(num-1)*10+"%";
                    var newelpar = document.createElement("p");            
                    newelpar.innerHTML = "Move Image "+num;
                    newelpar.className = "txt";
                    newel.append(newelpar);
                    document.getElementById("settings2").append(newel);
                    newel = document.createElement("img");
                    newel.id = "sourceimg"+num;
                    newel.style.display = "none";
                    document.getElementById("canvascontainer").append(newel);
                    newel = document.createElement("canvas");
                    newel.id = "source"+num;
                    newel.className = "canvas";
                    newel.style.position = "fixed";
                    newel.style.left = parseInt(document.getElementById("source").style.left) + document.getElementById("source").width + "px";
                    newel.style.top = document.getElementById("source").style.top;
                    document.getElementById("canvascontainer").append(newel);
                    img = new Image();
                    img.onload = function() {
                        document.getElementById("sourceimg"+num).width = this.width;
                        document.getElementById("sourceimg"+num).height = this.height;
                        document.getElementById("source"+num).width = this.width;
                        document.getElementById("source"+num).height = this.height;
                        var canvas = document.getElementById("source"+num);
                        var ctx = canvas.getContext("2d");
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        var img = document.getElementById("sourceimg"+num);
                        ctx.drawImage(img, 0, 0);
                        imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
                        data = imgData.data;
                        pxls = [];
                        for (let i = 0; i < data.length; i += 4) {
                            var pixel = [];
                            pixel.push(data[i]);
                            pixel.push(data[i+1]);
                            pixel.push(data[i+2]);
                            pixel.push(data[i+3]);
                            pxls.push(pixel);
                        }
                        var pxlsarray = [[]];
                        for(i = 0;i<pxls.length;i++){
                            pxlsarray[pxlsarray.length-1].push(pxls[i]);
                            if(i<pxls.length-2){
                                if((i+1)%canvas.width==0 && i>canvas.width-10){
                                    pxlsarray.push([]);
                                }
                            }
                            
                        }
                    };
                    img.src = URL.createObjectURL(file);
                    document.getElementById("sourceimg"+num).src = URL.createObjectURL(file);
                }
                
            }
            document.getElementById("settings2").append(newel);
            var newel = document.createElement("div");
            newel.id = "moveimg"+num;
            newel.addEventListener("click",moveimgenable);
            newel.className = "moveimgclass txt";
            newel.style.left = 5+(num-1)*10+"%";
            var newelpar = document.createElement("p");            
            newelpar.innerHTML = "Move Image "+num;
            newelpar.className = "txt";
            newel.append(newelpar);
            document.getElementById("settings2").append(newel);
            newel = document.createElement("img");
            newel.id = "sourceimg"+num;
            newel.style.display = "none";
            document.getElementById("canvascontainer").append(newel);
            newel = document.createElement("canvas");
            newel.id = "source"+num;
            newel.className = "canvas";
            newel.style.position = "fixed";
            newel.style.left = parseInt(document.getElementById("source").style.left) + document.getElementById("source").width + "px";
            newel.style.top = document.getElementById("source").style.top;
            document.getElementById("canvascontainer").append(newel);
            img = new Image();
            img.onload = function() {
                document.getElementById("sourceimg"+num).width = this.width;
                document.getElementById("sourceimg"+num).height = this.height;
                document.getElementById("source"+num).width = this.width;
                document.getElementById("source"+num).height = this.height;
                var canvas = document.getElementById("source"+num);
                var ctx = canvas.getContext("2d");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                var img = document.getElementById("sourceimg"+num);
                ctx.drawImage(img, 0, 0);
                imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
                data = imgData.data;
                pxls = [];
                for (let i = 0; i < data.length; i += 4) {
                    var pixel = [];
                    pixel.push(data[i]);
                    pixel.push(data[i+1]);
                    pixel.push(data[i+2]);
                    pixel.push(data[i+3]);
                    pxls.push(pixel);
                }
                var pxlsarray = [[]];
                for(i = 0;i<pxls.length;i++){
                    pxlsarray[pxlsarray.length-1].push(pxls[i]);
                    if(i<pxls.length-2){
                        if((i+1)%canvas.width==0 && i>canvas.width-10){
                            pxlsarray.push([]);
                        }
                    }
                    
                }
            };
            img.src = URL.createObjectURL(file);
            document.getElementById("sourceimg"+num).src = URL.createObjectURL(file);
        }
        fsimgs[num] = document.getElementById("sourceimg"+num).src;
        
    }
    grabwfsmodels();

    
    document.getElementById("compare").style.left = document.getElementById("source").style.left;
    document.getElementById("compare").style.width = parseInt(document.getElementById("oldimg").width)-parseInt(document.getElementById("oldimg").width)/2 + "px";
    document.getElementById("compare").style.height = document.getElementById("oldimg").style.height;
    document.getElementById("sourceimg").onload = function(){
        document.getElementById("oldimg").width = document.getElementById("sourceimg").width;
        document.getElementById("oldimg").height = document.getElementById("sourceimg").height;
    }
    document.getElementById("source").style.top = "5px";
    document.getElementById("source").style.left = "5px";


}

function prepare(){
    fetch('http://localhost:5000/prepare'
    ).then(function (response) {
        responseClone = response.clone(); // 2
        return response.json();
    })
    .then(data => {
        if(data["exist"]!="yes"){
            document.getElementById("sourceimg").src = "";
            document.getElementById("uploadcontainer").style.height = "100%";
            document.getElementById("uploadcontainer").style.visibility = "visible";
            document.getElementById("uploadcontainer").style.opacity = 1;
        }
        else{
            document.getElementById("uploadcontainer").style.visibility = "hidden";
            resetimg();
        }
    });
}

function setreftoggle(){
    setref = !setref;
    if(setref){
        document.getElementById("selector").style.border = "2px solid red";
        document.getElementById("setref").style.backgroundColor = "rgb(20,20,20)";
        document.getElementById("saveexit").style.visibility = "hidden";
        ll = true;
        lr = true;
        lt = true;
        lb = true;
        document.getElementById("lockleft").style.backgroundColor = "gray";
        document.getElementById("lockright").style.backgroundColor = "gray";
        document.getElementById("locktop").style.backgroundColor = "gray";
        document.getElementById("lockbottom").style.backgroundColor = "gray";
        document.getElementById("lockbox").style.backgroundColor = "rgb(20, 20, 20)";
    }
    else{
        document.getElementById("selector").style.border = "2px solid green";
        document.getElementById("setref").style.backgroundColor = "gray";
        ll = false;
        lr = false;
        lt = false;
        lb = false;
        document.getElementById("lockbox").style.backgroundColor = "gray";
    }
}
function resetimg(){
    var canvas = document.getElementById("source");
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var img = document.getElementById("sourceimg");
    ctx.drawImage(img, 0, 0);
}

function ffupdate(){
    savedata["ff"] = document.getElementById("ff").value;
}
function draw(){
    var c = document.getElementById("source");
    var ctx = c.getContext("2d");  
    var erasersize = 30
    try{
        erasersize = parseInt(document.getElementById("erasersize").value);
    }
    catch(err){
        console.log(err);
    }        
    if(eraseenableflag){
        
        var imgData = ctx.getImageData(Math.floor(mousex)-parseInt(document.getElementById('source').getBoundingClientRect()["left"]), Math.floor(mousey)-parseInt(document.getElementById('source').getBoundingClientRect()["top"]), erasersize,erasersize);
        var data = imgData.data;
        var pxls = [];
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = avg; // red
            data[i + 1] = avg; // green
            data[i + 2] = avg; // blue
            data[i + 3] = 0; // alpha
        }
        ctx.putImageData(imgData, (Math.floor(mousex)-parseInt(document.getElementById('source').getBoundingClientRect()["left"]))-(erasersize/2), (Math.floor(mousey)-parseInt(document.getElementById('source').getBoundingClientRect()["top"]))-(erasersize/2));
    }
    else if(drawenableflag){
        ctx.fillStyle = document.getElementById("clr").value+parseInt(document.getElementById("inprange").value).toString(16);
        ctx.fillRect((Math.floor(mousex)-parseInt(document.getElementById('source').getBoundingClientRect()["left"]))-(erasersize/2), (Math.floor(mousey)-parseInt(document.getElementById('source').getBoundingClientRect()["top"]))-(erasersize/2),erasersize/2,erasersize/2);
        
    }
    else{
        imgData = ctx.getImageData(0, 0, c.width,c.height);
        data = imgData.data;
        pxls = [];
        for (let i = 0; i < data.length; i += 4) {
            var pixel = [];
            pixel.push(data[i]);
            pixel.push(data[i+1]);
            pixel.push(data[i+2]);
            pixel.push(data[i+3]);
            pxls.push(pixel);
        }
        var pxlsarray = [[]];
        for(i = 0;i<pxls.length;i++){
            pxlsarray[pxlsarray.length-1].push(pxls[i]);
            if(i<pxls.length-2){
                if((i+1)%c.width==0 && i>c.width-10){
                    pxlsarray.push([]);
                }
            }
            
        }
        savedata["pxlsarray"] = pxlsarray;
    }
        
    
}
function drawenable(){
    if(drawflag){
        drawenableflag = !drawenableflag;
    }
    else{
        eraseenableflag = !eraseenableflag;
    }
}


function changesize(e){
    if(drawflag || eraseflag){
        if(e.deltaY < 0){
            document.getElementById("erasersize").value = parseInt(document.getElementById("erasersize").value) + 1;
        }
        else{
            if(parseInt(document.getElementById("erasersize").value)-1>0){
                document.getElementById("erasersize").value = parseInt(document.getElementById("erasersize").value) - 1;
            }
        }
    }
    else if(!moveimgenableflag){
        if(cropflag){
            if(cropcount==0){
                if(e.deltaY < 0 && parseInt(document.getElementById("selectorsize").value)+10<= parseInt(document.getElementById("source").width)){
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) + 10;
                    selectorsize = document.getElementById("selectorsize").value;
                    savedata["boxsz"] = selectorsize;
                    document.getElementById("selector").style.width = selectorsize+"px";                   
                }
                else if(e.deltaY < 0 && parseInt(document.getElementById("selectorsize").value)+10> parseInt(document.getElementById("source").width)){
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("source").width);
                    selectorsize = document.getElementById("selectorsize").value;
                    savedata["boxsz"] = selectorsize;
                    document.getElementById("selector").style.width = selectorsize+"px";   
                }
                else{
                    if(parseInt(document.getElementById("selectorsize").value)-1>0){
                        document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) - 10;
                        selectorsize = document.getElementById("selectorsize").value;
                        savedata["boxsz"] = selectorsize;
                        document.getElementById("selector").style.width = selectorsize+"px";  
                    }
                }
                document.getElementById("hinttext").innerHTML = "Horizontal: ";
                document.getElementById("hintinp").value = selectorsize;
            }
            else if(cropcount==1){
                if(e.deltaY < 0 && parseInt(document.getElementById("selectorsize").value)+10<= parseInt(document.getElementById("source").height)){
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) + 10;
                    selectorsize = document.getElementById("selectorsize").value;
                    savedata["boxsz"] = selectorsize;
                    document.getElementById("selector").style.height = selectorsize+"px";                   
                }
                else if(e.deltaY < 0 && parseInt(document.getElementById("selectorsize").value)+10> parseInt(document.getElementById("source").height)){
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("source").height);
                    selectorsize = document.getElementById("selectorsize").value;
                    savedata["boxsz"] = selectorsize;
                    document.getElementById("selector").style.height = selectorsize+"px"; 
                }
                else{
                    if(parseInt(document.getElementById("selectorsize").value)-10>0){
                        document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) - 10;
                        selectorsize = document.getElementById("selectorsize").value;
                        savedata["boxsz"] = selectorsize;
                        document.getElementById("selector").style.height = selectorsize+"px";  
                    }
                }
                document.getElementById("hinttext").innerHTML = "Vertical: ";
                document.getElementById("hintinp").value = selectorsize;
            }
        }
        else{
            if(e.deltaY < 0){
                if(!lockflag){
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) + 64;
                    changeselectorsize(event);
                }
                else if(lockflag && parseInt(document.getElementById("selectorsize").value)+64<= parseInt(document.getElementById("source").width) && parseInt(document.getElementById("selectorsize").value)+64<= parseInt(document.getElementById("source").height)){
                    document.getElementById("selector").style.left = parseFloat(document.getElementById("source").style.left) + Math.floor(selectorsize/2)+"px";
                    document.getElementById("selector").style.top = parseFloat(document.getElementById("source").style.top) + Math.floor(selectorsize/2)+"px";
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) + 64;
                    changeselectorsize(event);
                    
                } 
            }
            else{
                if(parseInt(document.getElementById("selectorsize").value)-64>0){
                    document.getElementById("selectorsize").value = parseInt(document.getElementById("selectorsize").value) - 64;
                    changeselectorsize(event);
                }
            }
        }
    }
    else if(moveimgenableflag){
        var canvas = document.getElementById("source"+imagetomove);
        var ctx = canvas.getContext("2d");
        if(e.deltaY < 0){      
            if(parseFloat(document.getElementById("source"+imagetomove).width)>parseFloat(document.getElementById("source"+imagetomove).height)){
                let r =  parseFloat(document.getElementById("source"+imagetomove).width)/parseFloat(document.getElementById("source"+imagetomove).height);
                document.getElementById("source"+imagetomove).width = parseFloat(document.getElementById("source"+imagetomove).width) + 10;
                document.getElementById("source"+imagetomove).height = parseFloat(document.getElementById("source"+imagetomove).height) + 10/r;
            }
            else if(parseFloat(document.getElementById("source"+imagetomove).width)<parseFloat(document.getElementById("source"+imagetomove).height)){ 
                let r =  parseFloat(document.getElementById("source"+imagetomove).height)/parseFloat(document.getElementById("source"+imagetomove).width);
                document.getElementById("source"+imagetomove).width = parseFloat(document.getElementById("source"+imagetomove).width) + 10/r;
                document.getElementById("source"+imagetomove).height = parseFloat(document.getElementById("source"+imagetomove).height) + 10;
            }
            else{
                document.getElementById("source"+imagetomove).width = parseFloat(document.getElementById("source"+imagetomove).width) + 10;
                document.getElementById("source"+imagetomove).height = parseFloat(document.getElementById("source"+imagetomove).height) + 10;
            }
            
        }  
        else if(e.deltaY > 0){
            if(parseFloat(document.getElementById("source"+imagetomove).width)>parseFloat(document.getElementById("source"+imagetomove).height)){
                let r =  parseFloat(document.getElementById("source"+imagetomove).width)/parseFloat(document.getElementById("source"+imagetomove).height);
                document.getElementById("source"+imagetomove).width = parseFloat(document.getElementById("source"+imagetomove).width) - 10;
                document.getElementById("source"+imagetomove).height = parseFloat(document.getElementById("source"+imagetomove).height) - 10/r;
            }
            else if(parseFloat(document.getElementById("source"+imagetomove).width)<parseFloat(document.getElementById("source"+imagetomove).height)){
                let r =  parseFloat(document.getElementById("source"+imagetomove).height)/parseFloat(document.getElementById("source"+imagetomove).width);
                document.getElementById("source"+imagetomove).width = parseFloat(document.getElementById("source"+imagetomove).width) - 10/r;
                document.getElementById("source"+imagetomove).height = parseFloat(document.getElementById("source"+imagetomove).height) - 10;
            }
            else{
                document.getElementById("source"+imagetomove).width = parseFloat(document.getElementById("source"+imagetomove).width) - 10;
                document.getElementById("source"+imagetomove).height = parseFloat(document.getElementById("source"+imagetomove).height) - 10;
            }
            
        
        }    
        if(e.deltaX < 0){      
            if(imgflip==false){
                ctx.translate(canvas.width, 0);
                imgflip = true;
                ctx.scale(-1, 1);
        
            }
            else{
                imgflip = false;
                
            }         
            
        
        }  
        else if(e.deltaX > 0){      
            if(imgflip==false){
                ctx.translate(0, canvas.height);
                imgflip = true;
                ctx.scale(1, -1);
        
            }            
            else{
                imgflip = false;
                
            }         
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        var img = new Image;
        if(imagetomove==""){
            img.src = fsimgs["main"];
        }
        else{
            console.log(imagetomove);
            console.log(fsimgs);
            img.src = fsimgs[imagetomove];
        }
        
        img.onload = function(){
            ctx.drawImage(img, 0, 0, img.width,img.height,0, 0, canvas.width, canvas.height);
            document.getElementById("sourceimg"+imagetomove).src = canvas.toDataURL();
            document.getElementById("oldimg").width = canvas.width;
            document.getElementById("oldimg").height = canvas.height;
            document.getElementById("oldimg").src = canvas.toDataURL();

        }
           
    
    }
}

function erasemodeon(){
    if(eraseflag){
        document.getElementById("selector").style.visibility = "visible";
        document.getElementById("moveimglistner").style.visibility = "visible";
        document.getElementById("eraseon").style.backgroundColor = "gray";
        eraseflag = !eraseflag
    }
    else{
        if(drawflag==false && moveimgenableflag==false){
            document.getElementById("selector").style.visibility = "hidden";
            document.getElementById("moveimglistner").style.visibility = "hidden";
            document.getElementById("eraseon").style.backgroundColor = "rgb(20, 20, 20)";
            eraseflag = !eraseflag
        }
    }
}

function drawmodeon(){
    if(drawflag){
        document.getElementById("selector").style.visibility = "visible";
        document.getElementById("moveimglistner").style.visibility = "visible";
        document.getElementById("drawon").style.backgroundColor = "gray";
        drawflag = !drawflag
    }
    else{
        if(eraseflag==false && moveimgenableflag==false){
            document.getElementById("selector").style.visibility = "hidden";
            document.getElementById("moveimglistner").style.visibility = "hidden";
            document.getElementById("drawon").style.backgroundColor = "rgb(20, 20, 20)";
            drawflag = !drawflag
        }
    }
}

function saveimg(){
    var canvas = document.getElementById("source");
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var img = document.getElementById("sourceimg");
    ctx.drawImage(img, 0, 0);
    imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
    data = imgData.data;
    pxls = [];
    for (let i = 0; i < data.length; i += 4) {
        var pixel = [];
        pixel.push(data[i]);
        pixel.push(data[i+1]);
        pixel.push(data[i+2]);
        pixel.push(data[i+3]);
        pxls.push(pixel);
    }
    var pxlsarray = [[]];
    for(i = 0;i<pxls.length;i++){
        pxlsarray[pxlsarray.length-1].push(pxls[i]);
        if(i<pxls.length-2){
            if((i+1)%canvas.width==0 && i>canvas.width-10){
                pxlsarray.push([]);
            }
        }
        
    }
    savedata["pxlsarray"] = pxlsarray;
    fetch('http://localhost:5000/saveimg',{
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"pixels":savedata["pxlsarray"]})
    })
    .then(function (response){
        return response.json();
    })
    .then(data=>{
    });
}

function grabwfsmodels(){
    fetch('http://localhost:5000/grabwfsmodels')
    .then(function (response){
        return response.json();
    })
    .then(data=>{
        console.log(data);
        for(let i =0;i<data["wfs"].length;i++){
            var el = document.createElement("a");
            el.innerHTML =  data["wfs"][i];
            el.onclick = function(evt){
                selwf = evt.target.innerHTML;
                document.getElementById("wfbutton").innerHTML = evt.target.innerHTML;
                grabwfparams();
            }
            document.getElementById("wfscont").append(el);
        }
        for(let i =0;i<data["models"].length;i++){
            var el = document.createElement("a");
            el.innerHTML =  data["models"][i];
            el.onclick = function(evt){
                selmod = evt.target.innerHTML;
                document.getElementById("modelbutton").innerHTML = evt.target.innerHTML;
            }
            document.getElementById("modelscont").append(el);
        }
    });
}

function grabwfparams(){
    fetch('http://localhost:5000/grabwfparams', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"wf":selwf})
    })
    .then(function (response){
        return response.json();
    })
    .then(data=>{
        console.log(data);
        let childs = document.getElementById("paramscont");
        while (childs.firstChild){
            childs.removeChild(childs.lastChild);
        }
        for(let i =0;i<data["params"].length;i++){
            var el = document.createElement("a");
            el.innerHTML =  data["params"][i];
            el.onclick = function(evt){
                selparam = evt.target.innerHTML;
                document.getElementById("paramsbutton").innerHTML = evt.target.innerHTML;
            }
            document.getElementById("paramscont").append(el);
        }
    });
}

function generate(){
    fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"wf":selwf,"model":selmod,"params":params})
    })
    .then(function (response){
        return response.json();
    })
    .then(data=>{
    });
}

function cancelgen(){
    fetch('http://localhost:5000/cancelgen')
    .then(function (response){
        return response.json();
    })
    .then(data=>{
    });
}
function saved(savedata){
    var responseClone; // 1    
    var canvas = document.getElementById("source");
    var ctx = canvas.getContext("2d");
    imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
    data = imgData.data;
    pxls = [];
    for (let i = 0; i < data.length; i += 4) {
        var pixel = [];
        pixel.push(data[i]);
        pixel.push(data[i+1]);
        pixel.push(data[i+2]);
        pixel.push(data[i+3]);
        pxls.push(pixel);
    }
    var pxlsarray = [[]];
    for(i = 0;i<pxls.length;i++){
        pxlsarray[pxlsarray.length-1].push(pxls[i]);
        if(i<pxls.length-2){
            if((i+1)%canvas.width==0 && i>canvas.width-10){
                pxlsarray.push([]);
            }
        }
        
    }
    savedata["pxlsarray"] = pxlsarray;
    fetch('http://localhost:5000/savedata', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"savedata":savedata,"selectorsize":selectorsize,"taesd":taesd,"img2img":img2img})
    })
    .then(function (response) {
        responseClone = response.clone(); // 2
        return response.json();
    })
    .then(data => {
        document.getElementById("settings").style.visibility = "hidden";
        document.getElementById("selector").style.visibility = "hidden";
        document.getElementById("saveexit").style.visibility = "hidden";
        document.getElementById("undo").style.visibility = "hidden";
        document.getElementById("moveimglistner").style.visibility = "hidden";
        document.getElementById("cancel").style.visibility = "visible";
        img2img = data["img2img"];
        if(genflag==false){
            generate();
            genflag = true;
        }
        if(data["taesd"]=="no"){
            if(data["flag"]=="False"){
                if(document.getElementById("taesdcanvas")){
                    var canvas = document.getElementById("taesdcanvas");
                }
                else{
                    var canvas = document.createElement("canvas");
                    canvas.id = "taesdcanvas";
                    canvas.width = data["width"];
                    canvas.height = data["height"];
                    canvas.style.left = boxleft;
                    canvas.style.top = boxtop;
                    //var maincanv = document.getElementById("source");
                    //document.getElementById("source").remove();
                    console.log(data["img2img"]);
                    if(data["img2img"]=="img2img"){
                        img2img = "img2img";
                        canvas.style.zIndex = "4";
                    }
                    document.getElementById("canvascontainer").append(canvas);
                    //document.getElementById("canvascontainer").append(maincanv);
                }
                img = new Image();
                img.onload = function() {
                    canvas = document.getElementById("taesdcanvas");
                    var ctx = canvas.getContext("2d");
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                }
                img.src = data["img"];

                
            }
            else{
                if(canvasflag){
                    backup["width"] = document.getElementById("source").width;
                    backup["height"] = document.getElementById("source").height;
                    backup["left"] = document.getElementById("source").style.left;
                    backup["top"] = document.getElementById("source").style.top;
                    backup["img"] = document.getElementById("sourceimg").src;
    
                    document.getElementById("sourceimg").width = data["width"];
                    document.getElementById("sourceimg").height =data["height"];
                    document.getElementById("source").width = data["width"];
                    document.getElementById("source").height = data["height"];
                }
                canvasflag = false;
    
    
    
                document.getElementById("sourceimg").src = data["img"];
                img = new Image();
                img.onload = function() {            
                    var canvas = document.getElementById("source");
                    var ctx = canvas.getContext("2d");
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    var img = document.getElementById("sourceimg");
                    ctx.drawImage(img, 0, 0);
                    imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
                    data = imgData.data;
                    pxls = [];
                    for (let i = 0; i < data.length; i += 4) {
                        var pixel = [];
                        pixel.push(data[i]);
                        pixel.push(data[i+1]);
                        pixel.push(data[i+2]);
                        pixel.push(data[i+3]);
                        pxls.push(pixel);
                    }
                    var pxlsarray = [[]];
                    for(i = 0;i<pxls.length;i++){
                        pxlsarray[pxlsarray.length-1].push(pxls[i]);
                        if(i<pxls.length-2){
                            if((i+1)%canvas.width==0 && i>canvas.width-10){
                                pxlsarray.push([]);
                            }
                        }
                        
                    }
                    savedata["pxlsarray"] = pxlsarray;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                    fsimgs["main"] = canvas.toDataURL();
                }
                img.src = data["img"];

            }
            
        }
        
        if(data["flag"]=="False"){
            taesd = "true";
            setTimeout(() => {
                saved(savedata);
            }, 100);
        }
        else{
            document.getElementById("settings").style.visibility = "visible";
            document.getElementById("selector").style.visibility = "visible";
            document.getElementById("undo").style.visibility = "visible";
            document.getElementById("moveimglistner").style.visibility = "visible";
            document.getElementById("cancel").style.visibility = "hidden";
            taesd = "false";
            canvasflag = true;
            genflag = false;
            boxleft = "";
            boxtop = "";
            params = {};
            if(parseInt(data["data"]["additionaldims"]["left"])!=0){
                document.getElementById("source").style.left = parseFloat(document.getElementById("source").style.left) - parseInt(data["data"]["additionaldims"]["left"]) +"px";
                document.getElementById("compare").style.left = parseFloat(document.getElementById("source").style.left) - parseInt(data["data"]["additionaldims"]["left"]) +"px";
            }
            else{
                document.getElementById("compare").style.left = parseFloat(document.getElementById("source").style.left)+"px";
            }
            if(parseInt(data["data"]["additionaldims"]["top"])!=0){
                document.getElementById("source").style.top = parseFloat(document.getElementById("source").style.top) - parseInt(data["data"]["additionaldims"]["top"]) +"px";
                document.getElementById("compare").style.top = parseFloat(document.getElementById("source").style.top) - parseInt(data["data"]["additionaldims"]["top"]) +"px"
            }
            else{
                document.getElementById("compare").style.top = parseFloat(document.getElementById("source").style.top)+"px";
            }
            document.getElementById("taesdcanvas").remove();
            savedata["crpdims"] = {
                "left":0,
                "top": 0,
                "right":0,
                "bottom":0
            };
            
            savedata["additionaldims"] = {
                "left":0,
                "top": 0,
                "right":0,
                "bottom":0
            };
            
            document.getElementById("paramsbutton").innerHTML = "Change Parameter";
            document.getElementById("paramvalue").value = "";

            //document.getElementById("sourceunder").style.visibility = "hidden";
        }
        //refresh(savedata);


    }, function (rejectionReason) { // 3
        console.log('Error parsing JSON from response:', rejectionReason, responseClone); // 4
        responseClone.text() // 5
        .then(function (bodyText) {
            console.log('Received the following instead of valid JSON:', bodyText); // 6
        });
    });
    
    //document.getElementById("msg").style.visibility = "visible";
    //document.getElementById("container").remove();
    
}
function mousecoordinates(event){
    prevmousex = mousex;
    prevmousey =mousey;
    mousex = event.clientX;
    mousey = event.clientY;
    
    
}
function selector(){
    var selec = document.getElementById("selector");
    selec.style.left = Math.floor(mousex)+"px";
    selec.style.top = Math.floor(mousey)+"px";
    if(lockflag==false){
        if((parseFloat(document.getElementById("selector").style.left)-(selectorsize/2))<0){
            selec.style.left = Math.floor(selectorsize/2)+"px";
        }
        if((parseFloat(document.getElementById('selector').getBoundingClientRect()["right"])-(selectorsize/2))>parseInt(screen.availWidth)){
            selec.style.left = parseInt(screen.availWidth)-selectorsize+"px";
        }
        if((parseFloat(document.getElementById("selector").style.top)-(selectorsize/2))<0){
            selec.style.top = Math.floor(selectorsize/2)+"px";
        }
        if((parseFloat(document.getElementById('selector').getBoundingClientRect()["bottom"])-(selectorsize/2))>parseInt(screen.availHeight)){
            selec.style.bottom = parseInt(screen.availHeight)-selectorsize+"px";
        }
    }
    
    if(cropflag){
        if(ll){
            if((parseFloat(document.getElementById("selector").style.left)-parseInt(parseInt(window.getComputedStyle(selec).getPropertyValue("width"))/2))<parseFloat(document.getElementById("source").style.left)){
                selec.style.left = parseFloat(document.getElementById("source").style.left) + parseInt(parseInt(window.getComputedStyle(selec).getPropertyValue("width"))/2)+"px";
            }
        }
        if(lr){
            if((parseFloat(document.getElementById('selector').getBoundingClientRect()["right"]))>parseFloat(document.getElementById('source').getBoundingClientRect()["right"])){
                selec.style.left = parseFloat(document.getElementById("source").style.left)+parseFloat(document.getElementById("source").width) - parseInt(parseInt(window.getComputedStyle(selec).getPropertyValue("width"))/2)+"px";
            }
        }
        if(lt){
            if((parseFloat(document.getElementById("selector").style.top)-parseInt(parseInt(window.getComputedStyle(selec).getPropertyValue("height"))/2))<parseFloat(document.getElementById("source").style.top)){
                selec.style.top = parseFloat(document.getElementById("source").style.top) + parseInt(parseInt(window.getComputedStyle(selec).getPropertyValue("height"))/2)+"px";
                console.log((Math.floor(Math.floor(window.getComputedStyle(selec).getPropertyValue("height"))/2)));
            }
        }
        if(lb){
            if((parseFloat(document.getElementById('selector').getBoundingClientRect()["bottom"]))>parseFloat(document.getElementById('source').getBoundingClientRect()["bottom"])){
                selec.style.top = parseFloat(document.getElementById("source").style.top)+parseFloat(document.getElementById("source").height) - parseInt(parseInt(window.getComputedStyle(selec).getPropertyValue("height"))/2)+"px";
            }
        }
    }
    else{
        if(ll){
            if((parseFloat(document.getElementById("selector").style.left)-(selectorsize/2))<parseFloat(document.getElementById("source").style.left)){
                selec.style.left = parseFloat(document.getElementById("source").style.left) + Math.floor(selectorsize/2)+1+"px";
            }
        }
        if(lr){
            if((parseFloat(document.getElementById('selector').getBoundingClientRect()["right"]))>parseFloat(document.getElementById('source').getBoundingClientRect()["right"])){
                selec.style.left = parseFloat(document.getElementById("source").style.left)+parseFloat(document.getElementById("source").width) - Math.floor(selectorsize/2) +"px";
            }
        }
        if(lt){
            if((parseFloat(document.getElementById("selector").style.top)-(selectorsize/2))<parseFloat(document.getElementById("source").style.top)){
                selec.style.top = parseFloat(document.getElementById("source").style.top) + Math.floor(selectorsize/2)+"px";
            }
        }
        if(lb){
            if((parseFloat(document.getElementById('selector').getBoundingClientRect()["bottom"]))>parseFloat(document.getElementById('source').getBoundingClientRect()["bottom"])){
                selec.style.top = parseFloat(document.getElementById("source").style.top)+parseFloat(document.getElementById("source").height) - Math.floor(selectorsize/2)+"px";
            }
        }
    }
     
    
    
}


function imgmover(){
    if(!compareflag){    
        var selec = document.getElementById("source"+imagetomove);
        console.log(selec.id);
        if(mousex>prevmousex){
            selec.style.left = parseInt(selec.getBoundingClientRect()["left"])+10+"px";
        }
        else if(mousex<prevmousex){
            selec.style.left = parseInt(selec.getBoundingClientRect()["left"])-10+"px";
        }
        if(mousey>prevmousey){
            selec.style.top = parseInt(selec.getBoundingClientRect()["top"])+10+"px";
        }
        else if(mousey<prevmousey){
            selec.style.top = parseInt(selec.getBoundingClientRect()["top"])-10+"px";
        } 
        let addleft = 0;
        let addtop = 0; 
        try{
            if(parseInt(data["data"]["additionaldims"]["left"])!=0){
                addleft = parseInt(data["data"]["additionaldims"]["left"]) +"px";
            }
            if(parseInt(data["data"]["additionaldims"]["top"])!=0){
                addtop =  parseInt(data["data"]["additionaldims"]["top"]) +"px";
            }
        } 
        catch(error){

        }
        document.getElementById("compare").style.left = parseInt(selec.style.left)-addleft+"px";
        document.getElementById("compare").style.top = parseInt(selec.style.top)-addtop+"px";
        document.getElementById("hint").style.left = parseInt(selec.style.left) +"px";
        document.getElementById("hint").style.top = parseInt(selec.style.top) - 100 +"px";
    } 
    else{
        if(mousex>prevmousex){
            if(parseInt(document.getElementById("compare").style.width) + 10<=parseInt(document.getElementById("oldimg").width)){
                document.getElementById("compare").style.width = parseInt(document.getElementById("compare").style.width) + 10 +"px";
            }
            else{
                document.getElementById("compare").style.width = parseInt(document.getElementById("oldimg").width) +"px";
            }
        }
        else if(mousex<prevmousex){
            if(parseInt(document.getElementById("compare").style.width) - 10>=0){
                document.getElementById("compare").style.width = parseInt(document.getElementById("compare").style.width) - 10 +"px";
            }
            else{
                document.getElementById("compare").style.width = "0px";
            }
        }
    }
}
function snapshot(){
    imagedims["width"] = parseInt(document.getElementById('source').width);
    imagedims["height"] = parseInt(document.getElementById('source').height);
    if(cropflag){
        let oldleft = savedata["crpdims"]["left"];
        let oldright = savedata["crpdims"]["right"];
        let oldtop = savedata["crpdims"]["top"];
        let oldbottom = savedata["crpdims"]["bottom"];
        savedata["crpdims"]["left"] = (parseFloat(document.getElementById("selector").style.left)-(parseInt(parseInt(window.getComputedStyle(document.getElementById("selector")).getPropertyValue("width"))/2)))-document.getElementById('source').getBoundingClientRect()["left"];
        savedata["crpdims"]["right"] = (parseInt(parseInt(window.getComputedStyle(document.getElementById("selector")).getPropertyValue("width"))/1)) + ((parseFloat(document.getElementById("selector").style.left)-(parseInt(parseInt(window.getComputedStyle(document.getElementById("selector")).getPropertyValue("width"))/2)))-parseFloat(document.getElementById('source').getBoundingClientRect()["left"]));
        savedata["crpdims"]["top"] = (parseFloat(document.getElementById("selector").style.top)-(parseInt(parseInt(window.getComputedStyle(document.getElementById("selector")).getPropertyValue("height"))/2)))-document.getElementById('source').getBoundingClientRect()["top"];
        savedata["crpdims"]["bottom"] = (((parseFloat(document.getElementById("selector").style.top)-(parseInt(parseInt(window.getComputedStyle(document.getElementById("selector")).getPropertyValue("height"))/2)))-document.getElementById('source').getBoundingClientRect()["top"])+(parseInt(parseInt(window.getComputedStyle(document.getElementById("selector")).getPropertyValue("height"))/1)));
        if(cropcount==1){
            var canvas = document.getElementById("source");
            var ctx = canvas.getContext("2d");
            let tempcanvas = document.createElement("canvas");
            tempcanvas.width = canvas.width;
            tempcanvas.height = canvas.height;
            document.getElementById("canvascontainer").append(tempcanvas);
            let tempctx = tempcanvas.getContext("2d");
            tempctx.drawImage(canvas, 0, 0);
            let cw = tempcanvas.width;
            let cl = 0;
            let ch = tempcanvas.height;
            let ct = 0;
            canvas.width = savedata["crpdims"]["right"]-savedata["crpdims"]["left"];
            canvas.height = savedata["crpdims"]["bottom"]-savedata["crpdims"]["top"];
            console.log(savedata["crpdims"]["right"]-savedata["crpdims"]["left"]);
            ctx.drawImage(tempcanvas, savedata["crpdims"]["left"], savedata["crpdims"]["top"], savedata["crpdims"]["right"]-savedata["crpdims"]["left"], savedata["crpdims"]["bottom"]-savedata["crpdims"]["top"], 0, 0, canvas.width, canvas.height); 
            cropcount = 0;
            cropflag = false;
            lockflag = false;
            ll = false;
            lr = false;
            lt = false;
            lb = false;
            changeselectorsize(event);
            tempcanvas.remove();
            document.getElementById("settings").style.visibility = "visible";
            document.getElementById("undo").style.visibility = "visible";
            document.getElementById("sourceimg").src = canvas.toDataURL();
            document.getElementById("selector").style.border = "2px solid green";
            fsimgs["main"] = document.getElementById("sourceimg").src
            document.getElementById("oldimg").src = document.getElementById("sourceimg").src;
            document.getElementById("sourceimg").onload = function(){
                document.getElementById("oldimg").width = document.getElementById("source").width;
                document.getElementById("oldimg").height = document.getElementById("source").height;
            }
            document.getElementById("hint").style.visibility = "hidden";
            return 0;
        }
        else{
            cropcount = 1;
            document.getElementById("hinttext").innerHTML = "Vertical: ";
            document.getElementById("hintinp").value = selectorsize;
        }
    }
    if(!setref){
        savedata["crpdims"] = {
            "left":0,
            "top": 0,
            "right":0,
            "bottom":0
        };
        if(!cropflag){
            document.getElementById("saveexit").style.visibility = "visible";
        }
        if((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))<document.getElementById('source').getBoundingClientRect()["left"]){
            if(document.getElementById('selector').getBoundingClientRect()["right"]<document.getElementById('source').getBoundingClientRect()["right"]){
                savedata["additionaldims"]["left"] = document.getElementById('source').getBoundingClientRect()["left"]-(parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2));
                savedata["crpdims"]["left"] = 0;
                savedata["crpdims"]["right"] = parseFloat(selectorsize) - (parseFloat(document.getElementById('source').getBoundingClientRect()["left"])-(parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2)));
                console.log(savedata["crpdims"]["right"]);
            }
            else{
                savedata["crpdims"]["right"] = imagedims["width"];
                console.log(savedata["crpdims"]["right"]);
                savedata["additionaldims"]["left"] = document.getElementById('source').getBoundingClientRect()["left"]-(parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2));
                let lft = (parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"];
                savedata["additionaldims"]["right"] = parseFloat(selectorsize) - (imagedims["width"]-lft);
            }
        }
        else if((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))>=document.getElementById('source').getBoundingClientRect()["left"]){
            if(((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"])+parseFloat(selectorsize)>imagedims["width"]){
                let lft = (parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"];
                savedata["additionaldims"]["right"] = parseFloat(selectorsize) - (imagedims["width"]-lft);
                savedata["additionaldims"]["left"] = 0;
                savedata["crpdims"]["left"] = lft;
                savedata["crpdims"]["right"] = imagedims["width"];
                console.log(savedata["crpdims"]["right"]);
            }
            else{
        
                savedata["crpdims"]["left"] = (parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"];
                savedata["crpdims"]["right"] = parseFloat(selectorsize) + ((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-parseFloat(document.getElementById('source').getBoundingClientRect()["left"]));
                console.log(savedata["crpdims"]["right"]);
            }
        }
        
        if((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))<document.getElementById('source').getBoundingClientRect()["top"]){
            if(document.getElementById('selector').getBoundingClientRect()["bottom"]<document.getElementById('source').getBoundingClientRect()["bottom"]){
                savedata["additionaldims"]["top"] = document.getElementById('source').getBoundingClientRect()["top"]-(parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2));
                savedata["crpdims"]["top"] = 0;
                savedata["crpdims"]["bottom"] = parseFloat(selectorsize) - (document.getElementById('source').getBoundingClientRect()["top"]-(parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2)));
            }
            else{
                savedata["crpdims"]["bottom"] = imagedims["height"];
                savedata["additionaldims"]["top"] = document.getElementById('source').getBoundingClientRect()["top"]-(parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2));
                let lft = (parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"];
                savedata["additionaldims"]["bottom"] = parseFloat(selectorsize) - (imagedims["height"]-lft);
            }
        }
        else if((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))>=document.getElementById('source').getBoundingClientRect()["top"]){
            if(((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"])+parseFloat(selectorsize)>imagedims["height"]){
                let tp = (parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"];
                savedata["additionaldims"]["bottom"] = parseFloat(selectorsize) - (imagedims["height"]-tp);
                savedata["additionaldims"]["top"] = 0;
                savedata["crpdims"]["top"] = tp;
                savedata["crpdims"]["bottom"] = imagedims["height"];
            }
            else{
                savedata["crpdims"]["top"] = (parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"];
                savedata["crpdims"]["bottom"] = (((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"])+parseFloat(selectorsize));
            }
        }
        boxleft = parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2)+"px";
        boxtop = parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2) + "px";
    }
    else{
        savedata["crpdimsref"] = {
            "left":0,
            "top": 0,
            "right":0,
            "bottom":0
        };
        if((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))<document.getElementById('source').getBoundingClientRect()["left"]){
            if(document.getElementById('selector').getBoundingClientRect()["right"]<document.getElementById('source').getBoundingClientRect()["right"]){
                savedata["crpdimsref"]["left"] = 0;
                savedata["crpdimsref"]["right"] = parseFloat(selectorsize) - (parseFloat(document.getElementById('source').getBoundingClientRect()["left"])-(parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2)));
                console.log(savedata["crpdimsref"]["right"]);
            }
            else{
                savedata["crpdimsref"]["right"] = imagedims["width"];
                console.log(savedata["crpdimsref"]["right"]);
            }
        }
        else if((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))>=document.getElementById('source').getBoundingClientRect()["left"]){
            if(((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"])+parseFloat(selectorsize)>imagedims["width"]){
                let lft = (parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"];
                savedata["crpdimsref"]["left"] = lft;
                savedata["crpdimsref"]["right"] = imagedims["width"];
                console.log(savedata["crpdimsref"]["right"]);
            }
            else{
        
                savedata["crpdimsref"]["left"] = (parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["left"];
                savedata["crpdimsref"]["right"] = parseFloat(selectorsize) + ((parseFloat(document.getElementById("selector").style.left)-(parseFloat(selectorsize)/2))-parseFloat(document.getElementById('source').getBoundingClientRect()["left"]));
                console.log(savedata["crpdimsref"]["right"]);
            }
        }
        
        if((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))<document.getElementById('source').getBoundingClientRect()["top"]){
            if(document.getElementById('selector').getBoundingClientRect()["bottom"]<document.getElementById('source').getBoundingClientRect()["bottom"]){
                savedata["crpdimsref"]["top"] = 0;
                savedata["crpdimsref"]["bottom"] = parseFloat(selectorsize) - (document.getElementById('source').getBoundingClientRect()["top"]-(parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2)));
            }
            else{
                savedata["crpdimsref"]["bottom"] = imagedims["height"];
                
            }
        }
        else if((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))>=document.getElementById('source').getBoundingClientRect()["top"]){
            if(((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"])+parseFloat(selectorsize)>imagedims["height"]){
                let tp = (parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"];
                
                savedata["crpdimsref"]["top"] = tp;
                savedata["crpdimsref"]["bottom"] = imagedims["height"];
            }
            else{
                savedata["crpdimsref"]["top"] = (parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"];
                savedata["crpdimsref"]["bottom"] = (((parseFloat(document.getElementById("selector").style.top)-(parseFloat(selectorsize)/2))-document.getElementById('source').getBoundingClientRect()["top"])+parseFloat(selectorsize));
            }
        }
    }
    
}

function lockleft(){
    if(ll){
        document.getElementById("lockleft").style.backgroundColor = "gray";
    }
    else{
        document.getElementById("lockleft").style.backgroundColor = "rgb(20,20,20)";
    }
    if(!lockflag){
        ll = !ll;
    }
}
function lockright(){
    if(lr){
        document.getElementById("lockright").style.backgroundColor = "gray";
    }
    else{
        document.getElementById("lockright").style.backgroundColor = "rgb(20,20,20)";
    }
    if(!lockflag){
        lr = !lr;
    }
    
}
function locktop(){
    if(lt){
        document.getElementById("locktop").style.backgroundColor = "gray";
    }
    else{
        document.getElementById("locktop").style.backgroundColor = "rgb(20,20,20)";
    }
    if(!lockflag){
        lt = !lt;
    }
    
}
function lockbottom(){
    if(lb){
        document.getElementById("lockbottom").style.backgroundColor = "gray";
    }
    else{
        document.getElementById("lockbottom").style.backgroundColor = "rgb(20,20,20)";
    }
    if(!lockflag){
        lb = !lb;
    }

    
}
function lockbox(){
    if(!lockflag){
        ll = true;
        lr = true;
        lt = true;
        lb = true;
        document.getElementById("lockleft").style.backgroundColor = "gray";
        document.getElementById("lockright").style.backgroundColor = "gray";
        document.getElementById("locktop").style.backgroundColor = "gray";
        document.getElementById("lockbottom").style.backgroundColor = "gray";
        document.getElementById("lockbox").style.backgroundColor = "rgb(20, 20, 20)";
    }
    else{
        document.getElementById("lockbox").style.backgroundColor = "gray";
        ll = false;
        lr = false;
        lt = false;
        lb = false;
    }
    lockflag = !lockflag;
}

function grabselector(){
    if(lockflag==false){
        document.getElementById("selector").style.left = Math.floor(mousex)+"px";
        document.getElementById("selector").style.top = Math.floor(mousey)+"px";
    }
    

}

function moveimgenablefirst(){
    if(moveimgenableflag){
        document.getElementById("selector").style.visibility = "visible";        
        document.getElementById("moveimg").style.backgroundColor = "gray";         
        moveimgenableflag = !moveimgenableflag
        imagetomove = "";
    }
    else{
        if(drawflag==false && eraseflag==false && !compareflag){
            document.getElementById("selector").style.visibility = "hidden";
            document.getElementById("moveimg").style.backgroundColor = "rgb(20,20,20)";   
            moveimgenableflag = !moveimgenableflag;
            imagetomove = "";
        }
    }
}
function moveimgenable(t){
    if(moveimgenableflag){
        try{
            document.getElementById(t.currentTarget.id).style.backgroundColor = "gray"; 
        }
        catch(error){
            document.getElementById("moveimg1").style.backgroundColor = "gray";
        }      
        moveimgenableflag = !moveimgenableflag
        imagetomove = "";
    }
    else{
        if(drawflag==false && eraseflag==false){ 
            try{
                document.getElementById(t.currentTarget.id).style.backgroundColor = "rgb(20,20,20)";
                document.getElementById("canvascontainer").appendChild(document.getElementById("source"+t.currentTarget.id.match(/\d+/)[0]));
            }          
            catch(error){
                document.getElementById("moveimg1").style.backgroundColor = "rgb(20,20,20)";
                document.getElementById("canvascontainer").appendChild(document.getElementById("source"));
            }
            
            moveimgenableflag = !moveimgenableflag;
            try{
                imagetomove = String(t.currentTarget.id.match(/\d+/)[0]);
                comporder.push(t.currentTarget.id.match(/\d+/)[0]);
            }
            catch(error){
                imagetomove = "";
                comporder.push("");
            }
        }
    }
}

function compimage(){
    var order = [""];
    var coordinates = [[parseInt(document.getElementById('source').getBoundingClientRect()["left"]),parseInt(document.getElementById('source').getBoundingClientRect()["top"])]];
    var num = document.getElementById("settings2").children.length - 3;
    //document.getElementById("openimgadd").remove();
    var newel = document.createElement("input");
    newel.id = "openimgadd";
    newel.type = "file";
    newel.onchange =function(e) {
        document.getElementById("settings").style.visibility = "hidden";
        document.getElementById("undo").style.visibility = "hidden";
        var file, img;
        if ((file = this.files[0])) {
            comporder = [];
            document.getElementById("selector").style.visibility = "hidden";
            var num = (document.getElementById("canvascontainer").children.length /2)+1;
            num = String(num);
            document.getElementById("settings2").style.visibility = "visible";
            var newel = document.createElement("div");
            newel.id = "moveimg"+num;
            newel.addEventListener("click",moveimgenable);
            newel.className = "moveimgclass txt";
            newel.style.left = 5+(num-1)*10+"%";
            var newelpar = document.createElement("p");            
            newelpar.innerHTML = "Move Image "+num;
            newelpar.className = "txt";
            newel.append(newelpar);
            document.getElementById("settings2").append(newel);
            newel = document.createElement("img");
            newel.id = "sourceimg"+num;
            newel.style.display = "none";
            document.getElementById("canvascontainer").append(newel);
            newel = document.createElement("canvas");
            newel.id = "source"+num;
            newel.style.position = "fixed";
            newel.style.left = parseInt(document.getElementById("source").style.left) + document.getElementById("source").width + "px";
            newel.style.top = document.getElementById("source").style.top;
            document.getElementById("canvascontainer").append(newel);
            img = new Image();
            img.onload = function() {
                document.getElementById("sourceimg"+num).width = this.width;
                document.getElementById("sourceimg"+num).height = this.height;
                document.getElementById("source"+num).width = this.width;
                document.getElementById("source"+num).height = this.height;
                var canvas = document.getElementById("source"+num);
                var ctx = canvas.getContext("2d");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                var img = document.getElementById("sourceimg"+num);
                ctx.drawImage(img, 0, 0);
                imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
                data = imgData.data;
                pxls = [];
                for (let i = 0; i < data.length; i += 4) {
                    var pixel = [];
                    pixel.push(data[i]);
                    pixel.push(data[i+1]);
                    pixel.push(data[i+2]);
                    pixel.push(data[i+3]);
                    pxls.push(pixel);
                }
                var pxlsarray = [[]];
                for(i = 0;i<pxls.length;i++){
                    pxlsarray[pxlsarray.length-1].push(pxls[i]);
                    if(i<pxls.length-2){
                        if((i+1)%canvas.width==0 && i>canvas.width-10){
                            pxlsarray.push([]);
                        }
                    }
                    
                }
            };
            img.src = URL.createObjectURL(file);
            document.getElementById("sourceimg"+num).src = URL.createObjectURL(file);
        }
        document.getElementById("source"+num).className = "canvas";
        fsimgs[num] = document.getElementById("sourceimg"+num).src;
        document.getElementById("compare").style.left = document.getElementById("source").style.left;
        document.getElementById("compare").style.width = parseInt(document.getElementById("oldimg").width)-parseInt(document.getElementById("oldimg").width)/2 + "px";
        document.getElementById("compare").style.height = document.getElementById("oldimg").style.height;
        document.getElementById("sourceimg").onload = function(){
        document.getElementById("oldimg").width = document.getElementById("sourceimg").width;
        document.getElementById("oldimg").height = document.getElementById("sourceimg").height;
    }
        
    }
    document.getElementById("settings").append(newel);
    for(let i=1;i<num;i++){
        order.push(String(i+1));
    }
    console.log(order);
    if(comporder.length>0){
        comporder.reverse();
        order.reverse();
        for(let i=0;i<order.length;i++){
            comporder.push(order[i]);
        }
        order = [];
        for(let i=0;i<comporder.length;i++){
            if(!order.includes(comporder[i])){
                order.push(comporder[i]);
            }
        }
        
    }


    



    var leftest = parseInt(document.getElementById('source').getBoundingClientRect()["left"]);
    var topest = parseInt(document.getElementById('source').getBoundingClientRect()["top"]);
    var rightest = parseInt(document.getElementById('source').getBoundingClientRect()["right"]);
    var bottomest = parseInt(document.getElementById('source').getBoundingClientRect()["bottom"]);
    for(let i =2;i<num+1;i++){
        coordinates.push([parseInt(document.getElementById('source'+i).getBoundingClientRect()["left"]),parseInt(document.getElementById('source'+i).getBoundingClientRect()["top"])])
        if(parseInt(document.getElementById('source'+i).getBoundingClientRect()["left"])<leftest){
            leftest = parseInt(document.getElementById('source'+i).getBoundingClientRect()["left"])
        }
        if(parseInt(document.getElementById('source'+i).getBoundingClientRect()["top"])<topest){
            topest = parseInt(document.getElementById('source'+i).getBoundingClientRect()["top"])
        }
        if(parseInt(document.getElementById('source'+i).getBoundingClientRect()["right"])>rightest){
            rightest = parseInt(document.getElementById('source'+i).getBoundingClientRect()["right"])
        }
        if(parseInt(document.getElementById('source'+i).getBoundingClientRect()["bottom"])>bottomest){
            bottomest = parseInt(document.getElementById('source'+i).getBoundingClientRect()["bottom"])
        }
    }
    console.log(coordinates);
    console.log(order);
    var canvas = document.getElementById("source");
    var ctx = canvas.getContext("2d");
    canvas.width = rightest-leftest;
    canvas.height = bottomest-topest;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    order.reverse();
    for(let i = 0;i<order.length;i++){
        if(order[i]==""){
            var img = document.getElementById("sourceimg");
            ctx.drawImage(img, coordinates[0][0]-leftest, coordinates[0][1]-topest);
        }
        else{
            img = document.getElementById("sourceimg"+order[i]);
            ctx.drawImage(img, coordinates[parseInt(order[i])-1][0]-leftest, coordinates[parseInt(order[i])-1][1]-topest);
            document.getElementById("sourceimg"+order[i]).remove();
            document.getElementById("source"+order[i]).remove();
            document.getElementById("moveimg"+order[i]).remove();
        }
    }
    document.getElementById("sourceimg").src = canvas.toDataURL();
    imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
    data = imgData.data;
    pxls = [];
    for (let i = 0; i < data.length; i += 4) {
        var pixel = [];
        pixel.push(data[i]);
        pixel.push(data[i+1]);
        pixel.push(data[i+2]);
        pixel.push(data[i+3]);
        pxls.push(pixel);
    }
    var pxlsarray = [[]];
    for(i = 0;i<pxls.length;i++){
        pxlsarray[pxlsarray.length-1].push(pxls[i]);
        if(i<pxls.length-2){
            if((i+1)%canvas.width==0 && i>canvas.width-10){
                pxlsarray.push([]);
            }
        }
        
    }
    savedata["pxlsarray"] = pxlsarray;
    document.getElementById("settings2").style.visibility="hidden";
    document.getElementById("selector").style.visibility = "visible";
    document.getElementById("sourceimg").src = canvas.toDataURL();
    document.getElementById("oldimg").src = document.getElementById("sourceimg").src;
    document.getElementById("sourceimg").onload = function(){
        document.getElementById("oldimg").width = document.getElementById("source").width;
        document.getElementById("oldimg").height = document.getElementById("source").height;
    }
    imagetomove = "";
    fsimgs = {};
    fsimgs["main"] = document.getElementById("sourceimg").src;
    let childs = document.getElementById("settings").children;
    let numoia = 0;
    for (let x = 0;x<childs.length;x++){
        if(childs[x].id=="openimgadd"){
            numoia++;
        }
    }
    numoia-=1;
    for (let x = 0;x<childs.length;x++){
        if(childs[x].id=="openimgadd"){
            if(numoia>0){
                childs[x].remove();
            }
        }
    }
    document.getElementById("settings").style.visibility = "visible";
    document.getElementById("undo").style.visibility = "visible";

}
function moveimg(){    
    if(clickflag && (moveimgenableflag || compareflag)){
        document.getElementById("moveimglistner").removeEventListener("mousemove", imgmover);
        clickflag = !clickflag
    }
    else{   
        document.getElementById("moveimglistner").addEventListener("mousemove", imgmover);
        clickflag = !clickflag
    }
}
function changeselectorsize(evt){
    if(evt.target.id=="hintinp"){
        var elem = evt.target;
        selectorsize = elem.value;
        savedata["boxsz"] = selectorsize;
        if(cropcount==1){
            document.getElementById("selector").style.height = selectorsize+"px";
        }
        else{
            document.getElementById("selector").style.width = selectorsize+"px";
        }
               
    }
    else{
        var elem = document.getElementById("selectorsize");
        selectorsize = elem.value;
        savedata["boxsz"] = selectorsize;
        document.getElementById("selector").style.width = selectorsize+"px";
        document.getElementById("selector").style.height = selectorsize+"px";
    }
}

function savejson(){
    document.getElementById("oldimg").src = document.getElementById("sourceimg").src;
    document.getElementById("sourceimg").onload = function(){
        document.getElementById("oldimg").width = document.getElementById("source").width;
        document.getElementById("oldimg").height = document.getElementById("source").height;
    }
    var canvas = document.getElementById("source");
    var ctx = canvas.getContext("2d");
    let tempcanvas = document.createElement("canvas");
    tempcanvas.width = canvas.width+savedata["additionaldims"]["left"]+savedata["additionaldims"]["right"];
    tempcanvas.height = canvas.height+savedata["additionaldims"]["top"]+savedata["additionaldims"]["bottom"];
    document.getElementById("canvascontainer").append(tempcanvas);
    let tempctx = tempcanvas.getContext("2d");
    tempctx.drawImage(canvas, savedata["additionaldims"]["left"], savedata["additionaldims"]["top"]);    
    document.getElementById("oldimg").src = tempcanvas.toDataURL();
    tempcanvas.remove();
    saved(savedata);
    
}

function reset(){
    var selec = document.getElementById("selector");
    selec.style.left = mousex+"px";
    selec.style.top = mousey+"px";
}

function cropimg(){
    cropflag = true;
    document.getElementById("hint").style.visibility = "visible";
    document.getElementById("selector").style.border = "2px solid orange";
    document.getElementById("settings").style.visibility = "hidden";
    document.getElementById("undo").style.visibility = "hidden";
    savedata["crpdims"] = {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    };
    ll = true;
    lr = true;
    lt = true;
    lb = true;
    lockflag = true;
    document.getElementById("saveexit").style.visibility = "hidden";
    document.getElementById("hinttext").innerHTML = "Horizontal: ";
    document.getElementById("hintinp").value = selectorsize;

}

function compare(){
    if(!moveimgenableflag){
        compareflag = !compareflag;
        if(!compareflag){
            document.getElementById("compare").style.visibility = "hidden";
            document.getElementById("selector").style.visibility = "visible";        
            document.getElementById("compbtn").style.backgroundColor = "gray";
        }
        else{
            document.getElementById("compare").style.visibility = "visible";
            document.getElementById("selector").style.visibility = "hidden";
            document.getElementById("compbtn").style.backgroundColor = "rgb(20,20,20)";  
        }
    }
}

function resetref(){
    savedata["crpdimsref"] = {
        "left":0,
        "top": 0,
        "right":0,
        "bottom":0
    };
}

function fit(){
    var canvas = document.getElementById("source"+imagetomove);
    var ctx = canvas.getContext("2d");
    if(parseFloat(document.getElementById("source"+imagetomove).width)>parseFloat(document.getElementById("source"+imagetomove).height)){
        let r =  parseFloat(document.getElementById("source"+imagetomove).width)/parseFloat(document.getElementById("source"+imagetomove).height);
        document.getElementById("source"+imagetomove).width = 512;
        document.getElementById("source"+imagetomove).height = 512/r;
    }
    else if(parseFloat(document.getElementById("source"+imagetomove).width)<parseFloat(document.getElementById("source"+imagetomove).height)){
        let r =  parseFloat(document.getElementById("source"+imagetomove).height)/parseFloat(document.getElementById("source"+imagetomove).width);
        document.getElementById("source"+imagetomove).width = 512/r;
        document.getElementById("source"+imagetomove).height = 512;
    }
    else{
        document.getElementById("source"+imagetomove).width = 512;
        document.getElementById("source"+imagetomove).height = 512;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var img = new Image;
    img.src = fsimgs["main"];  
    img.onload = function(){
        ctx.drawImage(img, 0, 0, img.width,img.height,0, 0, canvas.width, canvas.height);
        document.getElementById("sourceimg"+imagetomove).src = canvas.toDataURL();
        document.getElementById("oldimg").width = canvas.width;
        document.getElementById("oldimg").height = canvas.height;
        document.getElementById("oldimg").src = canvas.toDataURL();

    }
}

function uploadnewimage(){
    document.getElementById("uploadcontainer").style.visibility = "visible";
    var file, img;   
    if ((file = this.files[0])) {
        img = new Image();
        img.onload = function() {
            document.getElementById("sourceimg").width = this.width;
            document.getElementById("sourceimg").height = this.height;
            document.getElementById("source").width = this.width;
            document.getElementById("source").height = this.height;
            var canvas = document.getElementById("source");
            var ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            var img = document.getElementById("sourceimg");
            ctx.drawImage(img, 0, 0);
            imgData = ctx.getImageData(0, 0, canvas.width,canvas.height);
            data = imgData.data;
            pxls = [];
            for (let i = 0; i < data.length; i += 4) {
                var pixel = [];
                pixel.push(data[i]);
                pixel.push(data[i+1]);
                pixel.push(data[i+2]);
                pixel.push(data[i+3]);
                pxls.push(pixel);
            }
            var pxlsarray = [[]];
            for(i = 0;i<pxls.length;i++){
                pxlsarray[pxlsarray.length-1].push(pxls[i]);
                if(i<pxls.length-2){
                    if((i+1)%canvas.width==0 && i>canvas.width-10){
                        pxlsarray.push([]);
                    }
                }
                
            }
            savedata["pxlsarray"] = pxlsarray;
            console.log(savedata["pxlsarray"]);
            backup["img"] = document.getElementById("sourceimg").src;
        };
        img.src = URL.createObjectURL(file);
        document.getElementById("sourceimg").src = URL.createObjectURL(file);
        document.getElementById("oldimg").src = URL.createObjectURL(file);
        document.getElementById("uploadcontainer").style.visibility = "hidden";
        


    }
    fsimgs["main"] = document.getElementById("sourceimg").src;
}

