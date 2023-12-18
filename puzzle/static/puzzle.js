var pieces = {};
var validate = [];
var ratio = 0;
var size = 10;
var mousex = 0;
var mousey = 0;
var moveflag = false;
var imgs = [];
var savedata = {
    "pxlsarray":[],
    size: 5,
    "prompt": "a beautiful landscape, golden hour, photography, 8k"
}
var starttime = 0;
var endtime = 0;

window.onload = function(){
    document.getElementById("openimgadd").onchange =function(e) {
        var file, img;
    
    
        if ((file = this.files[0])) {
            img = new Image();
            img.onload = function() {
                let w = this.width;
                let h = this.height;
                document.getElementById("sourceimg").style.visibility = "visible";
                document.getElementById("sourceimg").width = w;
                document.getElementById("sourceimg").height = h;
                try{
                    document.getElementById("source").width = w;
                    document.getElementById("source").height = h;
                }
                catch(error){

                }
                
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
                if(this.width<=this.height){
                    if(this.width>800){
                        w = 800;
                        h = parseInt(Math.round((800/this.width)*this.height));
                    }
                    else{
                        w = this.width;
                        h = this.height;
                    }
                }
                else{
                    if(this.height>800){
                        h = 800;
                        w = parseInt(Math.round((800/this.height)*this.width));
                    }
                    else{
                        w = this.width;
                        h = this.height;
                    }
                }
                document.getElementById("sourceimg").width = w;
                document.getElementById("sourceimg").height = h;
                try{
                    document.getElementById("source").width = w;
                    document.getElementById("source").height = h;
                }
                catch(error){

                }
                
                getdata();
            };
            img.src = URL.createObjectURL(file);
            document.getElementById("sourceimg").src = URL.createObjectURL(file);
            try{
                document.getElementById("uploadcontainer").remove();
            }
            catch(error){

            }
            
    
    
        }
        
    
    }
}

function checkcompletion(){
    console.log(validate);
    for(let i=0;i<validate.length;i++){
        for(let j =0;j<validate[i].length;j++){
            if(validate[i][j]==0 && pieces[i][j]!=999){
                return false;
            }
        }
    }
    return true;
}

function changesize(e){    
    var img = document.getElementById("sourceimg");
    if(e.deltaY < 0){
        img.style.zIndex = 100;
        img.style.opacity = 1;
    }
    else{
        img.style.zIndex = 1;
        img.style.opacity = 0.25;
    }    
}

function getsize(t){
    savedata["size"] = t.value;
    document.getElementById("piecesnumber").innerHTML = String(parseInt(t.value)*parseInt(t.value))+" PIECES";
    console.log(t.value);
}

function getprompt(t){
    savedata["prompt"] = t.value;
    console.log(t.value);
}

function generate(t){
    getdatagen(t);
}

function mousecoordinates(event){
    prevmousex = mousex;
    prevmousey =mousey;
    mousex = event.clientX;
    mousey = event.clientY;  
    
}


function getdata(){    
    document.getElementById("animation").style.visibility = "visible";
    document.getElementById("animation").style.animation = "appear 1s forwards";
    document.getElementById("b1").style.animation = "b 2s 1s forwards infinite";
    document.getElementById("b2").style.animation = "b 2s 1.1s forwards infinite";
    document.getElementById("b3").style.animation = "b 2s 1.2s forwards infinite";
    document.getElementById("b4").style.animation = "b 2s 1.3s forwards infinite";
    let childs = document.getElementById("piececont");
    while (childs.firstChild ){        
        console.log(childs.lastChild.tagName); 
        if(childs.lastChild.tagName == "DIV"){
            childs.removeChild(childs.lastChild); 
        }        
        else{
            break;
        }                           
    }
    pieces = {};
    ratio = 0;
    size = 0;
    imgs = [];
    fetch('http://localhost:5000/getdata', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"savedata":savedata})
    })
    .then(function (response) {
        return response.json();
    })
    .then(data => {
        pieces = data["pieces"];
        validate = [];
        for(let i=0;i<pieces.length;i++){
            let arr = [];
            for(let j =0;j<pieces[i].length;j++){
                if(pieces[i][j]!=999){
                    arr.push(0);
                }
                else{
                    arr.push(1);
                }
            }
            validate.push(arr);
        }
        ratio = data["ratio"]
        console.log(ratio);
        size = parseInt(data["size"]);
        imgs = data["imgs"];
        createapiece();
    }
    );
}

function getdatagen(t){    
    document.getElementById("animation").style.visibility = "visible";
    document.getElementById("animation").style.animation = "appear 1s forwards";
    document.getElementById("b1").style.animation = "b 2s 1s forwards infinite";
    document.getElementById("b2").style.animation = "b 2s 1.1s forwards infinite";
    document.getElementById("b3").style.animation = "b 2s 1.2s forwards infinite";
    document.getElementById("b4").style.animation = "b 2s 1.3s forwards infinite";
    let childs = document.getElementById("piececont");
    while (childs.firstChild ){        
        console.log(childs.lastChild.tagName); 
        if(childs.lastChild.tagName == "DIV"){
            childs.removeChild(childs.lastChild); 
        }        
        else{
            break;
        }                           
    }
    pieces = {};
    ratio = 0;
    size = 0;
    imgs = [];
    fetch('http://localhost:5000/getdatagen', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"savedata":savedata,"model":t.id})
    })
    .then(function (response) {
        return response.json();
    })
    .then(data => {
        pieces = data["pieces"];
        validate = [];
        for(let i=0;i<pieces.length;i++){
            let arr = [];
            for(let j =0;j<pieces[i].length;j++){
                if(pieces[i][j]!=999){
                    arr.push(0);
                }
                else{
                    arr.push(1);
                }
            }
            validate.push(arr);
        }
        ratio = data["ratio"]
        console.log(ratio);
        size = parseInt(data["size"]);
        imgs = data["imgs"];
        document.getElementById("sourceimg").style.visibility = "visible";
        document.getElementById("sourceimg").src = String(data["image"]);
        document.getElementById("sourceimg").width = data["width"];
        document.getElementById("sourceimg").height = data["height"];
        createapiece();

    }
    );
}

function isCollide(a, b) {
    console.log(a);
    a = document.getElementById(a);
    console.log(a);
    b = document.getElementById(b);
    console.log(b);
    var viewportOffseta = a.getBoundingClientRect();
    var viewportOffsetb = b.getBoundingClientRect();
    return !(
        ((parseInt(viewportOffseta.top) + parseInt(a.style.height)) < (parseInt(viewportOffsetb.top))) ||
        (parseInt(viewportOffseta.top) > (parseInt(viewportOffsetb.top) + parseInt(b.style.height))) ||
        ((parseInt(viewportOffseta.left) + parseInt(a.style.width)) < parseInt(viewportOffsetb.left)) ||
        (parseInt(viewportOffseta.left) > (parseInt(viewportOffsetb.left) + parseInt(b.style.width)))
    );
}

function movepiece(id){
    let firstcond = false;
    let secondcond = false;
    let attach = [];
    document.getElementById("cont"+String(parseInt(id))).style.left = mousex-Math.floor(Math.floor(121*ratio)/2)+"px";
    document.getElementById("cont"+String(parseInt(id))).style.top = mousey-Math.floor(Math.floor(121*ratio)/2)+"px";
    let x = parseInt(id);
    try{
        if(isCollide("extra_"+String(x-1)+"_"+String(2),"extra_"+String(x)+"_"+String(0))) {
            firstcond = true;
            attach.push("left");
            if(parseInt(document.getElementById("cont"+String(parseInt(id))).style.transform.match(/\d+/)[0]) ==0 && parseInt(document.getElementById("cont"+String(x-1)).style.transform.match(/\d+/)[0])==0){
                secondcond = true;
            }
        }
    }
    catch(error){

    }

    
    try{
        console.log(isCollide("extra_"+String(x-size)+"_"+String(3),"extra_"+String(x)+"_"+String(1)));
        if(isCollide("extra_"+String(x-size)+"_"+String(3),"extra_"+String(x)+"_"+String(1))) {
            firstcond = true;
            attach.push("top");
            if(parseInt(document.getElementById("cont"+String(parseInt(id))).style.transform.match(/\d+/)[0]) ==0 && parseInt(document.getElementById("cont"+String(x-size)).style.transform.match(/\d+/)[0])==0){
                secondcond = true;
            }
        }

    }
    catch(error){

    }
    try{
        if(isCollide("extra_"+String(x+size)+"_1","extra_"+String(x)+"_3")) {
            firstcond = true;
            attach.push("bottom");
            if(parseInt(document.getElementById("cont"+String(parseInt(id))).style.transform.match(/\d+/)[0]) ==0 && parseInt(document.getElementById("cont"+String(x+size)).style.transform.match(/\d+/)[0])==0){
                secondcond = true;
            }
        }
    }
    catch(error){

    }
    try{
        if(isCollide("extra_"+String(x+1)+"_0","extra_"+String(x)+"_2")) {
            firstcond = true;
            attach.push("right");
            if(parseInt(document.getElementById("cont"+String(parseInt(id))).style.transform.match(/\d+/)[0]) ==0 && parseInt(document.getElementById("cont"+String(x+1)).style.transform.match(/\d+/)[0])==0){
                secondcond = true;
            }
        }
    }
    catch(error){

    }

    if(firstcond && secondcond){
        if(attach.includes("top")){
            document.getElementById("cont"+String(parseInt(id))).style.left = document.getElementById("cont"+String(x-size)).style.left;
            document.getElementById("cont"+String(parseInt(id))).style.top = parseInt(document.getElementById("cont"+String(x-size)).style.top) + Math.floor(121*ratio) + "px";
            validate[x][1] = 1;
            validate[x-size][3] = 1;
        }
        if(attach.includes("bottom")){
            document.getElementById("cont"+String(parseInt(id))).style.left = document.getElementById("cont"+String(x+size)).style.left;
            document.getElementById("cont"+String(parseInt(id))).style.top = parseInt(document.getElementById("cont"+String(x+size)).style.top) - Math.floor(121*ratio) + "px";
            validate[x][3] = 1;
            validate[x+size][1] = 1;
        }
        if(attach.includes("left")){
            document.getElementById("cont"+String(parseInt(id))).style.top = document.getElementById("cont"+String(x-1)).style.top;
            document.getElementById("cont"+String(parseInt(id))).style.left = parseInt(document.getElementById("cont"+String(x-1)).style.left) + Math.floor(121*ratio) + "px";
            validate[x][0] = 1;
            validate[x-1][2] = 1;
        }
        if(attach.includes("right")){
            document.getElementById("cont"+String(parseInt(id))).style.top = document.getElementById("cont"+String(x+1)).style.top;
            document.getElementById("cont"+String(parseInt(id))).style.left = parseInt(document.getElementById("cont"+String(x+1)).style.left) - Math.floor(121*ratio) + "px";
            validate[x][2] = 1;
            validate[x+1][0] = 1;
        }
        
    }
    else{
        validate[x][0] = 0;
        if(x-1>=0){
            validate[x-1][2] = 0;
        }        
        validate[x][1] = 0;
        if(x-size>=0){
            validate[x-size][3] = 0;
        } 
        validate[x][2] = 0;
        if(x+1<pieces.length){
            validate[x+1][0] = 0;
        }        
        validate[x][3] = 0;
        if(x+size<pieces.length){
            validate[x+size][1] = 0;
        }

    }
    console.log(checkcompletion());
    if(checkcompletion()){
        endtime = Date.now();
        document.getElementById("banner").style.visibility = "visible";
        if(parseInt((endtime-starttime)/1000)<60){
            var totaltime = String(parseInt((endtime-starttime)/1000)) + " seconds ";
        }
        else if(parseInt(parseInt((endtime-starttime)/1000)/60)<60){
            var totaltime = String(parseInt(parseInt((endtime-starttime)/1000)/60)) + " minutes " + String(parseInt((endtime-starttime)/1000)-parseInt(parseInt((endtime-starttime)/1000)/60)*60) + " seconds ";
        }
        else{
            var totaltime = String(parseInt(parseInt(parseInt((endtime-starttime)/1000)/60))/60) +" hours "+String(parseInt(parseInt((endtime-starttime)/1000)/60)) + " minutes " + String(parseInt((endtime-starttime)/1000)-parseInt(parseInt((endtime-starttime)/1000)/60)*60) + " seconds ";
        }
        document.getElementById("bannermessage").innerHTML = "Completed in "+ totaltime;
        moveflag = !moveflag;
    }

    if(moveflag){
        setTimeout(() => {
            movepiece(id);
        }, 10);
    }
}

function createapiece(){
    document.getElementById("animation").style.visibility = "hidden";
    document.getElementById("animation").style.top = "110%";
    starttime = 0;
    endtime = 0;
    angles = [0,90,180,270];
    for(let x = 0;x<pieces.length;x++){
        var imgwidth = Math.floor(121*ratio);
        var imgheight = Math.floor(121*ratio);
        var el = document.createElement("div");
        el.className = "cont"+x;
        el.id = "cont"+x;
        el.style.width = Math.floor(121*ratio)+"px";
        el.style.height = Math.floor(121*ratio)+"px";
        //el.style.background = "red";
        el.style.position = "fixed";
        el.style.left = Math.floor(Math.random() * ((screen.availWidth-(Math.floor(121*ratio)+Math.floor(30*ratio)+Math.floor(30*ratio))) - 0) ) + 0+"px";
        el.style.top = Math.floor(Math.random() * ((screen.availHeight-(Math.floor(121*ratio)+Math.floor(30*ratio)+Math.floor(30*ratio))) - 0) ) + 0+"px";
        el.style.transform = "rotate("+ angles[Math.floor(Math.random() * 4)]+"deg)";
        el.oncontextmenu = function(evt){
            var angle = parseInt(document.getElementById("cont"+String(parseInt(evt.target.id))).style.transform.match(/\d+/)[0])+90;
            if(angle>=360){
                angle = 0;
            }
            document.getElementById("cont"+String(parseInt(evt.target.id))).style.transform = "rotate("+ angle+"deg)";
        }

        el.onclick = function(evt){
            moveflag = !moveflag;
            var elem = document.getElementById("cont"+evt.target.id);
            document.getElementById("cont"+evt.target.id).remove();   
            document.getElementById("piececont").append(elem);     
            movepiece(evt.target.id);
        }
        document.getElementById("piececont").append(el);       
        for(let i = 0;i<pieces[x].length;i++){
            if(pieces[x][i]==1){
                var newel = document.createElement("div");
                newel.id = "extra_"+x+"_"+i;
                newel.style.width = Math.floor(30*ratio)+"px";
                newel.style.height = Math.floor(30*ratio)+"px";
                //newel.style.background = "yellow";
                newel.style.position = "absolute";
                
                
                if(i==0){
                    newel.style.left = 0 - Math.floor(30*ratio) + "px";
                    newel.style.top = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(0%,-50%)";
                    imgwidth+=Math.floor(30*ratio);
                }
                else if(i==1){
                    newel.style.top = 0 - Math.floor(30*ratio) + "px";
                    newel.style.left = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(-50%,0%)";
                    imgheight+=Math.floor(30*ratio);
                } 
                else if(i==2){
                    newel.style.left = Math.floor(121*ratio) + "px";
                    newel.style.top = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(0%,-50%)";
                    imgwidth+=Math.floor(30*ratio);
                }
                else{
                    newel.style.top = Math.floor(121*ratio) + "px";
                    newel.style.left = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(-50%,0%)";
                    imgheight+=Math.floor(30*ratio);
                }
                newel.style.zIndex = "2";
                el.append(newel);
            }
            else if(pieces[x][i]==0){
                var newel = document.createElement("div");
                newel.id = "extra_"+x+"_"+i;
                newel.style.width = Math.floor(30*ratio)+"px";
                newel.style.height = Math.floor(30*ratio)+"px";
                //newel.style.background = "blue";
                newel.style.position = "absolute";
                if(i==0){
                    newel.style.left = 0 + "px";
                    newel.style.top = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(0%,-50%)";
                }
                else if(i==1){
                    newel.style.top = 0  + "px";
                    newel.style.left = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(-50%,0%)";
                } 
                else if(i==2){
                    newel.style.left = Math.floor(121*ratio) - Math.floor(30*ratio)+ "px";
                    newel.style.top = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(0%,-50%)";
                }
                else{
                    newel.style.top = Math.floor(121*ratio) - Math.floor(30*ratio)+ "px";
                    newel.style.left = Math.floor(121*ratio/2) + "px"
                    newel.style.transform = "translate(-50%,0%)";
                }
                el.append(newel);
            }
            
            
        }
        var img = document.createElement("img");
        img.id = x;
        img.src = imgs[x];
        img.style.position = "absolute";
        if(pieces[x][0]==1){
            img.style.left = -1*Math.floor(30*ratio)+"px";
        }
        else if(pieces[x][0]==0){
            img.style.left = 0+"px";
        }
        if(pieces[x][1]==1){
            img.style.top = -1*Math.floor(30*ratio)+"px";
        }
        else if(pieces[x][1]==0){
            img.style.top = 0+"px";
        }
        img.style.zIndex = "1";
        el.append(img);
    }
    starttime = Date.now(); 



}


function hidepanel(){
    if(document.getElementById("settings").style.top=="-10%"){
        document.getElementById("settings").style.top = "0%";
    }
    else{
        document.getElementById("settings").style.top = "-10%";
    }
}

function closebanner(){
    document.getElementById("banner").style.visibility = "hidden";
}