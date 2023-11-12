var mousex = 0;
var mousey = 0;
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
var savedata = {
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
    "imgpos": ["25%","25%"],
    "boxsz":512,
    "ff":10,
    "pxlsarray":""


};

var imagedims = {
    "width":128,
    "height":128
};



window.onload = function(){
    var canvas = document.getElementById("source");
    var ctx = canvas.getContext("2d");
    var img = document.getElementById("sourceimg");
    ctx.drawImage(img, 0, 0);
    document.getElementById("selector").style.left = "50%";
    document.getElementById("selector").style.top = "50%";
    
    
    //preparestuff();
    //alert("hello");
}

function ffupdate(){
    savedata["ff"] = document.getElementById("ff").value;
}

function draw(){
    var erasersize = parseInt(document.getElementById("erasersize").value);
    if(erasersize==null || erasersize==undefined ){
        erasersize = 30;
    }
    var c = document.getElementById("source");
    var ctx = c.getContext("2d");    
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
    console.log(pxls);
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

function erasemodeon(){
    if(eraseflag){
        document.getElementById("selector").style.visibility = "visible";
        document.getElementById("moveimglistner").style.visibility = "visible";
        document.getElementById("eraseon").style.backgroundColor = "gray";
        eraseflag = !eraseflag
    }
    else{
        document.getElementById("selector").style.visibility = "hidden";
        document.getElementById("moveimglistner").style.visibility = "hidden";
        document.getElementById("eraseon").style.backgroundColor = "rgb(20, 20, 20)";
        eraseflag = !eraseflag
    }
}




function preparestuff(){
    fetch('http://localhost:5000/prepare')
    .then(response => response.text())
    .then(data => {
        data = JSON.parse(data);
        document.getElementById("source").style.left = data["imgpos"][0];
        document.getElementById("source").style.top = data["imgpos"][1];
        savedata["imgpos"][0] = data["imgpos"][0];
        savedata["imgpos"][1] = data["imgpos"][1];
        selectorsize = parseInt(data["boxsz"]);    
        savedata["boxsz"] = parseInt(data["boxsz"]);  
    });
    
}
console.log("here");

function saved(savedata){
    var responseClone; // 1
    fetch('http://localhost:5000/savedata', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"savedata":savedata,"selectorsize":selectorsize})
    })
    .then(function (response) {
        responseClone = response.clone(); // 2
        return response.json();
    })
    .then(data => {
        window.close();
    }, function (rejectionReason) { // 3
        console.log('Error parsing JSON from response:', rejectionReason, responseClone); // 4
        responseClone.text() // 5
        .then(function (bodyText) {
            console.log('Received the following instead of valid JSON:', bodyText); // 6
        });
    });
    
    document.getElementById("msg").style.visibility = "visible";
    document.getElementById("container").remove();
    
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
    if((parseFloat(document.getElementById("selector").style.left)-(selectorsize/2))<0){
        selec.style.left = selectorsize/2+"px";
    }
    if((parseFloat(document.getElementById('selector').getBoundingClientRect()["right"])-(selectorsize/2))>parseInt(screen.availWidth)){
        selec.style.left = parseInt(screen.availWidth)-selectorsize+"px";
    }
    if((parseFloat(document.getElementById("selector").style.top)-(selectorsize/2))<0){
        selec.style.top = selectorsize/2+"px";
    }
    if((parseFloat(document.getElementById('selector').getBoundingClientRect()["bottom"])-(selectorsize/2))>parseInt(screen.availHeight)){
        selec.style.bottom = parseInt(screen.availHeight)-selectorsize+"px";
    }
}
function imgmover(){
    var selec = document.getElementById("source");
    if(mousex>prevmousex){
        selec.style.left = parseInt(document.getElementById('source').getBoundingClientRect()["left"])+10+"px";
    }
    else if(mousex<prevmousex){
        selec.style.left = parseInt(document.getElementById('source').getBoundingClientRect()["left"])-10+"px";
    }
    if(mousey>prevmousey){
        selec.style.top = parseInt(document.getElementById('source').getBoundingClientRect()["top"])+10+"px";
    }
    else if(mousey<prevmousey){
        selec.style.top = parseInt(document.getElementById('source').getBoundingClientRect()["top"])-10+"px";
    }

    
}
function snapshot(){
    console.log("hello");
    savedata = {
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
        "imgpos": savedata["imgpos"],
        "boxsz":savedata["boxsz"],
        "ff":savedata["ff"],
        "pxlsarray":savedata["pxlsarray"]
    
    };
    imagedims["width"] = parseInt(document.getElementById('source').width);
    imagedims["height"] = parseInt(document.getElementById('source').height);
    document.getElementById("saveexit").style.visibility = "visible";
    console.log((parseFloat(document.getElementById("selector").style.left)-(selectorsize/2)));
    console.log(document.getElementById('source').getBoundingClientRect()["left"]);
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
    
}
function moveimgenable(){
    if(moveimgenableflag){
        document.getElementById("selector").style.visibility = "visible";
        document.getElementById("moveimg").style.backgroundColor = "gray";
        moveimgenableflag = !moveimgenableflag
    }
    else{
        document.getElementById("selector").style.visibility = "hidden";
        document.getElementById("moveimg").style.backgroundColor = "rgb(20, 20, 20)";
        moveimgenableflag = !moveimgenableflag
    }
}
function moveimg(){    
    if(clickflag && moveimgenableflag){
        document.getElementById("moveimglistner").removeEventListener("mousemove", imgmover);
        clickflag = !clickflag
    }
    else{   
        document.getElementById("moveimglistner").addEventListener("mousemove", imgmover);
        clickflag = !clickflag
    }
}
function changeselectorsize(){
    var elem = document.getElementById("selectorsize");
    selectorsize = elem.value;
    savedata["boxsz"] = selectorsize;
    document.getElementById("selector").style.width = selectorsize+"px";
    document.getElementById("selector").style.height = selectorsize+"px";
}

function savejson(){
    console.log(savedata)
    /*if(savedata["additionaldims"]["top"]<0){
        savedata["additionaldims"]["top"] = savedata["additionaldims"]["top"]*-1;
    }
    if(savedata["additionaldims"]["left"]<0){
        savedata["additionaldims"]["left"] = savedata["additionaldims"]["left"]*-1;
    }
    if(savedata["additionaldims"]["right"]<0){
        savedata["additionaldims"]["right"] = savedata["additionaldims"]["right"]*-1;
    }
    if(savedata["additionaldims"]["bottom"]<0){
        savedata["additionaldims"]["bottom"] = savedata["additionaldims"]["bottom"]*-1;
    }

    if(savedata["crpdims"]["left"]<0){
        savedata["crpdims"]["left"] = savedata["crpdims"]["left"]*-1;
    }
    if(savedata["crpdims"]["top"]<0){
        savedata["crpdims"]["top"] = savedata["crpdims"]["top"]*-1;
    }
    if(savedata["crpdims"]["right"]<0){
        savedata["crpdims"]["right"] = savedata["crpdims"]["right"]*-1;
    }
    if(savedata["crpdims"]["bottom"]<0){
        savedata["crpdims"]["bottom"] = savedata["crpdims"]["bottom"]*-1;
    }*/

    saved(savedata);
}

function reset(){
    var selec = document.getElementById("selector");
    selec.style.left = mousex+"px";
    selec.style.top = mousey+"px";
}
