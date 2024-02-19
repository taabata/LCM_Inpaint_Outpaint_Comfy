var image = "";
var image1 = "";
var image2 = "";
var setting = 1;
var animationflag = false;
var animationdirection = false;

function select(evt){

    evt.target.style.background = "rgba(128, 128, 128, 0.4)";
    setting = parseInt(evt.target.id.slice(7));
    console.log(setting);
    for(let i = 0; i<3;i++){
        if(evt.target.id != "setting1"){
            document.getElementById("setting1").style.background = "rgba(128, 128, 128, 0.2)";
        }
        if(evt.target.id != "setting2"){
            document.getElementById("setting2").style.background = "rgba(128, 128, 128, 0.2)";
        }
        if(evt.target.id != "setting3"){
            document.getElementById("setting3").style.background = "rgba(128, 128, 128, 0.2)";
        }
    }

}

function changeimg(evt){
    
    if(image!=""){
        image1 = image;
    }
    document.getElementById("image").style.visibility = "visible";
    document.getElementById("image").style.boxShadow = "10px 10px 10px black";
    document.getElementById("inplbl").style.opacity = "0";
    document.getElementById("image").src = URL.createObjectURL(evt.target.files[0]);
    var img = new Image;
    img.src = URL.createObjectURL(evt.target.files[0]);
    img.onload = function(){
        var canvas = document.getElementById("canvas");
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(document.getElementById("image"), 0, 0);
        data = ctx.getImageData(0, 0, canvas.width,canvas.height).data;
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
        image = pxlsarray;
        if(image1!=""){
            image2 = image;
            mix();
        }
    }    

}

function stickerize(){
    
    document.getElementById("sticker").style.visibility = "hidden";
    document.getElementById("settings").style.opacity = "0";
    document.getElementById("cpcontainer").style.animation = null;
    document.getElementById("cpcontainer").offsetHeight;
    document.getElementById("cpcontainer").style.animation = "changeprogressbar 1s forwards";
    setTimeout(() => {
        document.getElementById("progressanimation").style.visibility = "visible";
        document.getElementById("progressanimation").style.animation = null;
        document.getElementById("progressanimation").offsetHeight;
        document.getElementById("progressanimation").style.animation = "progressbar 1s infinite";
    }, 1000);
    fetch("http://localhost:5000/stickerize",{
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"image":image})
    }).
    then(function (response) {
        return response.json();
    })
    .then(data => {
        document.getElementById("image").src = String(data["byte_im"]);
        document.getElementById("image").style.boxShadow = "0px 0px 0px transparent";
        image = "";
        image1 = "";
        image2 = "";
        animationflag = true;
        document.getElementById("progressanimation").style.visibility = "hidden";
        document.getElementById("progressanimation").style.animation = null;
        document.getElementById("progressanimation").offsetHeight;
        document.getElementById("cpcontainer").style.animation = null;
        document.getElementById("cpcontainer").offsetHeight;
        document.getElementById("cpcontainer").style.animation = "changeprogressbar 1s reverse";
        setTimeout(() => {
            document.getElementById("settings").style.opacity = "1";
            document.getElementById("sticker").style.visibility = "visible";
            animationflag = false;
        }, 1000);
    })

}

function mix(){

    document.getElementById("sticker").style.visibility = "hidden";
    document.getElementById("settings").style.opacity = "0";
    document.getElementById("cpcontainer").style.animation = null;
    document.getElementById("cpcontainer").offsetHeight;
    document.getElementById("cpcontainer").style.animation = "changeprogressbar 1s forwards";
    setTimeout(() => {
        document.getElementById("progressanimation").style.visibility = "visible";
        document.getElementById("progressanimation").style.animation = null;
        document.getElementById("progressanimation").offsetHeight;
        document.getElementById("progressanimation").style.animation = "progressbar 1s infinite";
    }, 1000);
    fetch("http://localhost:5000/mix",{
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({"image1":image1,"image2":image2,"setting":setting})
    }).
    then(function (response) {
        return response.json();
    })
    .then(data => {
        document.getElementById("image").src = String(data["byte_im"]);
        var img = new Image;
        img.src = document.getElementById("image").src;
        img.onload = function(){
            var canvas = document.getElementById("canvas");
            var ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(document.getElementById("image"), 0, 0);
            data = ctx.getImageData(0, 0, canvas.width,canvas.height).data;
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
            image = pxlsarray;
        }
        image1 = "";
        image2 = "";
        animationflag = true;
        document.getElementById("progressanimation").style.visibility = "hidden";
        document.getElementById("progressanimation").style.animation = null;
        document.getElementById("progressanimation").offsetHeight;
        document.getElementById("cpcontainer").style.animation = null;
        document.getElementById("cpcontainer").offsetHeight;
        document.getElementById("cpcontainer").style.animation = "changeprogressbar 1s reverse";
        setTimeout(() => {
            document.getElementById("settings").style.opacity = "1";
            document.getElementById("sticker").style.visibility = "visible";
            animationflag = false;
        }, 1000);
    })

}