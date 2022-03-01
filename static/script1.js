const DD_Area = document.querySelector('#drop_zone')

var uploads = 0;

function dropHandler(ev) {
    console.log('File(s) dropped');

    // Prevent default behavior (Prevent file from being opened)
    ev.stopPropagation();
    ev.preventDefault();

    const fileList = ev.dataTransfer.files;

    var DD = document.getElementById("drop_zone")
    removeAllChildNodes(DD)

    readImage(fileList[0]);

    fileUpload.files = ev.dataTransfer.files;

    submitImgForm();

    uploads += 1;
    enableValidation();
}

function dragOverHandler(ev) {
    console.log('File(s) in drop zone');
  
    // Prevent default behavior (Prevent file from being opened)
    ev.preventDefault();
  }


function readImage(file) {
    const reader = new FileReader();
    reader.addEventListener('load', (event) => {
    uploaded_drone_img = event.target.result;
    document.querySelector("#drop_zone").style.backgroundImage =`url(${uploaded_drone_img})`;
    });
    reader.readAsDataURL(file);
 }

async function clearDD(ev){
    await fetch('/delete_images')
    document.querySelector('#drop_zone').style.backgroundImage = null
    document.querySelector('#sat_img').style.backgroundImage = null
    document.getElementById('form_drone_img').value = null
    document.getElementById('bottom_left_long').value = 0
    document.getElementById('bottom_left_lat').value = 0
    document.getElementById('top_right_long').value = 0
    document.getElementById('top_right_lat').value = 0
    removeAllChildNodes(document.querySelector('#drop_zone'))
    removeAllChildNodes(document.querySelector('#sat_img'))
    var paragraph1 = document.createElement("p")
    paragraph1.innerText = "Drag drone image to this Drop Zone ..."
    var paragraph2 = document.createElement("p")
    paragraph2.innerText = "Here corresponding satellite image will be shown..."
    document.querySelector('#drop_zone').appendChild(paragraph1)
    document.querySelector('#sat_img').appendChild(paragraph2)
    uploads = 0;
    document.getElementById("val_btn").disabled = true
    document.getElementById("val_btn").style.visibility = "visible"
    document.getElementById("val_btn").style.display = "block"
    document.getElementById("btn_container").removeChild(document.getElementById("result"))
    document.getElementById("sat_btn").style.display = "block"

}

function submitImgForm(){
    // alert('test');
    document.getElementById("form_drone_img").submit();
}

function submitCoordinatesForm(){
    //alert('test');
    uploads += 1;
    document.getElementById("form_coordinates").submit();
    // add a loading circle
    var sat = document.getElementById("sat_img")
    removeAllChildNodes(sat)
    var loader = document.createElement("div");
    loader.className = "loader";
    sat.appendChild(loader);
    getapi(api_url);
}

const api_url = "./static/images/satellite.png"

// Defining async function
async function getapi(url) {
    const response1 = await fetch(url)
    console.log(response1.status)
    if(response1.status != 200){
        alert('Invalid coordinate information. Please reload and enter valid coordinates.\n\nTypically the selected are is too large.')
        clearDD()
        return
    }
    
    
    // Storing response
    const response = await fetch(url).then(function(data){
        return data.blob();
    }).then(function(img){
        var dd = URL.createObjectURL(img);
        document.querySelector("#sat_img").style.backgroundImage =`url(${dd})`;
    });    
    var sat = document.getElementById("sat_img")
    sat.removeChild(sat.childNodes[0]);
    enableValidation()
}


function removeAllChildNodes(parent) {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
}

function enableValidation(){
    if(uploads ==2){
        document.getElementById("val_btn").style.visibility = "visible"
        document.getElementById("val_btn").disabled = false
    }
}

async function validate(){
    // add a loading circle
    var btn_cont = document.getElementById("btn_container")
    // btn_cont.removeChild(document.getElementById("val_btn"))
    document.getElementById("val_btn").style.display = "none"
    // btn_cont.removeChild(document.getElementById("sat_btn"))
    document.getElementById("sat_btn").style.display = "none"
    var loader = document.createElement("div");
    loader.className = "loader-small";
    loader.id = "val_loader"
    btn_cont.appendChild(loader);
    const response = await fetch('/validate')
    const json = await response.json()
    btn_cont.removeChild(document.getElementById("val_loader"))
    if(json.result==0){
        const result = document.createElement("div");
        result.id = "result"
        result.className = "negResult"
        const newContent = document.createTextNode("Not from same area!");
        result.appendChild(newContent)
        btn_cont.appendChild(result)
    }
    if(json.result==1){
        const result = document.createElement("div");
        result.id = "result"
        result.className = "posResult"
        const newContent = document.createTextNode("From same area!");
        result.appendChild(newContent)
        btn_cont.appendChild(result)
    }
    console.log(json.result)
}