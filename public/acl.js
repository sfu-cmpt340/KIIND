import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);



function init(){
    document.getElementById('fileInput').addEventListener('change', storeFile, false);

}

function storeFile(event){
    // store as base64

    var file = event.target.files[0];
    var reader = new FileReader();
    reader.onload = function(e){
        var fileContentBase64 = e.target.result;
        sessionStorage.setItem('uploadedFile', fileContentBase64);

    }
    reader.readAsDataURL(file);

    const spawn = require("child_process").spawn;

}

