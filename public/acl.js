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

    // oh maybe this can work on heroku
    const spawn = require("child_process").spawn;
    const pythonProcess = spawn('python',["./createGif.py", "arg1"]);


//     fetch('http://127.0.0.1:5000/upload_image', {
//         method: 'POST',
//         body: formData
//         })
//         .then(response => response.json())
//         .then(data => {
//         console.log('Prediction:', data.prediction);
//         })
//         .catch(error => {
//         console.error('Error:', error);
//     });

    pythonProcess.stdout.on('data', (data) => {
        // Do something with the data returned from python script
        console.log(data)
    });
}

