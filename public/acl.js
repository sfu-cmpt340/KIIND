// import { createRequire } from 'node:module';
// const require = createRequire(import.meta.url);
import {PythonShell} from 'python-shell';



// function init(){
//     document.getElementById('fileInput').addEventListener('change', storeFile, false);

// }

document.getElementById('fileInput').addEventListener('change', storeFile, false);


function storeFile(event){
    // store as base64

    var file = event.target.files[0];
    var reader = new FileReader();
    reader.onload = function(e){
        var fileContentBase64 = e.target.result;
        sessionStorage.setItem('uploadedFile', fileContentBase64);

    }
    reader.readAsDataURL(file);
    console.log("stored in session storage");



    PythonShell.run('./createGIF.py', null).then(messages=>{
        console.log('finished');
    });

    // oh maybe this can work on heroku
    // const spawn = require("child_process").spawn;
    // const pythonProcess = spawn('python',["./createGif.py", "arg1"]);


    // fetch('https://cmpt340-project-758b976dd842.herokuapp.com/createGIF', 
    //     // {   
    //     // method: 'POST',
    //     // body: formData
    //     // }
    //     )
    //     .then(response => response.json())
    //     .then(data => {
    //     console.log('Prediction:', data);
    //     })
    // //     .catch(error => {
    // //     console.error('Error:', error);
    // // })
    // ;


}

