function init(){
    document.getElementById('fileInput').addEventListener('change', storeFile, false);

}

// document.getElementById('fileInput').addEventListener('change', storeFile, false);


function storeFile(event){
    // store as base64

    var file = event.target.files[0];
    var reader = new FileReader();
    var fileContentBase64;
    reader.onload = function(e){
        fileContentBase64 = e.target.result;
        sessionStorage.setItem('uploadedFile', fileContentBase64);

    }
    reader.readAsDataURL(file);
    console.log("stored in session storage");



    // window.location.href = 'https://cmpt340-project-758b976dd842.herokuapp.com/createGIF';
    // newURL = window.location.protocol + "//" + window.location.host + '/createGIF';
    // window.location.href = newURL;

    const formData = new FormData();
    formData.append('file', fileContentBase64);

    let base64file = window.sessionStorage.getItem('uploadedFile');
    // console.log(base64file);

    fetch("/createGIF", {
        method: "POST", 
        headers: {
           'Content-Type': 'application/json'
            // 'Content-Type': 'application/octet-stream'

        //    'Content-Type': 'application/x-www-form-urlencoded'
        },
        // body: formData
        // body: JSON.stringify(base64file)
        // body: JSON.stringify(window.btoa(base64file))
        body: JSON.stringify({"file": base64file})
        // body: test


    })
    .then(response => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error('File upload failed');
        }
      })
    .then(data => {
        console.log('Server response:', data);
    })
    .catch(error => {
        console.error('Error uploading file:', error);
    });



}

