const express = require("express"); // this is the Sinatra-like MVC frameworks for Node.js
// var cors = require("cors"); // cross-origon resourse sharing
const path = require('path');
const request = require('request');
var session = require("express-session"); // used for login
const PythonShell = require('python-shell').PythonShell;
const fs = require('fs');
const fileUpload = require('express-fileupload');

// this is our app
var app = express();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({limit: '50mb'}));
app.use(express.urlencoded({limit: '50mb'}));
app.use(fileUpload());


// the user session
app.use(
    session({
      // secret should be random
      secret: "idfh3l4j5hl98fad9fj34or",
      resave: true,
      saveUninitialized: true,
  
      //Note about the session expire date:
      // By default cookie.maxAge is null
      // meaning no "expires" parameter is set
      //so the cookie becomes a browser-session cookie.
      //When the user closes the browser the cookie (and session) will be removed.
    })
  );
app.set("views", path.join(__dirname, "views"));
app.set('view engine', 'ejs');
app.get('/', (req, res) => res.render('pages/index'));


// pages
app.get("/acl", (req, res) => res.render("pages/acl", { gif: "" , diagnosis: ""}));
app.get("/meniscus", (req, res) => res.render("pages/meniscus", { gif: "" , diagnosis: ""}));
app.get("/about", (req, res) => res.render("pages/about"));


app.post("/fileUpload", (req, res) =>{
    // console.log(req.files);
    const uploadedFile_a = req.files.datafile_a;
    const uploadedFile_s = req.files.datafile_s;
    const uploadedFile_c = req.files.datafile_c;

    uploadPath_a = __dirname + "/uploads/data_a.npy"; 
    uploadPath_s = __dirname + "/uploads/data_s.npy"; 
    uploadPath_c = __dirname + "/uploads/data_c.npy"; 

    console.log("test");


    uploadedFile_a.mv(uploadPath_a, function (err) { 
        if (err) { 
          console.log(err); 
          res.send("Failed !!"); 
        }
         else {

            uploadedFile_s.mv(uploadPath_s, function (err){
                if (err){
                    console.log(err);
                    res.send("Failed !!");
                }
                else{
                    uploadedFile_c.mv(uploadPath_c, function (err){
                        if (err) {
                            console.log(err);
                            res.send("Failed !!")
                        }
                        else{
                            let pyshell = new PythonShell('public/creategif.py');

                            let options = {
                                // args: [req.body.file] // too long
                                args: "acl"
                            }
                            var d;
                            PythonShell.run('public/creategif.py', options).then(messages=>{
                                // results is an array consisting of messages collected during execution
                                console.log('results: %j', messages);
                                d = messages[0];
                                res.render("pages/acl", { gif: "./scan.gif", diagnosis: String(d) });
                            });


                        }
                    })
                }
            })
        
        }
      }); 
    
});

app.post("/fileUpload2", (req, res) =>{
    // console.log(req.files);
    const uploadedFile_a = req.files.datafile_a;
    const uploadedFile_s = req.files.datafile_s;
    const uploadedFile_c = req.files.datafile_c;

    uploadPath_a = __dirname + "/uploads/data_a.npy"; 
    uploadPath_s = __dirname + "/uploads/data_s.npy"; 
    uploadPath_c = __dirname + "/uploads/data_c.npy"; 

    console.log("test");


    uploadedFile_a.mv(uploadPath_a, function (err) { 
        if (err) { 
          console.log(err); 
          res.send("Failed !!"); 
        }
         else {

            uploadedFile_s.mv(uploadPath_s, function (err){
                if (err){
                    console.log(err);
                    res.send("Failed !!");
                }
                else{
                    uploadedFile_c.mv(uploadPath_c, function (err){
                        if (err) {
                            console.log(err);
                            res.send("Failed !!")
                        }
                        else{
                            let pyshell = new PythonShell('public/creategif.py');

                            let options = {
                                // args: [req.body.file] // too long
                                args: "meniscus"
                            }
                            var d;
                            PythonShell.run('public/creategif.py', options).then(messages=>{
                                // results is an array consisting of messages collected during execution
                                console.log('results: %j', messages);
                                d = messages[0];
                                res.render("pages/meniscus", { gif: "./scan.gif", diagnosis: String(d) });
                            });


                        }
                    })
                }
            })
        
        }
      }); 
    
});









