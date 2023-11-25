const express = require("express"); // this is the Sinatra-like MVC frameworks for Node.js
// var cors = require("cors"); // cross-origon resourse sharing
const path = require('path');
const request = require('request');
var session = require("express-session"); // used for login
const PythonShell = require('python-shell').PythonShell;
const fs = require('fs');
const fileUpload = require('express-fileupload');

// import {PythonShell} from 'python-shell';

// this is our app
var app = express();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
// app.use(express.bodyParser({limit: '50mb'}));

// app.get("/", (req, res) => res.render("pages/index"));
// app.get("/", (req, res) => res.send("hello world"));
app.use(express.static(path.join(__dirname, 'public')));
// app.use(express.json());  
app.use(express.json({limit: '50mb'}));
// app.use(express.urlencoded({ extended: true }));
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

    const uploadedFile = req.files.datafile;
    uploadPath = __dirname + "/uploads/data.npy"; 

    uploadedFile.mv(uploadPath, function (err) { 
        if (err) { 
          console.log(err); 
          res.send("Failed !!"); 
        }
         else {
            let pyshell = new PythonShell('public/createGIF.py');

            let options = {
                // args: [req.body.file] // too long
                args: "file upload"
            }
            var d;
            PythonShell.run('public/createGIF.py', options).then(messages=>{
                // results is an array consisting of messages collected during execution
                console.log('results: %j', messages);
                d = messages[0];
                res.render("pages/acl", { gif: "./scan.gif", diagnosis: String(d) });
            });
         }
      }); 
    
});


app.post("/fileUpload2", (req, res) =>{

    const uploadedFile = req.files.datafile;
    uploadPath = __dirname + "/uploads/data.npy"; 

    uploadedFile.mv(uploadPath, function (err) { 
        if (err) { 
          console.log(err); 
          res.send("Failed !!"); 
        }
         else {
            let pyshell = new PythonShell('public/createGIF.py');

            let options = {
                // args: [req.body.file] // too long
                args: "file upload"
            }
            var d;
            PythonShell.run('public/createGIF.py', options).then(messages=>{
                // results is an array consisting of messages collected during execution
                console.log('results: %j', messages);
                d = messages[0];
                console.log("messages", d);
                res.render("pages/meniscus", { gif: "./scan.gif", diagnosis: String(d) });
            });
         }
      }); 
    
});








