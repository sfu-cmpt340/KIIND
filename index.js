const express = require("express"); // this is the Sinatra-like MVC frameworks for Node.js
// var cors = require("cors"); // cross-origon resourse sharing
const path = require('path');
const request = require('request');
var session = require("express-session"); // used for login
const PythonShell = require('python-shell').PythonShell;
const fs = require('fs') 

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
app.get("/acl", (req, res) => res.render("pages/acl"));
app.get("/meniscus", (req, res) => res.render("pages/meniscus"));
app.get("/about", (req, res) => res.render("pages/about"));


app.post("/createGIF", (req, res) => {
    // console.log("yes?", req.body);          // this would be the data sent with the request
    console.log("type of", typeof req.body);
    // console.log("oops", req.body.file)
    // instead of passing to python, save to a text file in public?
    // and then python can access that


    fs.writeFile('public/file.txt', req.body.file, (err) => { 
          
        // In case of a error throw err. 
        if (err) throw err; 
    }) 







    let pyshell = new PythonShell('public/createGIF.py');


    let options = {
        // args: [req.body.file] // too long
        args: [req.body]
    }

    PythonShell.run('public/createGIF.py', options).then(messages=>{
        // results is an array consisting of messages collected during execution
        console.log('results: %j', messages);
    });

               


});


// app.get("/createGIF", (req, res) => {

//     let pyshell = new PythonShell('public/createGIF.py');
    
//     let options = {
//         // args: [file]
//         args: "hi"

//       };

      
//     PythonShell.run('public/createGIF.py', options).then(messages=>{
//     // results is an array consisting of messages collected during execution
//     console.log('results: %j', messages);
//     });

// })






