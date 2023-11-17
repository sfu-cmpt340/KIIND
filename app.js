const express = require("express"); // this is the Sinatra-like MVC frameworks for Node.js
// var cors = require("cors"); // cross-origon resourse sharing

// this is our app
var app = express();

const PORT = process.env.PORT || 3000;


// app.get("/", (req, res) => res.render("pages/index"));
app.get("/", (req, res) => res.send("hello world"));

app.listen(PORT, () => console.log(`Listening on ${PORT}`));
