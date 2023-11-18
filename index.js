const express = require("express"); // this is the Sinatra-like MVC frameworks for Node.js
// var cors = require("cors"); // cross-origon resourse sharing
const path = require('path');

// this is our app
var app = express();

const PORT = process.env.PORT || 3000;


// app.get("/", (req, res) => res.render("pages/index"));
// app.get("/", (req, res) => res.send("hello world"));
app.use(express.static(path.join(__dirname, 'public')));
app.set("views", path.join(__dirname, "views"));
app.set('view engine', 'ejs');
app.get('/', (req, res) => res.render('pages/index'));


// pages
app.get("/acl", (req, res) => res.render("pages/acl"));
app.get("/meniscus", (req, res) => res.render("pages/meniscus"));



app.listen(PORT, () => console.log(`Listening on ${PORT}`));






// express()
//   .use(express.static(path.join(__dirname, 'public')))
//   .set('views', path.join(__dirname, 'views'))
//   .set('view engine', 'ejs')
//   .get('/', (req, res) => res.render('pages/index'))
//   .listen(PORT, () => console.log(`Listening on ${ PORT }`))