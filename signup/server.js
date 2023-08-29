const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const path= require('path'); 

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(bodyParser.json());
const static_path= path.join(__dirname, "/public");
// const main_path= "/main";
// console.log(path.join(__dirname, "./public/main/index.html"));

app.use(express.static(static_path));
app.use(bodyParser.urlencoded({ extended: true }));

const DB_URI = "mongodb+srv://Nitin:pzlpfWhNJxaQtYta@contactus.2oxbxjx.mongodb.net/?retryWrites=true&w=majority";

// MongoDB Connection
mongoose.connect(DB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;

db.on('error', () => console.error('Error connecting to the database'));
db.once('open', () => {
  console.log('Connected to the database');
});

// MongoDB Schema
const UserSchema = new mongoose.Schema({
  fname: String,
  lname: String,
  phone: String,
  password: String,
});

const User = mongoose.model('Userinfo', UserSchema);

// API Endpoint for Signup
app.post('/', async (req, res) => {
  try {
    const { fname, lname, phone, password } = req.body;
    const newUser = new User({
      fname,
      lname,
      phone,
      password,
    });
    await newUser.save();
    res.status(201).redirect('/login.html'); // Redirect to login page
  } catch (error) {
    res.status(500).json({ error: 'An error occurred' });
  }
});

// API Endpoint for Login
app.post("/logsubmit", async(req, res) =>{
     try{
        const phone= req.body.phone;
        const password= req.body.password;

       const userphone= await User.findOne({phone:phone});
       if(userphone.password === password){
        res.status(201).redirect('/main/index.html');
       }
       else{
            res.send("invalid credentials");
       }
     } catch (error){
      res.status(400).send("Something went wrong")
     }
})


app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});