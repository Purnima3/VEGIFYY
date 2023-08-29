const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const path= require('path'); 

const app = express();
const port = process.env.PORT || 4000;

// Middleware
app.use(bodyParser.json());
const static_path= path.join(__dirname, "/public");
app.use(express.static(static_path, { index: 'index1.html' }));
app.use(express.static('/public'));
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
const contactSchema = new mongoose.Schema({
    fname: String,
    lname: String,
    email: String,
    mobile: String,
    msg: String,
  });
  
  const Contact = mongoose.model('Contact', contactSchema);
  
  // API Endpoint for Form Submission
  app.post('/submit', async (req, res) => {
    try {
      const { fname, lname, email, mobile, msg } = req.body;
      const newContact = new Contact({
        fname,
        lname,
        email,
        mobile,
        msg,
      });
      await newContact.save();
      res.status(201).redirect('./success.html');
    } catch (error) {
      res.status(500).json({ error: 'An error occurred' });
    }
  });
  app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
  });