const express = require('express');
const MongoClient = require('mongodb').MongoClient;
const app = express();


const  url = "mongodb://localhost:27017/vegifyDatabase"

MongoClient.connect(url,(err,db)=>{
    if(err) throw err;
    console.log("Data Base has been created");
})











app.get('/',(req,res)=>{
    res.send("Hello")
})

app.listen(3000)

//http://localhost:3000/