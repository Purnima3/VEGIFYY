const express = require('express');
const MongoClient = require('mongodb').MongoClient;
const app = express();


const  url = "mongodb://localhost:27017/vegifyDatabase"

MongoClient.connect(url,(err,db)=>{
    if(err) throw err;
   const dbo = db.db('vegifyDatabase')
   
   var myobj = {username: 'Purnima3', Password : "hehe"}

   dbo.collection('Users').insertOne(myobj,(err,res)=>{
    if(err) throw err;
    console.log("Document inserted");
    db.close();
   })
})
  










app.get('/',(req,res)=>{
    res.send("Hello")
})

app.listen(3000)

//http://localhost:3000/