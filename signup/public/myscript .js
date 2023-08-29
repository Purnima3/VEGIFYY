// for email
function validateEmail(){
    let emailField= document.getElementById("email-field");
    let emailError= document.getElementById("email-error");
    if(!emailField.value.match(/^[A-Za-z\._\-0-9]*[@][A-Za-z]*[\.][a-z]{2,4}$/)){
        emailError.innerHTML= "*Please enter a valid email";
        // emailField.style.borderBottomColor= "red";
        return false;
    }
    emailError.innerHTML= "";
    return true;

} 

// for phone number 
function validphone()
{
    let phoneField= document.getElementById("phone-valid");
    let phoneError= document.getElementById("phone-error");
    if(!phoneField.value.match(/^[7-9][0-9]{9}$/)){
        phoneError.innerHTML= "*Please enter a valid phone number"
        return false;
    }
    phoneError.innerHTML="";
    return true;
}
// for first name
function validname1()
{
    let nameField= document.getElementById("fnamevalid");
    let nameError= document.getElementById("fname-error");
    if(!nameField.value.match(/^[A-Za-z]{2,10}$/)){
        nameError.innerHTML= "* Only Alphabets are allowed";
        return false;
    }
    nameError.innerHTML= "";
    return true;
}


