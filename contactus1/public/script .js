var emailField= document.getElementById("email-field");
var emailLabel= document.getElementById("email-label");
var emailError= document.getElementById("email-error");

function validateEmail(){
    if(!emailField.ariaValueMax.match(/^[A-za-z\._\-0-9]*[@][A-Za-z]*[\.][a-z]{2,4}$/)){
        emailError.innerHTML= "Please enter a valid email";
        return false;
    }
    emailError.innerHTML= "";
    return true;

}  