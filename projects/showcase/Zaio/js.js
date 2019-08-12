//********************Start of colour picker ************************//
function myFunction1() {
  document.getElementById("colour").innerHTML = "Arctic White";
  document.getElementById("modalcolour").innerHTML = "Arctic White";
  }
function myFunction2() {
  document.getElementById("colour").innerHTML = "Honey Dew";
  document.getElementById("modalcolour").innerHTML = "Honey Dew";
  }
function myFunction3() {
  document.getElementById("colour").innerHTML = "Coral";
  document.getElementById("modalcolour").innerHTML = "Coral";
  }
function myFunction4() {
  document.getElementById("colour").innerHTML = "Salmon";
  document.getElementById("modalcolour").innerHTML = "Salmon";
  }
function myFunction5() {
  document.getElementById("colour").innerHTML = "Fuschia";
  document.getElementById("modalcolour").innerHTML = "Fuschia";
  }
function myFunction6() {
  document.getElementById("colour").innerHTML = "Spring Green";
  document.getElementById("modalcolour").innerHTML = "Spring Green";
  }
function myFunction7() {
  document.getElementById("colour").innerHTML = "Rosy Brown";
  document.getElementById("modalcolour").innerHTML = "Rosy Brown";
  }
function myFunction8() {
  document.getElementById("colour").innerHTML = "Silver";
  document.getElementById("modalcolour").innerHTML = "Silver";
  }
function myFunction9() {
  document.getElementById("colour").innerHTML = "Peru";
  document.getElementById("modalcolour").innerHTML = "Peru";
  }
function myFunction10() {
  document.getElementById("colour").innerHTML = "Sienna";
  document.getElementById("modalcolour").innerHTML = "Sienna";
  }
function myFunction11() {
  document.getElementById("colour").innerHTML = "Navy";
  document.getElementById("modalcolour").innerHTML = "Navy";
  }
function myFunction12() {
  document.getElementById("colour").innerHTML = "Black";
  document.getElementById("modalcolour").innerHTML = "Black";
  }
//********************End of colour picker ************************//  


//Global variables for cart quantity
var quantity = 0;
//Set accessibility
var viewMode = getCookie("view-mode");
if(viewMode == "desktop"){
    viewport.setAttribute('content', 'width=1024');
}else if (viewMode == "mobile"){
    viewport.setAttribute('content', 'width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no');
	
}//Function to increase number in cart
function myFunctionadd() {
  quantity = quantity + 1;
  document.getElementById("quantity").innerHTML = quantity;
  document.getElementById("quantity1").innerHTML = quantity;
  }
//Function to decrease number in cart
function myFunctionminus() {
  quantity = quantity - 1;
  
  //Make sure quantity is not less than 0
  if (quantity < 0)
  {quantity = 0;}
  
  document.getElementById("quantity").innerHTML = quantity;
  document.getElementById("quantity1").innerHTML = quantity;
  }
  
function myFunctionAgree() {
  document.getElementById("cart").innerHTML = "Check out now ($"+ (quantity*14.99) +")";
  }
  
function myFunctionCancel() {
  quantity = 0;
  document.getElementById("quantity").innerHTML = quantity;
  document.getElementById("quantity1").innerHTML = quantity;
  }
  
