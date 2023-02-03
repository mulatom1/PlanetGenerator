import React, { Component } from 'react'
import { useState, useEffect } from 'react';
import { Routes, Route, Outlet, Link } from "react-router-dom";
import GenText from './GenText.jsx';
import Footer from './Footer.jsx';
import '/src/styles/Generate.css'

const Generate = () => {
    //  React.useEffect(() => {
    //     console.log('MyComponent onMount');
    //     return () => {
    //       console.log('MyComponent onUnmount');
    //     };
    // }, []);
    //   function getRandomInt() {
    // console.log(Math.floor(Math.random() * 1000))
    //   }
    //   getRandomInt()
    // const [data, setData] = React.useState([])
    const [spinner, setSpinner] = useState(true);
    const [loading, setLoading] = useState(false);

        const imageUrl = "http://tomsoft1dev1.northeurope.cloudapp.azure.com:8000/get_planet";
  React.useEffect(() => {
    setTimeout(() => setSpinner(false), 10000)
      });


    return <div>
            <div className="hero__stars-bg">
                <div id='stars'></div>
                <div id='stars2'></div>
                <div id='stars3'></div>
            </div>
            <div className="App hero__section--container">
            <div className="hero__section--two-columns two-left">
                <GenText/>
            </div>
            <div className="hero__section--two-columns two-right">

            {!spinner 
            ? <div>
                {<img src={imageUrl}></img>}
              </div>
            : <div>
                <img src="src/assets/images/epochs64_3.gif"></img>
                <p style={{position:"relative", bottom:"0"}}>Modeluję...</p>
             </div>}
             </div>
            </div>
            <Footer/>
           </div>
        }

    export default Generate;


    

