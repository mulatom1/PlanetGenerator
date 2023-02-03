import React, { Component } from 'react'
import { useState, useEffect } from 'react';
import { Routes, Route, Outlet, Link } from "react-router-dom";
import GenText from './GenText.jsx';
import Footer from './Footer.jsx';
import '/src/styles/Generate.css'
import {
    MouseParallaxChild,
    MouseParallaxContainer
  } from "react-parallax-mouse";

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

   const FolderImages = [
    '/assets/images/examples/1.jpg',
    '/assets/images/examples/2.jpg',
    '/assets/images/examples/3.jpg',
    '/assets/images/examples/4.jpg',
    '/assets/images/examples/5.jpg',
    '/assets/images/examples/6.jpg',
    '/assets/images/examples/7.jpg',
    '/assets/images/examples/8.jpg',
    '/assets/images/examples/9.jpg',
    '/assets/images/examples/10.jpg',
    '/assets/images/examples/11.jpg',
    '/assets/images/examples/12.jpg',
    '/assets/images/examples/13.jpg',
    '/assets/images/examples/14.jpg',
    '/assets/images/examples/15.jpg',
    '/assets/images/examples/16.jpg',
    '/assets/images/examples/17.jpg',
    '/assets/images/examples/18.jpg',
    '/assets/images/examples/19.jpg',
    '/assets/images/examples/20.jpg',
    '/assets/images/examples/21.jpg',
    '/assets/images/examples/22.jpg',
    '/assets/images/examples/23.jpg',
    '/assets/images/examples/24.jpg',
    '/assets/images/examples/25.jpg',
    '/assets/images/examples/26.jpg',
    '/assets/images/examples/27.jpg',
    '/assets/images/examples/28.jpg',
    '/assets/images/examples/29.jpg',
    '/assets/images/examples/30.jpg'
   ]
    const [currentImageIndex, setCurrentImageIndex] = useState(Math.floor(Math.random() * FolderImages.length))

    const changeImage = () => {
    const randomNumber = Math.floor(Math.random() * FolderImages.length);
      setCurrentImageIndex(randomNumber);
    }
    useEffect(() => changeImage(), [])


    React.useEffect(() => {
    setTimeout(() => setSpinner(false), 6000)

      });

    return <div>
         <MouseParallaxContainer globalFactorX={0.1} globalFactorY={0.1} containerStyle={{overflow: "inherit",}}>
            <div className="hero__stars-bg">
            <MouseParallaxChild factorX={.43} factorY={-0.3} containerStyle={{zindex: "3",
      position:"inherit"}}>
                <div id='stars'></div>
                <div id='stars2'></div>
                <div id='stars3'></div>
                </MouseParallaxChild>
            </div>
            
            
            <div className="App hero__section--container">
                <div className="hero__section--two-columns two-left">
                    <GenText/>
                </div>
                <div className="hero__section--two-columns two-right">
                {!spinner 
            ? <div>
                {loading ? null :
                    <div>
                        <img className="generated_image" src={FolderImages[currentImageIndex]}/>
                        <p style={{position:"relative", bottom:"0",  width:'500px'}}></p>
                    </div>
                }
                  
              </div>
            : <div>
                <img src="/assets/images/epochs64_3.gif"></img>
                <p style={{position:"relative", bottom:"0"}}>Modeluję...</p>
             </div>}

              
            {/* <img
            style={loading ? {} : {display: 'none'}}
            src={imageUrl}
            onLoad={() => setLoading(true)}
            /> */}
                </div>
            </div>
            </MouseParallaxContainer>
            <Footer/>
           </div>
        }

    export default Generate;


    

