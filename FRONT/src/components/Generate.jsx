import React, { Component } from "react";
import { useState, useEffect } from "react";
import { Routes, Route, Outlet, Link } from "react-router-dom";
import GenText from "./GenText.jsx";
import Footer from "./Footer.jsx";
import "/src/styles/Generate.css";
import {
  MouseParallaxChild,
  MouseParallaxContainer,
} from "react-parallax-mouse";

const Generate = () => {
  const [spinner, setSpinner] = useState(true);
  const [imageUrlisFired, setImageUrlisFired] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const API = "http://tomsoft1dev1.northeurope.cloudapp.azure.com:8000/get_planet";

  if (imageUrlisFired === false) {
    setTimeout(() => {
      const time = Date.now();
      setImageUrl(API + "/?data=" + time);
      setImageUrlisFired(true);
      setSpinner(false);
    }, 2000)
  }

  return (
    <div>
      <MouseParallaxContainer
        globalFactorX={0.1}
        globalFactorY={0.1}
        containerStyle={{ overflow: "inherit" }}
      >
        <div className="hero__stars-bg">
          <MouseParallaxChild
            factorX={0.43}
            factorY={-0.3}
            containerStyle={{ zindex: "3", position: "inherit" }}
          >
            <div id="stars"></div>
            <div id="stars2"></div>
            <div id="stars3"></div>
          </MouseParallaxChild>
        </div>

        <div className="App hero__section--container">
          <div className="hero__section--two-columns two-left">
            <GenText />
          </div>
          <div className="hero__section--two-columns two-right">
            {!spinner ? (
              <div>{<img className="generated_image"
                style={{ width: '512px' }}
                src={imageUrl}></img>}</div>

            ) : (
              <div>
                <img src="assets/images/epochs64_3.gif"></img>
                <p style={{position: "relative", bottom: "0" }}>Modeluję i pobieram...</p>
              </div>
            )}
          </div>
        </div>
      </MouseParallaxContainer>
      <Footer />
    </div>
  );
};

export default Generate;
