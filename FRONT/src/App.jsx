import Hero from './components/Hero.jsx'
import Footer from './components/Footer.jsx'
import { React, useEffect, useState } from "react";
import Generate from './components/Generate.jsx'
import {
  MouseParallaxChild,
  MouseParallaxContainer
} from "react-parallax-mouse";
import { Routes, Route, Outlet, Link } from "react-router-dom";

import './App.css'

export default function App() {
  return (
    <div>
      {/* Routes nest inside one another. Nested route paths build upon
            parent route paths, and nested route elements render inside
            parent route elements. See the note about <Outlet> below. */}
            
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="generate" element={<Generate />} />
          {/* Using path="*"" means "match anything", so this route
                acts like a catch-all for URLs that we don't have explicit
                routes for. */}
          <Route path="*" element={<NoMatch />} />
          <Route path="ania" element={<Ania />} />
        </Route>
      </Routes>
    </div>
  );
}
function Home() {
  return (
    <div className="App"> 
    <MouseParallaxContainer globalFactorX={0.1} globalFactorY={0.1} containerStyle={{overflow: "inherit",}}>
       <div className="hero__stars-bg">
      <MouseParallaxChild factorX={.43} factorY={-0.3} containerStyle={{zindex: "3",
      position:"inherit"}}>
        <div id='stars'></div>
        <div id='stars2'></div>
        <div id='stars3'></div>
        </MouseParallaxChild>
      </div>
      
       <div>
     
        <MouseParallaxChild factorX={0.43} factorY={-0.3} containerStyle={{zindex: "3",
      position:"inherit"}}>
          <img className="image-planet" src="/assets/images/planet.png"/>
          </MouseParallaxChild>

          <MouseParallaxChild factorX={-2} factorY={0.5} containerStyle={{zindex: "3",
      position:"inherit"}}>
       
          <img className="image-ship" src="/assets/images/ship.png"/>
        </MouseParallaxChild>
      
      <Hero/>
      </div>
     
      </MouseParallaxContainer>
    <Footer/>
    </div>
  );
}

function Layout() {
  return (
    <div>
      {/* An <Outlet> renders whatever child route is currently active,
          so you can think about this <Outlet> as a placeholder for
          the child routes we defined above. */}
      <Outlet />
    </div>
  );
}
  



function NoMatch() {
  return (
    <div>
      <h2>Nic tu nie ma</h2>
      <p>
        <Link to="/">Wróć do strony głównej</Link>
      </p>
    </div>
  );
}

function Ania() {
  return (
    <div>
      <h2>aniapage</h2>
      <p>
        
      </p>
    </div>
  );
}
