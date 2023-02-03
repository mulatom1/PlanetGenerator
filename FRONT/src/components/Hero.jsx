import React, { Component } from 'react'
import '/src/styles/hero.css'
import { Routes, Route, Outlet, Link } from "react-router-dom";
export default class Hero extends Component {
  render() {
    return (
      <div>
        
        <div className="hero__section">
      <h1 className="hero__title">PLANET<br/>GENERATOR</h1>
      <p className="hero__subtitle">Projekt końcowy kursu SDA Data Science</p>
      <br/>
      <Link to="/generate" className="btn btn-primary button" >WYGENERUJ PLANETĘ</Link>
    
  </div>

      </div>
    )
  }
}
