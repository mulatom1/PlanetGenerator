import React, { Component } from 'react'
import '/src/styles/GenText.css'
import { Routes, Route, Outlet, Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";

const GenText = () => {
    // function refreshPage() {
    //     window.location.reload(false);
    //   }

    
const navigate = useNavigate();

const refreshPage = () => {
    navigate(0);
}
{
    return (
      <div>
        <div className="hero__section">
            <h1 className="hero__title">TWOJA<br/>PLANETA:</h1>
            <p className="hero__subtitle">Projekt końcowy kursu SDA Data Science</p>
            <br/>
            {/* <a onClick={refreshPage} className="btn btn-primary button" >WYGENERUJ NOWĄ PLANETĘ</a> */}
            <Link to="/" className="btn btn-primary button" >WRÓĆ</Link>
            
            <br/>
            {/* <Link to="/" className="btn btn-primary button" >ZAPISZ PLANETĘ</Link> */}
        </div>
      </div>
    )
  }
}
export default GenText;
