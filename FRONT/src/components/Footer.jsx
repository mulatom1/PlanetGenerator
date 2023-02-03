import React, { Component } from 'react'

export default class Footer extends Component {
  render() {
    return (
      <div style={{display:"flex",
      flexDirection:"column",
      justifyContent: "flex-end",
      height: "inherit"
      }}>
        <div className="footer__content" style={{margin: "2rem 2rem", marginLeft:"10rem", }}>
        Copyright by Piotr Pawłowski & Tomasz Mularczyk. 2023
        </div>
        </div>
    )
  }
}
