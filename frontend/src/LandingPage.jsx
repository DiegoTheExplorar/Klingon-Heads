import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

const LandingPage = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const auth = getAuth();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      setIsLoggedIn(!!user); // Update state here directly
    });

    return () => unsubscribe(); // Clean up the subscription
  }, []);

  return (
    <div className="landing">
      <h1 className="title">Welcome to Klingon to English Translator</h1>
      <p className="subtitle">Translate your Klingon phrases to English!</p>
      {isLoggedIn ? (
        <Link to="/translator">
          <button className="button">Translate Now</button>
        </Link>
      ) : (
        <Link to="/signin">
          <button className="button">Sign In</button>
        </Link>
      )}
    </div>
  );
};

export default LandingPage;
