import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const auth = getAuth(); 

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      setIsLoggedIn(!!user);
      if(isLoggedIn) console.log('Logged in');
      else console.log('Not Logged in');
    });

    return () => unsubscribe();
  }, []);  

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-800 text-white">
      <h1 className="text-4xl font-bold">Welcome to Klingon to English Translator</h1>
      <p className="text-xl mt-4">Translate your Klingon phrases to English!</p>
      {isLoggedIn ? (
        <Link to="/translator">
          <button className="mt-8 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
            Translate Now
          </button>
        </Link>
      ) : (
        <Link to="/signin">
          <button className="mt-8 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
            Sign In
          </button>
        </Link>
      )}
    </div>
  );
};

export default LandingPage;
