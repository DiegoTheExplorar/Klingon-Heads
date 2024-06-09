import React from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-800 text-white">
      <h1 className="text-4xl font-bold">Welcome to Klingon to English Translator</h1>
      <p className="text-xl mt-4">Translate your Klingon phrases to English!</p>
      <Link to="/translator">
        <button className="mt-8 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
          Translate Now
        </button>
      </Link>
    </div>
  );
};

export default LandingPage;
