import firebase from 'firebase/app';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from './firebaseConfig';

export const Auth = () => {
  const navigate = useNavigate();

  const signInWithGoogle = () => {
    const provider = new firebase.auth.GoogleAuthProvider();
    auth.signInWithPopup(provider)
      .then((result) => {
        // This gives you a Google Access Token. You can use it to access the Google API.
        console.log(result);
        navigate('/translator'); // Redirect to Translator page after authentication
      }).catch((error) => {
        console.log(error);
      });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-800 text-white">
      <button onClick={signInWithGoogle} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Sign in with Google
      </button>
    </div>
  );
};

export default Auth;
