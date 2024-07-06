import { signInWithPopup } from 'firebase/auth';
import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth, provider } from './FireBase/firebaseConfig';
import './SignIn.css';

function SignIn() {
    const navigate = useNavigate(); 

    useEffect(() => {
        auth.onAuthStateChanged(user => {
            if (user) {
                navigate('/translator');
            }
        });
    }, [navigate]);

    const handleSignIn = () => {
        signInWithPopup(auth, provider)
            .then(() => {
                navigate('/translator'); 
            })
            .catch((error) => {
                console.error('Error signing in: ', error);
                alert('Failed to sign in. Please try again.'); 
            });
    };

    return (
        <div className="signin-container">
            <div className="signin-content">
                <img src="/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" className="logo" />
                <h1>Welcome!</h1>
                <button onClick={handleSignIn} className="google-signin-button">Sign in with Google</button>
            </div>
        </div>
    );
}

export default SignIn;
