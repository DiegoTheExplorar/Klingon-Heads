import { signInWithPopup } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Translator from './Translator';
import { auth, provider } from './firebaseConfig';
import './SignIn.css';

function SignIn() {
    const [email, setEmail] = useState('');
    const navigate = useNavigate(); 

    useEffect(() => {
        const storedEmail = localStorage.getItem('email');
        if (storedEmail) {
            setEmail(storedEmail);
            navigate('/translator');
        }
    }, [navigate]);

    const handleSignIn = () => {
        signInWithPopup(auth, provider).then((result) => {
            const userEmail = result.user.email;
            setEmail(userEmail);
            localStorage.setItem('email', userEmail);
            navigate('/translator'); 
        }).catch((error) => {
            console.error('Error signing in: ', error);
        });
    };

    return (
        <div className="signin-container">
            {email ? <Translator /> : (
                <div className="signin-content">
                    <img src="/Klingon-Heads-Logo.png" alt="Klingon Heads Logo" className="logo" />
                    <h1>Welcome!</h1>
                    <button onClick={handleSignIn} className="google-signin-button">Sign in with Google</button>
                </div>
            )}
        </div>
    );
}

export default SignIn;
