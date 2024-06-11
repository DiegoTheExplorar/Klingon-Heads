import { signInWithPopup } from 'firebase/auth';
import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth, provider } from './firebaseConfig';

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
        <div>
            <button onClick={handleSignIn}>Sign in with Google</button>
        </div>
    );
}

export default SignIn;
