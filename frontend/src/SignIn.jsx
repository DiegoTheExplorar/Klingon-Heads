import { signInWithPopup } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import Translator from './Translator';
import { auth, provider } from './firebaseConfig';

function SignIn() {
    const [email, setEmail] = useState('');
    const navigate = useNavigate(); // Hook for programmatic navigation

    useEffect(() => {
        const storedEmail = localStorage.getItem('email');
        if (storedEmail) {
            setEmail(storedEmail);
            navigate('/translator'); // Navigate to the translator page if already signed in
        }
    }, [navigate]); // Add navigate to the dependency array to ensure updated navigation behavior

    const handleSignIn = () => {
        signInWithPopup(auth, provider).then((result) => {
            const userEmail = result.user.email;
            setEmail(userEmail);
            localStorage.setItem('email', userEmail);
            navigate('/translator'); // Navigate after successful sign-in
        }).catch((error) => {
            console.error('Error signing in: ', error);
        });
    };

    return (
        <div>
            {email ? <Translator /> : ( // This conditional rendering might be redundant if you navigate away on sign-in
                <button onClick={handleSignIn}>Sign in with Google</button>
            )}
        </div>
    );
}

export default SignIn;
